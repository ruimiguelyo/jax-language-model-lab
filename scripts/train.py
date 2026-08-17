from __future__ import annotations

import argparse
from pathlib import Path

import jax

from jlm.checkpoint import restore_checkpoint, save_checkpoint
from jlm.config import ExperimentConfig, config_dict, load_config, save_config
from jlm.data import cycling_batches, documents_to_blocks, iter_batches, read_jsonl_documents
from jlm.metrics import append_jsonl
from jlm.model import count_parameters
from jlm.tokenizer import FixedTokenizer
from jlm.training import create_train_state, evaluate, to_jax_batch, train_step
from jlm.utils import environment_info, write_json


def load_split(data_dir: Path, dataset_name: str, split: str) -> list[str]:
    path = data_dir / dataset_name / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_data.py` before training."
        )
    return read_jsonl_documents(path)


def build_blocks(config: ExperimentConfig, data_dir: Path):
    tokenizer = FixedTokenizer.from_pretrained(
        config.data.tokenizer_name,
        cache_dir=config.data.cache_dir,
    )
    if config.model.vocab_size != tokenizer.vocab_size:
        config.model.vocab_size = tokenizer.vocab_size
    config.model.max_seq_len = config.data.seq_len
    general_train = load_split(data_dir, "wikitext", "train")
    general_validation = load_split(data_dir, "wikitext", "validation")
    train_blocks = documents_to_blocks(general_train, tokenizer, config.data.seq_len).complete()
    validation_blocks = documents_to_blocks(general_validation, tokenizer, config.data.seq_len)
    return tokenizer, train_blocks, validation_blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the decoder-only Transformer on the general corpus."
    )
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="results/pretrain")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, train_blocks, validation_blocks = build_blocks(config, Path(args.data_dir))
    target_steps = args.total_steps or config.train.total_steps
    if target_steps > config.train.total_steps:
        config.train.total_steps = target_steps
    save_config(config, output_dir / "config.yaml")
    write_json(output_dir / "environment.json", environment_info())

    _, state = create_train_state(config.model, config.train, config.train.seed)
    parameter_count = count_parameters(state.params)
    write_json(
        output_dir / "model.json",
        {"parameters": parameter_count, "vocab_size": tokenizer.vocab_size},
    )
    metrics_path = output_dir / "metrics.jsonl"

    if args.resume:
        state, _ = restore_checkpoint(state, checkpoint_dir)
    else:
        save_checkpoint(
            state, checkpoint_dir, metadata={"phase": "initial", "config": config_dict(config)}
        )
        initial_metrics = evaluate(
            state,
            iter_batches(
                validation_blocks,
                config.train.batch_size,
                config.train.seed,
                shuffle=False,
                drop_remainder=False,
            ),
        )
        append_jsonl(metrics_path, {"step": 0, "phase": "validation", **initial_metrics})
        print(
            f"step=0 validation_loss={initial_metrics['loss']:.4f} "
            f"perplexity={initial_metrics['perplexity']:.4f}"
        )

    batches = cycling_batches(
        train_blocks, config.train.batch_size, config.train.seed, shuffle=True
    )
    start_step = int(state.step)
    for step in range(start_step + 1, target_steps + 1):
        state, raw_metrics = train_step(state, to_jax_batch(next(batches)))
        jax.block_until_ready(raw_metrics["loss"])
        if step % config.train.log_every == 0 or step == target_steps:
            record = {
                "step": step,
                "phase": "train",
                "loss": float(raw_metrics["loss"]),
                "perplexity": float(raw_metrics["perplexity"]),
                "grad_norm": float(raw_metrics["grad_norm"]),
                "token_count": float(raw_metrics["token_count"]),
            }
            append_jsonl(metrics_path, record)
            print(
                f"step={step} train_loss={record['loss']:.4f} perplexity={record['perplexity']:.4f}"
            )
        if step % config.train.eval_every == 0 or step == target_steps:
            evaluation = evaluate(
                state,
                iter_batches(
                    validation_blocks,
                    config.train.batch_size,
                    config.train.seed,
                    shuffle=False,
                    drop_remainder=False,
                ),
            )
            append_jsonl(metrics_path, {"step": step, "phase": "validation", **evaluation})
            print(
                f"step={step} validation_loss={evaluation['loss']:.4f} "
                f"perplexity={evaluation['perplexity']:.4f}"
            )
        if step % config.train.checkpoint_every == 0 or step == target_steps:
            save_checkpoint(state, checkpoint_dir, metadata={"phase": "pretraining", "step": step})

    final_metrics = evaluate(
        state,
        iter_batches(
            validation_blocks,
            config.train.batch_size,
            config.train.seed,
            shuffle=False,
            drop_remainder=False,
        ),
    )
    write_json(
        output_dir / "summary.json",
        {
            "step": int(state.step),
            "parameters": parameter_count,
            "validation": final_metrics,
            "seed": config.train.seed,
            "dataset": config.data.general_dataset,
            "tokenizer": config.data.tokenizer_name,
        },
    )


if __name__ == "__main__":
    main()
