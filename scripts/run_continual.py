from __future__ import annotations

import argparse
from pathlib import Path

from jlm.checkpoint import restore_checkpoint, save_checkpoint
from jlm.config import config_dict, load_config, save_config
from jlm.continual.reservoir import build_reservoir
from jlm.continual.strategies import strategy_blocks
from jlm.data import (
    TokenBlocks,
    cycling_batches,
    documents_to_blocks,
    iter_batches,
    read_jsonl_documents,
)
from jlm.metrics import append_jsonl, build_continual_record, plot_continual_metrics
from jlm.tokenizer import FixedTokenizer
from jlm.training import create_train_state, evaluate, train_steps
from jlm.utils import environment_info, write_json

QUESTION = (
    "Can replay reduce catastrophic forgetting during continual pretraining "
    "under a fixed token budget?"
)


def load_blocks(
    data_dir: Path, dataset_name: str, split: str, tokenizer, seq_len: int
) -> TokenBlocks:
    path = data_dir / dataset_name / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_data.py` before running experiments."
        )
    documents = read_jsonl_documents(path)
    return documents_to_blocks(documents, tokenizer, seq_len)


def evaluate_domains(state, general_validation, target_validation, batch_size: int, seed: int):
    general_metrics = evaluate(
        state,
        iter_batches(general_validation, batch_size, seed, shuffle=False, drop_remainder=False),
    )
    target_metrics = evaluate(
        state,
        iter_batches(target_validation, batch_size, seed, shuffle=False, drop_remainder=False),
    )
    return general_metrics, target_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-budget continual-pretraining experiment."
    )
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--pretrain-output", default="results/pretrain")
    parser.add_argument("--output", default="results/continual")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = FixedTokenizer.from_pretrained(
        config.data.tokenizer_name, cache_dir=config.data.cache_dir
    )
    config.model.vocab_size = tokenizer.vocab_size
    config.model.max_seq_len = config.data.seq_len
    save_config(config, output_dir / "config.yaml")
    write_json(output_dir / "environment.json", environment_info())
    write_json(
        output_dir / "experiment.json",
        {
            "question": QUESTION,
            "configuration": config_dict(config),
        },
    )

    data_dir = Path(args.data_dir)
    general_train = load_blocks(
        data_dir, "wikitext", "train", tokenizer, config.data.seq_len
    ).complete()
    general_validation = load_blocks(
        data_dir, "wikitext", "validation", tokenizer, config.data.seq_len
    )
    target_train = load_blocks(
        data_dir, "ag_news", "train", tokenizer, config.data.seq_len
    ).complete()
    target_validation = load_blocks(
        data_dir, "ag_news", "validation", tokenizer, config.data.seq_len
    )

    if config.continual.adaptation_tokens % (config.data.seq_len * config.train.batch_size) != 0:
        raise ValueError(
            "adaptation_tokens must be divisible by seq_len * batch_size "
            "for an exact fixed-budget run"
        )
    total_steps = config.continual.adaptation_tokens // (
        config.data.seq_len * config.train.batch_size
    )
    fractions = sorted(set(config.continual.eval_fractions))
    if fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("eval_fractions must include 0.0 and 1.0")

    _, baseline_state = create_train_state(config.model, config.train, config.train.seed)
    baseline_state, checkpoint_metadata = restore_checkpoint(
        baseline_state,
        Path(args.pretrain_output) / "checkpoints",
    )
    write_json(output_dir / "baseline_checkpoint.json", checkpoint_metadata)
    reservoir = build_reservoir(
        general_train,
        capacity=config.continual.reservoir_capacity,
        seed=config.train.seed,
    )
    write_json(
        output_dir / "reservoir.json",
        {
            "capacity": reservoir.capacity,
            "stored_examples": len(reservoir),
            "stream_examples_seen": reservoir.seen,
            "replay_ratio": config.continual.replay_ratio,
            "updates": (
                "The reservoir is built from general-domain training blocks before adaptation "
                "and is not updated during adaptation."
            ),
        },
    )

    baseline_general, baseline_target = evaluate_domains(
        baseline_state,
        general_validation,
        target_validation,
        config.train.batch_size,
        config.train.seed,
    )
    metrics_path = output_dir / "metrics.jsonl"
    records: list[dict] = []
    strategies = ["sequential", "reservoir_replay", "joint_training_reference"]
    for strategy_index, strategy in enumerate(strategies):
        strategy_data = strategy_blocks(
            strategy=strategy,
            general_blocks=general_train,
            target_blocks=target_train,
            reservoir=reservoir,
            total_examples=config.continual.adaptation_tokens // config.data.seq_len,
            replay_ratio=config.continual.replay_ratio,
            joint_target_ratio=config.continual.joint_target_ratio,
            seed=config.train.seed + strategy_index,
        )
        if strategy_data.num_examples % config.train.batch_size != 0:
            raise ValueError("Strategy data must contain complete batches")
        batches = cycling_batches(
            strategy_data,
            config.train.batch_size,
            seed=config.train.seed + strategy_index,
            shuffle=False,
        )
        state = baseline_state
        previous_steps = 0
        for fraction in fractions:
            requested_steps = int(round(total_steps * fraction))
            if requested_steps > previous_steps:
                state, _ = train_steps(state, batches, requested_steps - previous_steps)
                previous_steps = requested_steps
            general_metrics, target_metrics = evaluate_domains(
                state,
                general_validation,
                target_validation,
                config.train.batch_size,
                config.train.seed,
            )
            record = build_continual_record(
                strategy=strategy,
                adaptation_tokens=requested_steps * config.train.batch_size * config.data.seq_len,
                general_metrics=general_metrics,
                target_metrics=target_metrics,
                baseline_general_loss=baseline_general["loss"],
            )
            record["fraction"] = fraction
            record["seed"] = config.train.seed
            record["batch_size"] = config.train.batch_size
            record["sequence_length"] = config.data.seq_len
            append_jsonl(metrics_path, record)
            records.append(record)
            print(
                f"strategy={strategy} fraction={fraction:.2f} "
                f"general_ppl={record['general_perplexity']:.4f} "
                f"target_ppl={record['target_perplexity']:.4f} "
                f"forgetting={record['forgetting']:.6f}"
            )
        save_checkpoint(
            state,
            output_dir / strategy / "checkpoints",
            metadata={"strategy": strategy, "seed": config.train.seed},
        )

    write_json(
        output_dir / "references.json",
        {
            "strategy": "no_adaptation",
            "adaptation_tokens": 0,
            "general": baseline_general,
            "target": baseline_target,
            "definition": "The pre-adaptation checkpoint is evaluated without additional training.",
        },
    )
    plot_continual_metrics(records, output_dir / "figures")
    write_json(
        output_dir / "summary.json",
        {
            "question": QUESTION,
            "strategies": strategies,
            "records": records,
            "seed_count": 1,
            "note": "These are single-seed exploratory results unless additional runs are added.",
        },
    )


if __name__ == "__main__":
    main()
