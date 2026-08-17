from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, load_dataset_builder

from jlm.data import write_jsonl_documents
from jlm.utils import write_json

DATASET_METADATA = {
    "wikitext": {
        "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "note": "The dataset card also references the original WikiText source and citation.",
    },
    "ag_news": {
        "source_url": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "license": "Unknown on the dataset card",
        "license_url": None,
        "note": (
            "The card describes the source as intended for research and non-commercial activity. "
            "Check the upstream terms before using it outside this project."
        ),
    },
}


def deduplicate_splits(train: list[str], validation: list[str]) -> tuple[list[str], list[str]]:
    validation_unique = list(dict.fromkeys(text.strip() for text in validation if text.strip()))
    validation_set = set(validation_unique)
    train_unique = list(
        dict.fromkeys(
            text.strip() for text in train if text.strip() and text.strip() not in validation_set
        )
    )
    return train_unique, validation_unique


def download_one(
    dataset_name: str,
    config_name: str | None,
    output_dir: Path,
    max_train_documents: int | None,
    max_validation_documents: int | None,
) -> dict:
    builder = (
        load_dataset_builder(dataset_name, config_name)
        if config_name
        else load_dataset_builder(dataset_name)
    )
    dataset = load_dataset(dataset_name, config_name) if config_name else load_dataset(dataset_name)
    validation_name = "validation" if "validation" in dataset else "test"
    train = [text for text in dataset["train"]["text"] if text and text.strip()]
    validation = [text for text in dataset[validation_name]["text"] if text and text.strip()]
    train, validation = deduplicate_splits(train, validation)
    if max_train_documents is not None:
        train = train[:max_train_documents]
    if max_validation_documents is not None:
        validation = validation[:max_validation_documents]

    dataset_dir = output_dir / dataset_name
    write_jsonl_documents(dataset_dir / "train.jsonl", train)
    write_jsonl_documents(dataset_dir / "validation.jsonl", validation)
    metadata = DATASET_METADATA[dataset_name]
    license_name = getattr(builder.info, "license", None) or metadata["license"]
    manifest = {
        "dataset": dataset_name,
        "config": config_name,
        "source": metadata["source_url"],
        "license": license_name,
        "license_url": metadata["license_url"],
        "license_note": metadata["note"],
        "train_split": "train",
        "validation_split_source": validation_name,
        "train_documents": len(train),
        "validation_documents": len(validation),
        "validation_deduplicated_against_train": True,
        "raw_data_is_not_committed": True,
    }
    write_json(dataset_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public corpora used by the experiments."
    )
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--max-train-documents", type=int, default=None)
    parser.add_argument("--max-validation-documents", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifests = [
        download_one(
            "wikitext",
            "wikitext-2-raw-v1",
            output_dir,
            args.max_train_documents,
            args.max_validation_documents,
        ),
        download_one(
            "ag_news",
            None,
            output_dir,
            args.max_train_documents,
            args.max_validation_documents,
        ),
    ]
    write_json(output_dir / "manifest.json", {"datasets": manifests})
    for manifest in manifests:
        print(manifest)


if __name__ == "__main__":
    main()
