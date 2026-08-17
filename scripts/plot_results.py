from __future__ import annotations

import argparse

from jlm.metrics import plot_continual_metrics, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot continual-pretraining metrics from JSONL records."
    )
    parser.add_argument("metrics", help="Path to continual metrics.jsonl")
    parser.add_argument("--output", default="results/continual/figures")
    args = parser.parse_args()
    paths = plot_continual_metrics(read_jsonl(args.metrics), args.output)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
