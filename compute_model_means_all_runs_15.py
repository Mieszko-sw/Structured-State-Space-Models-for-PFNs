"""Compute per-model mean inference time and metric from all_runs_15.csv."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Average the per-dataset inference_mean and mean_metric values for "
            "each model in all_runs_15.csv."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="result_csvs/all_runs_15.csv",
        help="Input per-dataset summary CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="result_csvs/all_runs_15_model_means.csv",
        help="Destination for the per-model means.",
    )
    return parser.parse_args()


def compute_model_means(data):
    required_columns = {"model", "did", "inference_mean", "mean_metric"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing_columns)
        )

    data = data.copy()
    data["inference_mean"] = pd.to_numeric(data["inference_mean"], errors="coerce")
    data["mean_metric"] = pd.to_numeric(data["mean_metric"], errors="coerce")

    return (
        data.groupby("model", as_index=False)
        .agg(
            mean_inference_time_seconds=("inference_mean", "mean"),
            mean_metric=("mean_metric", "mean"),
            num_datasets=("did", "nunique"),
            num_rows=("model", "size"),
        )
        .sort_values("mean_inference_time_seconds", ignore_index=True)
    )


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    summary = compute_model_means(pd.read_csv(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved per-model means to {output_path}")


if __name__ == "__main__":
    main()
