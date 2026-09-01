"""Compute THx4 mean inference time and metric across evaluated datasets."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Average the per-dataset inference_mean and mean_metric values in "
            "the synchronized THx4 summary CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="result_csvs/per_dataset_speed_synchronized_thx4_summary.csv",
        help="Input per-dataset THx4 summary CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "result_csvs/"
            "per_dataset_speed_synchronized_thx4_model_means.csv"
        ),
        help="Destination for the aggregated THx4 means.",
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

    valid_rows = data["inference_mean"].notna() & data["mean_metric"].notna()
    invalid_row_count = int((~valid_rows).sum())
    data = data.loc[valid_rows]
    if data.empty:
        raise ValueError("Input CSV contains no valid inference/metric rows.")

    summary = (
        data.groupby("model", as_index=False)
        .agg(
            mean_inference_time_seconds=("inference_mean", "mean"),
            mean_metric=("mean_metric", "mean"),
            num_datasets=("did", "nunique"),
            num_rows=("model", "size"),
        )
        .sort_values("model", ignore_index=True)
    )
    return summary, invalid_row_count


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    summary, invalid_row_count = compute_model_means(pd.read_csv(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    if invalid_row_count:
        print(f"Ignored {invalid_row_count} row(s) with invalid values.\n")
    print(summary.to_string(index=False))
    print(f"\nSaved THx4 model means to {output_path}")


if __name__ == "__main__":
    main()
