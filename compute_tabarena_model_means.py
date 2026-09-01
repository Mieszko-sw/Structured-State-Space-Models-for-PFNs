"""Compute per-model means across the TabArena 16-dataset result files."""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = [
    "result_csvs/tabarena_16.csv",
    "result_csvs/tabarena_3.csv",
    "result_csvs/tabarena_2.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine TabArena result CSVs and compute the mean inference time "
            "and mean metric across datasets for every model."
        )
    )
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Per-dataset summary CSVs to combine.",
    )
    parser.add_argument(
        "--output-csv",
        default="result_csvs/tabarena_all_models_means.csv",
        help="Destination for the per-model means.",
    )
    return parser.parse_args()


def read_results(paths):
    required_columns = {"did", "model", "inference_mean", "mean_metric"}
    frames = []

    for path in paths:
        input_path = Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

        frame = pd.read_csv(input_path)
        missing_columns = sorted(required_columns.difference(frame.columns))
        if missing_columns:
            raise ValueError(
                f"{input_path} is missing required columns: "
                + ", ".join(missing_columns)
            )

        frame = frame.copy()
        frame["source_csv"] = str(input_path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    duplicate_rows = combined.duplicated(["did", "model"], keep=False)
    if duplicate_rows.any():
        duplicates = (
            combined.loc[duplicate_rows, ["did", "model", "source_csv"]]
            .sort_values(["model", "did", "source_csv"])
            .to_string(index=False)
        )
        raise ValueError(
            "Duplicate dataset/model rows found across the input files:\n" + duplicates
        )

    combined["inference_mean"] = pd.to_numeric(
        combined["inference_mean"], errors="coerce"
    )
    combined["mean_metric"] = pd.to_numeric(combined["mean_metric"], errors="coerce")
    return combined


def compute_model_means(data):
    return (
        data.groupby("model", as_index=False)
        .agg(
            mean_inference=("inference_mean", "mean"),
            mean_metric=("mean_metric", "mean"),
            num_datasets=("did", "nunique"),
        )
        .sort_values("mean_inference", ignore_index=True)
    )


def main():
    args = parse_args()
    summary = compute_model_means(read_results(args.input_csvs))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved per-model means to {output_path}")


if __name__ == "__main__":
    main()
