"""Consolidate compatible synchronized per-dataset model evaluations."""

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("result_csvs")
OUTPUT_PATH = RESULTS_DIR / "elastic_looped_and_tabpfn_per_dataset.csv"

MODEL_SOURCES = [
    {
        "source": RESULTS_DIR
        / "per_dataset_speed_synchronized_elastic_looped_transformer_summary.csv",
        "source_model": "elastic_loopedx12",
        "model": "elastic_looped_1physical_core12x",
        "model_family": "elastic_looped",
        "physical_layers": 1,
        "max_loops": 12,
        "effective_layers": 12,
    },
    {
        "source": RESULTS_DIR
        / "per_dataset_speed_synchronized_elastic_looped_vs_tabpfn_summary.csv",
        "source_model": "elastic_looped_3physical_core6x",
        "model": "elastic_looped_3physical_core6x",
        "model_family": "elastic_looped",
        "physical_layers": 3,
        "max_loops": 6,
        "effective_layers": 18,
    },
    {
        "source": RESULTS_DIR
        / "per_dataset_speed_synchronized_loopedx6_x12_hybrid8l_tabpfn_summary.csv",
        "source_model": "loopedx6",
        "model": "looped_1physical_core6x",
        "model_family": "looped",
        "physical_layers": 1,
        "max_loops": 6,
        "effective_layers": 6,
    },
    {
        "source": RESULTS_DIR
        / "per_dataset_speed_synchronized_loopedx6_x12_hybrid8l_tabpfn_summary.csv",
        "source_model": "loopedx12",
        "model": "looped_1physical_core12x",
        "model_family": "looped",
        "physical_layers": 1,
        "max_loops": 12,
        "effective_layers": 12,
    },
    {
        "source": RESULTS_DIR / "tabpfn_12l_vs_looped_18x_synchronized_per_dataset.csv",
        "source_model": "looped_18x",
        "model": "looped_1physical_core18x",
        "model_family": "looped",
        "physical_layers": 1,
        "max_loops": 18,
        "effective_layers": 18,
    },
    {
        "source": RESULTS_DIR
        / "per_dataset_speed_synchronized_elastic_looped_vs_tabpfn_summary.csv",
        "source_model": "tabpfn",
        "model": "tabpfn",
        "model_family": "tabpfn",
        "physical_layers": 12,
        "max_loops": 1,
        "effective_layers": 12,
    },
]

OUTPUT_COLUMNS = [
    "did",
    "dataset_name",
    "model",
    "model_family",
    "physical_layers",
    "max_loops",
    "effective_layers",
    "num_samples",
    "num_features",
    "configured_eval_position",
    "real_eval_position",
    "mean_auc_roc",
    "inference_mean_seconds",
    "inference_median_seconds",
    "inference_std_seconds",
    "inference_min_seconds",
    "inference_max_seconds",
    "end_to_end_mean_seconds",
    "end_to_end_median_seconds",
    "measurement_count",
    "source_csv",
]


def load_model_results(info):
    frame = pd.read_csv(info["source"])
    frame = frame.loc[frame["model"] == info["source_model"]].copy()
    if frame.empty:
        raise ValueError(
            f"No rows for {info['source_model']} in {info['source']}"
        )

    frame["model"] = info["model"]
    for column in (
        "model_family",
        "physical_layers",
        "max_loops",
        "effective_layers",
    ):
        frame[column] = info[column]
    frame["source_csv"] = info["source"].name

    return frame.rename(
        columns={
            "mean_metric": "mean_auc_roc",
            "inference_mean": "inference_mean_seconds",
            "inference_median": "inference_median_seconds",
            "inference_std": "inference_std_seconds",
            "inference_min": "inference_min_seconds",
            "inference_max": "inference_max_seconds",
            "end_to_end_mean": "end_to_end_mean_seconds",
            "end_to_end_median": "end_to_end_median_seconds",
            "count": "measurement_count",
        }
    )[OUTPUT_COLUMNS]


def main():
    combined = pd.concat(
        [load_model_results(info) for info in MODEL_SOURCES], ignore_index=True
    )

    dataset_sets = [
        set(frame["did"])
        for _, frame in combined.groupby("model", sort=False)
    ]
    if not dataset_sets or any(dids != dataset_sets[0] for dids in dataset_sets[1:]):
        raise ValueError("Models do not cover the same set of datasets")
    if combined.duplicated(["did", "model"]).any():
        raise ValueError("Duplicate dataset/model rows found")

    combined = combined.sort_values(["did", "model_family", "model"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Saved {len(combined)} rows for {combined['model'].nunique()} models "
        f"and {combined['did'].nunique()} datasets to {OUTPUT_PATH}"
    )
    print(combined.groupby("model")["mean_auc_roc"].mean().to_string())


if __name__ == "__main__":
    main()
