"""Compare the completed elastic looped model with the original TabPFN.

This reuses the dataset selection, synchronized timing, split handling, metric,
and CSV schema from ``evaluation_per_dataset_speed_synchronized.py``.  The
model-level table is a macro-average over the per-dataset summary rows, so each
successfully evaluated dataset has equal weight.
"""

import argparse
import os
import random

import evaluation_per_dataset_speed_synchronized as benchmark
import numpy as np
import torch


MODELS = {
    "elastic_looped_3physical_core6x": {
        "path": (
            "tabpfn/models_diff/"
            "elastic_looped_transformer_3physical_core6x_18l.cpkt"
        ),
        "loader_type": "looped_transformer",
        "method_name": "transformer",
    },
    "tabpfn": {
        "path": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
        "loader_type": "tabpfn",
        "method_name": "transformer",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized per-dataset inference speed and AUC-ROC for "
            "the completed elastic three-physical-layer, six-loop model and "
            "the original TabPFN."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dids", nargs="+", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=30,
        help="Timed repetitions per benchmark split (splits 1-5).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_elastic_looped_vs_tabpfn_raw.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_elastic_looped_vs_tabpfn_summary.csv",
        ),
    )
    parser.add_argument(
        "--model-means-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_elastic_looped_vs_tabpfn_means.csv",
        ),
    )
    return parser.parse_args()


def summarize_across_datasets(summary_df):
    """Macro-average each model's per-dataset results."""
    if summary_df.empty:
        return summary_df

    return (
        summary_df.groupby("model", sort=False)
        .agg(
            dataset_count=("did", "nunique"),
            mean_auc_roc=("mean_metric", "mean"),
            mean_inference_seconds=("inference_mean", "mean"),
            mean_end_to_end_seconds=("end_to_end_mean", "mean"),
        )
        .reset_index()
    )


# Keep all benchmark behavior identical while limiting its model registry.
benchmark.MODELS = MODELS


def main():
    args = parse_args()
    args.device = benchmark.normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = benchmark.EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = benchmark.prepare_datasets(dids)
    models = {
        model_name: benchmark.load_model(model_name, args.device)
        for model_name in args.models
    }

    skipped = benchmark.run_warmups(models, datasets, args)
    raw_df = benchmark.run_timed_measurements(models, datasets, args, skipped)
    summary_df = benchmark.summarize(raw_df)
    model_means_df = summarize_across_datasets(summary_df)

    benchmark.write_csv(raw_df, args.raw_csv)
    benchmark.write_csv(summary_df, args.summary_csv)
    benchmark.write_csv(model_means_df, args.model_means_csv)

    print("\nSynchronized per-dataset inference summary:")
    if summary_df.empty:
        print("No successful measurements.")
    else:
        print(summary_df.to_string(index=False))

    print("\nMean across datasets (equal weight per dataset):")
    if model_means_df.empty:
        print("No successful measurements.")
    else:
        print(model_means_df.to_string(index=False))

    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved per-dataset summary to {args.summary_csv}")
    print(f"Saved model means to {args.model_means_csv}")


if __name__ == "__main__":
    main()
