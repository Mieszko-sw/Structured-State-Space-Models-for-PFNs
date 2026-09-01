"""Compare a one-physical-layer 18x looped model with regular 12-layer TabPFN.

The per-dataset evaluation and CUDA timing protocol are inherited from
``evaluation_per_dataset_speed_synchronized.py``.  This entry point adds model
validation and an aggregate comparison relative to regular TabPFN.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


TABPFN_NAME = "tabpfn_12l"
LOOPED_NAME = "looped_18x"
MODEL_NAMES = (TABPFN_NAME, LOOPED_NAME)

DEFAULT_TABPFN_CHECKPOINT = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
DEFAULT_LOOPED_CHECKPOINT = (
    "tabpfn/models_diff/new_looped_transformer_1physical_core18x_18l.cpkt"
)
DEFAULT_RAW_CSV = os.path.join(
    "result_csvs", "tabpfn_12l_vs_looped_18x_synchronized_raw.csv"
)
DEFAULT_PER_DATASET_CSV = os.path.join(
    "result_csvs", "tabpfn_12l_vs_looped_18x_synchronized_per_dataset.csv"
)
DEFAULT_OVERALL_CSV = os.path.join(
    "result_csvs", "tabpfn_12l_vs_looped_18x_synchronized_overall.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare synchronized inference time and ROC-AUC for the trained "
            "one-physical-layer 18x looped model and regular 12-layer TabPFN."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tabpfn-checkpoint",
        default=DEFAULT_TABPFN_CHECKPOINT,
        help="Path to the regular 12-layer TabPFN checkpoint.",
    )
    parser.add_argument(
        "--looped-checkpoint",
        default=DEFAULT_LOOPED_CHECKPOINT,
        help="Path to the one-physical-layer 18x looped checkpoint.",
    )
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs (default: the filtered 30-dataset CC18 list).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        choices=benchmark.SPLIT_NUMBERS,
        default=list(benchmark.SPLIT_NUMBERS),
        help="Evaluation splits (default: 1 2 3 4 5).",
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=30,
        help="Timed repetitions per dataset and split.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--raw-csv", default=DEFAULT_RAW_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_PER_DATASET_CSV)
    parser.add_argument("--overall-csv", default=DEFAULT_OVERALL_CSV)
    return parser.parse_args()


def configure_models(args):
    checkpoints = {
        TABPFN_NAME: Path(args.tabpfn_checkpoint),
        LOOPED_NAME: Path(args.looped_checkpoint),
    }
    for model_name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {model_name} does not exist: {checkpoint}"
            )

    benchmark.MODELS = {
        TABPFN_NAME: {
            "path": str(checkpoints[TABPFN_NAME]),
            "loader_type": "tabpfn",
            "method_name": "transformer",
        },
        LOOPED_NAME: {
            "path": str(checkpoints[LOOPED_NAME]),
            "loader_type": "looped_transformer",
            "method_name": "transformer",
        },
    }
    benchmark.SPLIT_NUMBERS = list(args.splits)


def validate_models(models):
    tabpfn_model, tabpfn_config = models[TABPFN_NAME]
    looped_model, looped_config = models[LOOPED_NAME]

    if tabpfn_config.get("nlayers") != 12:
        raise ValueError(
            "Expected the regular TabPFN checkpoint to contain 12 layers, got "
            f"nlayers={tabpfn_config.get('nlayers')!r}"
        )

    repeat_pattern = looped_config.get("looped_core_repeat_pattern")
    physical_layers = getattr(
        getattr(looped_model, "transformer_encoder", None), "layers", None
    )
    physical_layer_count = len(physical_layers) if physical_layers is not None else None
    if (
        looped_config.get("nlayers") != 18
        or repeat_pattern != [18]
        or physical_layer_count != 1
    ):
        raise ValueError(
            "Expected one physical Transformer layer repeated 18 times, got "
            f"nlayers={looped_config.get('nlayers')!r}, "
            f"looped_core_repeat_pattern={repeat_pattern!r}, "
            f"physical_layers={physical_layer_count!r}"
        )

    print(
        f"Verified {TABPFN_NAME}: 12 physical layers, "
        f"{sum(parameter.numel() for parameter in tabpfn_model.parameters()):,} parameters",
        flush=True,
    )
    print(
        f"Verified {LOOPED_NAME}: 1 physical layer x 18 loops, "
        f"{sum(parameter.numel() for parameter in looped_model.parameters()):,} parameters",
        flush=True,
    )


def summarize_overall(per_dataset_df):
    models_per_dataset = per_dataset_df.groupby("did")["model"].agg(set)
    paired_dids = [
        did
        for did, available_models in models_per_dataset.items()
        if set(MODEL_NAMES).issubset(available_models)
    ]
    paired = per_dataset_df[per_dataset_df["did"].isin(paired_dids)]
    if paired.empty:
        raise ValueError("No dataset has successful results for both models.")

    overall = (
        paired.groupby("model")
        .agg(
            datasets=("did", "nunique"),
            average_aucroc=("mean_metric", "mean"),
            average_inference_seconds=("inference_mean", "mean"),
            average_end_to_end_seconds=("end_to_end_mean", "mean"),
        )
        .reindex(MODEL_NAMES)
        .reset_index()
    )

    reference = overall.loc[overall["model"] == TABPFN_NAME]
    if reference.empty:
        raise ValueError("The regular 12-layer TabPFN reference has no successful results.")

    reference_aucroc = float(reference.iloc[0]["average_aucroc"])
    reference_inference = float(reference.iloc[0]["average_inference_seconds"])
    overall["aucroc_delta_vs_tabpfn"] = (
        overall["average_aucroc"] - reference_aucroc
    )
    overall["inference_delta_seconds_vs_tabpfn"] = (
        overall["average_inference_seconds"] - reference_inference
    )
    overall["inference_ratio_vs_tabpfn"] = (
        overall["average_inference_seconds"] / reference_inference
    )
    overall["inference_change_vs_tabpfn_percent"] = 100.0 * (
        overall["inference_ratio_vs_tabpfn"] - 1.0
    )
    overall["speedup_vs_tabpfn"] = (
        reference_inference / overall["average_inference_seconds"]
    )
    return overall


def main():
    args = parse_args()
    if args.warmup_runs < 0 or args.timed_runs < 1:
        raise ValueError("--warmup-runs must be >= 0 and --timed-runs must be >= 1")

    args.device = benchmark.normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    configure_models(args)
    eval_helper = EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = benchmark.prepare_datasets(dids)
    models = {
        model_name: benchmark.load_model(model_name, args.device)
        for model_name in MODEL_NAMES
    }
    validate_models(models)

    skipped = benchmark.run_warmups(models, datasets, args)
    raw_df = benchmark.run_timed_measurements(models, datasets, args, skipped)
    per_dataset_df = benchmark.summarize(raw_df)
    overall_df = summarize_overall(per_dataset_df)

    benchmark.write_csv(raw_df, args.raw_csv)
    benchmark.write_csv(per_dataset_df, args.summary_csv)
    benchmark.write_csv(overall_df, args.overall_csv)

    print("\nSynchronized per-dataset inference and performance summary:")
    print(per_dataset_df.to_string(index=False))
    print("\nAverage performance and inference time versus regular 12-layer TabPFN:")
    print(overall_df.to_string(index=False))
    print(f"\nSaved raw measurements to {args.raw_csv}")
    print(f"Saved per-dataset summary to {args.summary_csv}")
    print(f"Saved overall comparison to {args.overall_csv}")


if __name__ == "__main__":
    main()
