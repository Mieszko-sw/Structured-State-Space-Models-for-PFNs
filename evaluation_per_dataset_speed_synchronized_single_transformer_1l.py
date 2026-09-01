import argparse
import os
import random

import numpy as np
import torch

import evaluation_per_dataset_speed_synchronized as synchronized_evaluation
from evaluation_helper import EvalHelper


MODEL_NAME = "single_transformer_1l"
MODELS = {
    MODEL_NAME: {
        "path": "tabpfn/models_diff/single_transformer_1l.cpkt",
        "loader_type": "hybrid",
        "method_name": "transformer",
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized wall-clock inference speed for the trained "
            "single-layer, non-looped TabPFN transformer on the real OpenML "
            "datasets used by evaluation_script.py."
        )
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dids", nargs="+", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=30,
        help="Timed repetitions per benchmark split (splits 1-5 match evaluation_script.py).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_single_transformer_1l_raw.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_single_transformer_1l_summary.csv",
        ),
    )
    return parser.parse_args()


def load_single_transformer(device):
    # The shared evaluator looks up model metadata in its module-level registry.
    synchronized_evaluation.MODELS = MODELS
    model, config = synchronized_evaluation.load_model(MODEL_NAME, device)

    layer_types = config.get("hybrid_layer_types")
    if config.get("nlayers") != 1 or layer_types != ["transformer"]:
        raise ValueError(
            "Expected a one-layer non-looped transformer checkpoint, got "
            f"nlayers={config.get('nlayers')!r}, hybrid_layer_types={layer_types!r}"
        )
    return model, config


def main():
    args = parse_args()
    args.device = synchronized_evaluation.normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = synchronized_evaluation.prepare_datasets(dids)
    models = {MODEL_NAME: load_single_transformer(args.device)}

    skipped = synchronized_evaluation.run_warmups(models, datasets, args)
    raw_df = synchronized_evaluation.run_timed_measurements(
        models, datasets, args, skipped
    )
    summary_df = synchronized_evaluation.summarize(raw_df)

    synchronized_evaluation.write_csv(raw_df, args.raw_csv)
    synchronized_evaluation.write_csv(summary_df, args.summary_csv)

    print("\nSynchronized single-layer TabPFN per-dataset inference summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
