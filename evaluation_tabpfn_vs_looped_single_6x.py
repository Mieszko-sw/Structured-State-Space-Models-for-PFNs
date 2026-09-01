"""Benchmark TabPFN against one Transformer block looped 6 and 12 times.

The timing methodology matches evaluation_per_dataset_speed_synchronized.py:
CUDA is synchronized around each evaluation call, models are warmed up, and
model order is randomized within each timed repetition.
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


MODEL_NAMES = ("tabpfn", "looped_single_6x", "looped_single_12x")
DEFAULT_TABPFN_CHECKPOINT = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
DEFAULT_LOOPED_6X_CHECKPOINT = (
    "tabpfn/models_diff/callback_new_looped_transformer_1physical_core6x_latest.cpkt"
)
DEFAULT_LOOPED_12X_CHECKPOINT = (
    "tabpfn/models_diff/callback_new_looped_transformer_1physical_core12x_latest.cpkt"
)
DEFAULT_RAW_CSV = os.path.join(
    "result_csvs", "tabpfn_vs_looped_single_6x_12x_raw.csv"
)
DEFAULT_SUMMARY_CSV = os.path.join(
    "result_csvs", "tabpfn_vs_looped_single_6x_12x_per_dataset.csv"
)
DEFAULT_OVERALL_CSV = os.path.join(
    "result_csvs", "tabpfn_vs_looped_single_6x_12x_overall.csv"
)

EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}
METRIC_USED = tabular_metrics.auc_metric
RAW_COLUMNS = [
    "did",
    "dataset_name",
    "model",
    "num_samples",
    "num_features",
    "configured_eval_position",
    "real_eval_position",
    "split_number",
    "timing_repetition",
    "status",
    "error",
    "synchronized_elapsed_seconds",
    "synchronized_model_inference_seconds",
    "mean_metric",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized per-dataset inference time and AUC for standard "
            "TabPFN and one physical Transformer block looped 6 or 12 times."
        )
    )
    parser.add_argument(
        "--tabpfn-checkpoint",
        default=DEFAULT_TABPFN_CHECKPOINT,
        help="Path to the standard TabPFN checkpoint.",
    )
    parser.add_argument(
        "--looped-checkpoint",
        "--looped-6x-checkpoint",
        dest="looped_6x_checkpoint",
        default=DEFAULT_LOOPED_6X_CHECKPOINT,
        help="Path to the one-physical-layer, six-loop checkpoint.",
    )
    parser.add_argument(
        "--looped-12x-checkpoint",
        default=DEFAULT_LOOPED_12X_CHECKPOINT,
        help="Path to the one-physical-layer, twelve-loop checkpoint.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--evaluation-type",
        choices=("openmlcc18", "openmlcc18_large", "test", "valid"),
        default="openmlcc18",
        help="Dataset group used when --dids is not supplied.",
    )
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="Optional OpenML dataset IDs; overrides --evaluation-type.",
    )
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=15,
        help="Timed repetitions for every dataset and split.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--permutation-bagging", type=int, default=1)
    parser.add_argument("--sample-bagging", type=int, default=0)
    parser.add_argument("--jrt-prompt", action="store_true")
    parser.add_argument("--single-evaluation-prompt", action="store_true")
    parser.add_argument("--raw-csv", default=DEFAULT_RAW_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--overall-csv", default=DEFAULT_OVERALL_CSV)
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA is unavailable; using CPU.", flush=True)
            return "cpu"
        torch.cuda.set_device(torch.device(device))
    return device


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def clear_device_cache(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom_error(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def format_error(error):
    return str(error).splitlines()[0]


def check_checkpoint(path, label):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint does not exist: {checkpoint}")
    return checkpoint


def load_tabpfn(checkpoint, device):
    loaded, config, _ = transformer_load_model_workflow(
        2,
        -1,
        add_name="",
        base_path="",
        device=device,
        eval_addition="",
        only_inference=True,
        model_path_custom=str(checkpoint),
    )
    return loaded[2], config


def load_looped_single(checkpoint, device):
    loaded, config = load_model_only_inference(
        ".",
        str(checkpoint),
        device,
        model_name="looped_transformer",
    )
    return loaded[2], config


def load_models(args):
    checkpoints = {
        "tabpfn": check_checkpoint(args.tabpfn_checkpoint, "TabPFN"),
        "looped_single_6x": check_checkpoint(args.looped_6x_checkpoint, "Looped 6x"),
        "looped_single_12x": check_checkpoint(args.looped_12x_checkpoint, "Looped 12x"),
    }
    loaders = {
        "tabpfn": load_tabpfn,
        "looped_single_6x": load_looped_single,
        "looped_single_12x": load_looped_single,
    }

    models = {}
    for model_name in MODEL_NAMES:
        model, config = loaders[model_name](checkpoints[model_name], args.device)
        model.eval()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{model_name}: {parameters:,} parameters from {checkpoints[model_name]}",
            flush=True,
        )
        print(
            f"  nlayers={config.get('nlayers')} emsize={config.get('emsize')} "
            f"eval_positions={config.get('eval_positions')} "
            f"loop_pattern={config.get('looped_core_repeat_pattern')}",
            flush=True,
        )
        models[model_name] = (model, config)
    return models


def resolve_dids(eval_helper, args):
    if args.dids is not None:
        return args.dids
    if args.evaluation_type == "openmlcc18":
        return eval_helper.openml_cc18_dids_small
    if args.evaluation_type == "openmlcc18_large":
        return eval_helper.openml_cc18_dids_large
    if args.evaluation_type == "test":
        return eval_helper.test_dids_classification
    return eval_helper.valid_dids_classification


def prepare_datasets(eval_helper, dids, args):
    eval_helper.check_datasets_data(dids)
    eval_helper.make_limit_datasets(
        max_classes=args.max_classes,
        max_features=args.max_features,
        limit_dids=dids,
        eval_filters=EVALUATION_TYPE_FILTERS,
    )
    return {did: eval_helper.limit_dict[did] for did in dids if did in eval_helper.limit_dict}


def extract_metric(result):
    metric = result["mean_metric"]
    return metric.item() if hasattr(metric, "item") else float(metric)


def extract_model_inference_time(result, dataset_name, eval_position):
    inference = result.get(f"{dataset_name}_time_at_{eval_position}")
    if inference is None:
        raise KeyError(
            f"Missing inference timing for dataset={dataset_name}, "
            f"eval_position={eval_position}"
        )
    return float(inference)


def real_eval_position(dataset_list, configured_eval_position, bptt):
    num_samples = len(dataset_list[0][1])
    dataset_bptt = min(num_samples, bptt)
    if 2 * configured_eval_position > dataset_bptt:
        return int(dataset_bptt * 0.5)
    return configured_eval_position


def run_single_measurement(model, config, dataset_list, args, split_number):
    dataset_name = dataset_list[0][0]
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(f"Expected one evaluation position, got {eval_positions}")
    configured_eval_position = eval_positions[0]
    effective_eval_position = real_eval_position(
        dataset_list, configured_eval_position, args.bptt
    )

    synchronize(args.device)
    start = time.perf_counter()
    result = evaluate(
        datasets=dataset_list,
        bptt=args.bptt,
        eval_positions=eval_positions,
        metric_used=METRIC_USED,
        model=model,
        device=args.device,
        method_name="transformer",
        max_time=args.max_time,
        split_number=split_number,
        jrt_prompt=args.jrt_prompt,
        random_premutation=False,
        single_evaluation_prompt=args.single_evaluation_prompt,
        permutation_bagging=args.permutation_bagging,
        sample_bagging=args.sample_bagging,
    )
    synchronize(args.device)
    elapsed_seconds = time.perf_counter() - start

    return (
        elapsed_seconds,
        extract_model_inference_time(result, dataset_name, configured_eval_position),
        extract_metric(result),
        configured_eval_position,
        effective_eval_position,
    )


def make_failure_row(did, dataset_list, model_name, split_number, repetition, status, error):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    return {
        "did": did,
        "dataset_name": dataset_name,
        "model": model_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "configured_eval_position": np.nan,
        "real_eval_position": np.nan,
        "split_number": split_number,
        "timing_repetition": repetition,
        "status": status,
        "error": error,
        "synchronized_elapsed_seconds": np.nan,
        "synchronized_model_inference_seconds": np.nan,
        "mean_metric": np.nan,
    }


def make_success_row(
    did,
    dataset_list,
    model_name,
    split_number,
    repetition,
    elapsed,
    inference,
    metric,
    configured_eval_position,
    effective_eval_position,
):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    return {
        "did": did,
        "dataset_name": dataset_name,
        "model": model_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "configured_eval_position": configured_eval_position,
        "real_eval_position": effective_eval_position,
        "split_number": split_number,
        "timing_repetition": repetition,
        "status": "ok",
        "error": "",
        "synchronized_elapsed_seconds": elapsed,
        "synchronized_model_inference_seconds": inference,
        "mean_metric": metric,
    }


def run_warmups(models, datasets, args):
    skipped = set()
    for did, dataset in datasets.items():
        for model_name, (model, config) in models.items():
            for warmup_index in range(args.warmup_runs):
                try:
                    run_single_measurement(model, config, dataset, args, split_number=1)
                except RuntimeError as error:
                    if not is_oom_error(error):
                        raise
                    clear_device_cache(args.device)
                    skipped.add((did, model_name))
                    print(
                        f"skip warmup did={did} model={model_name} "
                        f"run={warmup_index + 1} error={format_error(error)}",
                        flush=True,
                    )
                    break
                print(
                    f"warmup did={did} model={model_name} run={warmup_index + 1}",
                    flush=True,
                )
    return skipped


def run_timed_measurements(models, datasets, args, skipped):
    rng = random.Random(args.seed)
    rows = [
        make_failure_row(
            did,
            datasets[did],
            model_name,
            1,
            0,
            "skipped_warmup_oom",
            "warmup CUDA out of memory",
        )
        for did, model_name in sorted(skipped)
    ]
    failed = set(skipped)

    for did, dataset in datasets.items():
        for split_number in args.splits:
            for repetition_index in range(args.timed_runs):
                model_names = list(MODEL_NAMES)
                rng.shuffle(model_names)
                for model_name in model_names:
                    if (did, model_name) in failed:
                        continue
                    model, config = models[model_name]
                    try:
                        measurement = run_single_measurement(
                            model, config, dataset, args, split_number
                        )
                    except RuntimeError as error:
                        if not is_oom_error(error):
                            raise
                        clear_device_cache(args.device)
                        failed.add((did, model_name))
                        rows.append(
                            make_failure_row(
                                did,
                                dataset,
                                model_name,
                                split_number,
                                repetition_index + 1,
                                "timed_oom",
                                format_error(error),
                            )
                        )
                        print(
                            f"skip timed did={did} model={model_name} "
                            f"split={split_number} repetition={repetition_index + 1} "
                            f"error={format_error(error)}",
                            flush=True,
                        )
                        continue

                    elapsed, inference, metric, configured_position, real_position = measurement
                    rows.append(
                        make_success_row(
                            did,
                            dataset,
                            model_name,
                            split_number,
                            repetition_index + 1,
                            elapsed,
                            inference,
                            metric,
                            configured_position,
                            real_position,
                        )
                    )
                    print(
                        f"timed did={did} model={model_name} split={split_number} "
                        f"repetition={repetition_index + 1} "
                        f"auc={metric:.6f} inference_seconds={inference:.6f} "
                        f"end_to_end_seconds={elapsed:.6f}",
                        flush=True,
                    )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize_per_dataset(raw_df):
    ok = raw_df[raw_df["status"] == "ok"]
    return (
        ok.groupby(
            [
                "did",
                "dataset_name",
                "model",
                "num_samples",
                "num_features",
                "configured_eval_position",
                "real_eval_position",
            ]
        )
        .agg(
            inference_mean_seconds=("synchronized_model_inference_seconds", "mean"),
            inference_median_seconds=("synchronized_model_inference_seconds", "median"),
            inference_std_seconds=("synchronized_model_inference_seconds", "std"),
            inference_min_seconds=("synchronized_model_inference_seconds", "min"),
            inference_max_seconds=("synchronized_model_inference_seconds", "max"),
            end_to_end_mean_seconds=("synchronized_elapsed_seconds", "mean"),
            end_to_end_median_seconds=("synchronized_elapsed_seconds", "median"),
            mean_metric=("mean_metric", "mean"),
            measurement_count=("synchronized_model_inference_seconds", "count"),
        )
        .reset_index()
    )


def summarize_overall(per_dataset_df):
    return (
        per_dataset_df.groupby("model")
        .agg(
            average_inference_seconds=("inference_mean_seconds", "mean"),
            average_end_to_end_seconds=("end_to_end_mean_seconds", "mean"),
            average_performance_auc=("mean_metric", "mean"),
            datasets_evaluated=("did", "nunique"),
        )
        .reindex(MODEL_NAMES)
        .reset_index()
    )


def write_csv(dataframe, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    dataframe.to_csv(path, index=False)


def main():
    args = parse_args()
    if args.warmup_runs < 0 or args.timed_runs < 1:
        raise ValueError("--warmup-runs must be >= 0 and --timed-runs must be >= 1")
    args.device = normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = EvalHelper()
    dids = resolve_dids(eval_helper, args)
    datasets = prepare_datasets(eval_helper, dids, args)
    models = load_models(args)

    skipped = run_warmups(models, datasets, args)
    raw_df = run_timed_measurements(models, datasets, args, skipped)
    per_dataset_df = summarize_per_dataset(raw_df)
    overall_df = summarize_overall(per_dataset_df)

    write_csv(raw_df, args.raw_csv)
    write_csv(per_dataset_df, args.summary_csv)
    write_csv(overall_df, args.overall_csv)

    print("\nSynchronized per-dataset inference and performance summary:")
    print(per_dataset_df.to_string(index=False))
    print("\nAverage inference and average performance across datasets:")
    print(overall_df.to_string(index=False))
    print(f"\nSaved raw measurements to {args.raw_csv}")
    print(f"Saved per-dataset summary to {args.summary_csv}")
    print(f"Saved overall summary to {args.overall_csv}")


if __name__ == "__main__":
    main()
