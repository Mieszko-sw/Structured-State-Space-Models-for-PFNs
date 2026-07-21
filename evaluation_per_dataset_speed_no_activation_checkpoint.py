import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import tabpfn.scripts.hydra_prediction_interface as hydra_prediction_interface
import tabpfn.scripts.transformer_prediction_interface as transformer_prediction_interface

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


MODELS = {
    "hybrid_8l": {
        "path": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
    "tabpfn": {
        "path": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
        "loader_type": "tabpfn",
        "method_name": "transformer",
    },
    "hydra": {
        "path": "tabpfn/models_diff/callback_pure_hydra_12_layers_512e_latest.cpkt",
        "loader_type": "hydra",
        "method_name": "hydra",
    },
}

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
    "eval_position",
    "run",
    "status",
    "error",
    "synchronized_elapsed_seconds",
    "internal_unsynchronized_seconds",
    "mean_metric",
]


def direct_forward_without_activation_checkpoint(function, *args, **kwargs):
    """Run the inference function directly, ignoring checkpoint-only options."""
    kwargs.pop("use_reentrant", None)
    return function(*args, **kwargs)


def disable_inference_activation_checkpointing():
    """Bypass activation checkpoint wrappers; trained weights stay unchanged."""
    hydra_prediction_interface.checkpoint = direct_forward_without_activation_checkpoint
    transformer_prediction_interface.checkpoint = direct_forward_without_activation_checkpoint



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized wall-clock inference speed without activation checkpointing for Hybrid 8L, TabPFN, "
            "and Hydra on the real OpenML datasets used by evaluation_script.py."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dids", nargs="+", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join("result_csvs", "per_dataset_speed_no_activation_checkpoint_raw.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join("result_csvs", "per_dataset_speed_no_activation_checkpoint_summary.csv"),
    )
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
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


def extract_metric(result):
    metric = result["mean_metric"]
    if hasattr(metric, "item"):
        return metric.item()
    return float(metric)


def extract_internal_inference_time(result, dataset_name, eval_position):
    return result.get(f"{dataset_name}_time_at_{eval_position}")


def load_model(model_name, device):
    info = MODELS[model_name]
    if info["loader_type"] == "tabpfn":
        loaded, config, _ = transformer_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=info["path"],
        )
    else:
        loaded, config = load_model_only_inference(
            ".",
            info["path"],
            device,
            model_name=info["loader_type"],
        )

    model = loaded[2]
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {params:,} parameters from {info['path']}", flush=True)
    return model, config


def prepare_datasets(dids):
    eval_helper = EvalHelper()
    eval_helper.check_datasets_data(dids)
    eval_helper.make_limit_datasets(
        max_classes=10,
        max_features=100,
        limit_dids=dids,
        eval_filters=EVALUATION_TYPE_FILTERS,
    )
    return eval_helper.limit_dict


def run_single_measurement(model, model_name, config, dataset_list, device, split_number):
    dataset = dataset_list[0]
    dataset_name = dataset[0]
    eval_positions = config["eval_positions"]
    eval_position = eval_positions[0] if len(eval_positions) == 1 else max(eval_positions)

    synchronize(device)
    start = time.perf_counter()
    result = evaluate(
        datasets=dataset_list,
        bptt=config.get("bptt", 1024),
        eval_positions=eval_positions,
        metric_used=METRIC_USED,
        model=model,
        device=device,
        method_name=MODELS[model_name]["method_name"],
        max_time=300,
        split_number=split_number,
        jrt_prompt=False,
        random_premutation=False,
        single_evaluation_prompt=False,
        permutation_bagging=1,
        sample_bagging=0,
    )
    synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    return (
        elapsed_seconds,
        extract_internal_inference_time(result, dataset_name, eval_position),
        extract_metric(result),
        eval_position,
    )


def make_failure_row(did, dataset_list, model_name, run, status, error):
    dataset = dataset_list[0]
    _, x, _, _, _, _ = dataset
    return {
        "did": did,
        "dataset_name": dataset[0],
        "model": model_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "eval_position": np.nan,
        "run": run,
        "status": status,
        "error": error,
        "synchronized_elapsed_seconds": np.nan,
        "internal_unsynchronized_seconds": np.nan,
        "mean_metric": np.nan,
    }


def make_success_row(did, dataset_list, model_name, run, elapsed, internal, metric, eval_position):
    dataset = dataset_list[0]
    _, x, _, _, _, _ = dataset
    return {
        "did": did,
        "dataset_name": dataset[0],
        "model": model_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "eval_position": eval_position,
        "run": run,
        "status": "ok",
        "error": "",
        "synchronized_elapsed_seconds": elapsed,
        "internal_unsynchronized_seconds": internal,
        "mean_metric": metric,
    }


def run_warmups(models, datasets, args):
    skipped = set()
    for did, dataset in datasets.items():
        for model_name, (model, config) in models.items():
            for warmup_idx in range(args.warmup_runs):
                try:
                    run_single_measurement(model, model_name, config, dataset, args.device, 1)
                except RuntimeError as error:
                    if not is_oom_error(error):
                        raise
                    clear_device_cache(args.device)
                    skipped.add((did, model_name))
                    print(
                        f"skip warmup did={did} model={model_name} "
                        f"run={warmup_idx + 1} error={format_error(error)}",
                        flush=True,
                    )
                    break
                print(
                    f"warmup did={did} model={model_name} run={warmup_idx + 1}",
                    flush=True,
                )
    return skipped


def run_timed_measurements(models, datasets, args, skipped):
    rng = random.Random(args.seed)
    rows = [
        make_failure_row(did, datasets[did], model_name, 0, "skipped_warmup_oom", "warmup CUDA out of memory")
        for did, model_name in sorted(skipped)
    ]
    failed = set(skipped)

    for did, dataset in datasets.items():
        for run_idx in range(args.timed_runs):
            model_names = list(models)
            rng.shuffle(model_names)
            for model_name in model_names:
                if (did, model_name) in failed:
                    continue
                model, config = models[model_name]
                try:
                    elapsed, internal, metric, eval_position = run_single_measurement(
                        model,
                        model_name,
                        config,
                        dataset,
                        args.device,
                        split_number=run_idx + 1,
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
                            run_idx + 1,
                            "timed_oom",
                            format_error(error),
                        )
                    )
                    print(
                        f"skip timed did={did} model={model_name} "
                        f"run={run_idx + 1} error={format_error(error)}",
                        flush=True,
                    )
                    continue

                rows.append(
                    make_success_row(
                        did,
                        dataset,
                        model_name,
                        run_idx + 1,
                        elapsed,
                        internal,
                        metric,
                        eval_position,
                    )
                )
                print(
                    f"timed did={did} model={model_name} run={run_idx + 1} "
                    f"seconds={elapsed:.6f}",
                    flush=True,
                )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw_df):
    ok = raw_df[raw_df["status"] == "ok"]
    return (
        ok.groupby(["did", "dataset_name", "model", "num_samples", "num_features", "eval_position"])
        .agg(
            mean=("synchronized_elapsed_seconds", "mean"),
            median=("synchronized_elapsed_seconds", "median"),
            std=("synchronized_elapsed_seconds", "std"),
            min=("synchronized_elapsed_seconds", "min"),
            max=("synchronized_elapsed_seconds", "max"),
            count=("synchronized_elapsed_seconds", "count"),
            mean_metric=("mean_metric", "mean"),
        )
        .reset_index()
    )


def write_csv(df, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    disable_inference_activation_checkpointing()
    print("Inference activation checkpointing disabled; trained weights are unchanged.", flush=True)
    args.device = normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = prepare_datasets(dids)
    models = {model_name: load_model(model_name, args.device) for model_name in args.models}

    skipped = run_warmups(models, datasets, args)
    raw_df = run_timed_measurements(models, datasets, args, skipped)
    summary_df = summarize(raw_df)

    write_csv(raw_df, args.raw_csv)
    write_csv(summary_df, args.summary_csv)

    print("\nSynchronized per-dataset summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
