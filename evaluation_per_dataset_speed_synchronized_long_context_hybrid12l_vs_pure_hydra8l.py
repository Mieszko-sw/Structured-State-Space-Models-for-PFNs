import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


MODELS = {
    "hybrid_12l": {
        "path": "tabpfn/models_diff/hydra_tabpfn_hybrid_12_layers_512e_lr0p0001_12l.cpkt",
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
    "pure_hydra_8l": {
        "path": "tabpfn/models_diff/pure_hydra_8_layers_8l.cpkt",
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
BPTT = 32768
EVAL_POSITION = BPTT // 2
# TabArena-v0.1 classification datasets with N * d > 200,000 and d <= 100,
# using the full-dataset sample (N) and feature (d) counts from Table B.2 of
# the TabArena paper.
DEFAULT_DIDS = [
    46950,  # polish_companies_bankruptcy: 5,910 * 65
    46962,  # taiwanese_bankruptcy_prediction: 6,819 * 95
    46969,  # NATICUSdroid: 7,491 * 87
    46916,  # coil2000_insurance_policies: 9,822 * 86
    46932,  # heloc: 10,459 * 24
    46979,  # jm1: 10,885 * 22
    46947,  # online_shoppers_intention: 12,330 * 18
    46937,  # in_vehicle_coupon_recommendation: 12,684 * 25
    46935,  # HR_Analytics_Job_Change_of_Data_Scientists: 19,158 * 13
    46919,  # credit_card_clients_default: 30,000 * 24
    46905,  # Amazon_employee_access: 32,769 * 10
    46910,  # bank-marketing: 45,211 * 14
    46922,  # Diabetes130US: 71,518 * 48
    46955,  # SDSS17: 78,053 * 12
    46920,  # customer_satisfaction_in_airline: 129,880 * 22
    46929,  # GiveMeSomeCredit: 150,000 * 11
]
SPLIT_NUMBERS = [1, 2, 3, 4, 5]
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
            "Compare synchronized wall-clock inference speed for Hybrid 12L and "
            "Pure Hydra 8L at 32k context length on TabArena-v0.1 classification datasets "
            "with N * d > 200,000 and at most 100 features."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--dids", nargs="+", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=15,
        help="Timed repetitions per benchmark split (splits 1-5 match evaluation_script.py).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join("result_csvs", "per_dataset_speed_synchronized_long_context_hybrid12l_vs_pure_hydra8l_raw_tabarena_16.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join("result_csvs", "per_dataset_speed_synchronized_long_context_hybrid12l_vs_pure_hydra8l_summary_tabarena_16.csv"),
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


def extract_model_inference_time(result, dataset_name, eval_position):
    return result.get(f"{dataset_name}_time_at_{eval_position}")


def real_eval_position(dataset_list, configured_eval_position):
    num_samples = len(dataset_list[0][1])
    dataset_bptt = min(num_samples, BPTT)
    if 2 * configured_eval_position > dataset_bptt:
        return int(dataset_bptt * 0.5)
    return configured_eval_position


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
    elif info["loader_type"] == "hydra_workflow":
        loaded, config, _ = hydra_load_model_workflow(
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
    # Use a half-context split instead of the checkpoint training position (usually 972).
    configured_eval_position = EVAL_POSITION
    eval_positions = [configured_eval_position]
    effective_eval_position = real_eval_position(dataset_list, configured_eval_position)

    synchronize(device)
    start = time.perf_counter()
    result = evaluate(
        datasets=dataset_list,
        bptt=BPTT,
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
        extract_model_inference_time(result, dataset_name, configured_eval_position),
        extract_metric(result),
        configured_eval_position,
        effective_eval_position,
    )


def make_failure_row(did, dataset_list, model_name, split_number, repetition, status, error):
    dataset = dataset_list[0]
    _, x, _, _, _, _ = dataset
    return {
        "did": did,
        "dataset_name": dataset[0],
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
    dataset = dataset_list[0]
    _, x, _, _, _, _ = dataset
    return {
        "did": did,
        "dataset_name": dataset[0],
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
        for split_number in SPLIT_NUMBERS:
            for repetition_idx in range(args.timed_runs):
                model_names = list(models)
                rng.shuffle(model_names)
                for model_name in model_names:
                    if (did, model_name) in failed:
                        continue
                    model, config = models[model_name]
                    try:
                        (
                            elapsed,
                            inference,
                            metric,
                            configured_position,
                            effective_position,
                        ) = run_single_measurement(
                            model,
                            model_name,
                            config,
                            dataset,
                            args.device,
                            split_number=split_number,
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
                                repetition_idx + 1,
                                "timed_oom",
                                format_error(error),
                            )
                        )
                        print(
                            f"skip timed did={did} model={model_name} split={split_number} "
                            f"repetition={repetition_idx + 1} error={format_error(error)}",
                            flush=True,
                        )
                        continue

                    rows.append(
                        make_success_row(
                            did,
                            dataset,
                            model_name,
                            split_number,
                            repetition_idx + 1,
                            elapsed,
                            inference,
                            metric,
                            configured_position,
                            effective_position,
                        )
                    )
                    print(
                        f"timed did={did} model={model_name} split={split_number} "
                        f"repetition={repetition_idx + 1} "
                        f"inference_seconds={inference:.6f} end_to_end_seconds={elapsed:.6f}",
                        flush=True,
                    )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw_df):
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
            inference_mean=("synchronized_model_inference_seconds", "mean"),
            inference_median=("synchronized_model_inference_seconds", "median"),
            inference_std=("synchronized_model_inference_seconds", "std"),
            inference_min=("synchronized_model_inference_seconds", "min"),
            inference_max=("synchronized_model_inference_seconds", "max"),
            end_to_end_mean=("synchronized_elapsed_seconds", "mean"),
            end_to_end_median=("synchronized_elapsed_seconds", "median"),
            count=("synchronized_model_inference_seconds", "count"),
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
    args.device = normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dids = args.dids if args.dids is not None else DEFAULT_DIDS
    datasets = prepare_datasets(dids)
    models = {model_name: load_model(model_name, args.device) for model_name in args.models}

    skipped = run_warmups(models, datasets, args)
    raw_df = run_timed_measurements(models, datasets, args, skipped)
    summary_df = summarize(raw_df)

    write_csv(raw_df, args.raw_csv)
    write_csv(summary_df, args.summary_csv)

    print("\nSynchronized per-dataset inference summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
