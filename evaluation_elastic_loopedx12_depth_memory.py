"""Benchmark elastic_loopedx12 at fixed depths on OpenML datasets.

The benchmark reports AUC-ROC, synchronized model-only and end-to-end latency,
and CUDA allocated/reserved memory for loop depths 3, 6, 9, and 12 by default.
All depths share one loaded checkpoint; only the recurrent compute budget changes.
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark


MAX_LOOPS = 12
DEFAULT_DEPTHS = [3, 6, 9, 12]
MODEL_PATH = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_1physical_core12x_latest.cpkt"
)
MIB = 1024 ** 2


def model_name(num_loops):
    return f"elastic_loopedx12_depth_{num_loops:02d}"


MODELS = {
    model_name(num_loops): {
        "path": MODEL_PATH,
        "loader_type": "looped_transformer",
        "method_name": "transformer",
        "num_loops": num_loops,
    }
    for num_loops in range(1, MAX_LOOPS + 1)
}

# benchmark.load_model consults the registry in its own module.
benchmark.MODELS = MODELS


RAW_COLUMNS = [
    "did",
    "dataset_name",
    "model",
    "num_loops",
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
    "aucroc",
    "model_parameter_mib",
    "gpu_allocated_before_mib",
    "gpu_peak_allocated_mib",
    "gpu_incremental_peak_allocated_mib",
    "gpu_reserved_before_mib",
    "gpu_peak_reserved_mib",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark elastic_loopedx12 accuracy, synchronized inference speed, "
            "and CUDA memory at fixed recurrent depths on OpenML datasets."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        choices=range(1, MAX_LOOPS + 1),
        default=DEFAULT_DEPTHS,
        metavar="N",
        help="Loop depths to benchmark (default: 3 6 9 12).",
    )
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs (default: the 30-dataset filtered CC18 list).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        choices=benchmark.SPLIT_NUMBERS,
        default=benchmark.SPLIT_NUMBERS,
        help="Evaluation splits (default: 1 2 3 4 5).",
    )
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=15,
        help="Timed repetitions per dataset, depth, and split.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs", "elastic_loopedx12_depth_memory_raw.csv"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs", "elastic_loopedx12_depth_memory_summary.csv"
        ),
    )
    return parser.parse_args()


def is_cuda(device):
    return str(device).startswith("cuda") and torch.cuda.is_available()


def to_mib(num_bytes):
    return float(num_bytes) / MIB


def begin_memory_measurement(device):
    if not is_cuda(device):
        return {
            "allocated_before": np.nan,
            "reserved_before": np.nan,
        }

    cuda_device = torch.device(device)
    torch.cuda.synchronize(cuda_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    return {
        "allocated_before": torch.cuda.memory_allocated(cuda_device),
        "reserved_before": torch.cuda.memory_reserved(cuda_device),
    }


def end_memory_measurement(device, before):
    if not is_cuda(device):
        return {
            "gpu_allocated_before_mib": np.nan,
            "gpu_peak_allocated_mib": np.nan,
            "gpu_incremental_peak_allocated_mib": np.nan,
            "gpu_reserved_before_mib": np.nan,
            "gpu_peak_reserved_mib": np.nan,
        }

    cuda_device = torch.device(device)
    torch.cuda.synchronize(cuda_device)
    peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
    peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
    return {
        "gpu_allocated_before_mib": to_mib(before["allocated_before"]),
        "gpu_peak_allocated_mib": to_mib(peak_allocated),
        "gpu_incremental_peak_allocated_mib": to_mib(
            max(0, peak_allocated - before["allocated_before"])
        ),
        "gpu_reserved_before_mib": to_mib(before["reserved_before"]),
        "gpu_peak_reserved_mib": to_mib(peak_reserved),
    }


def run_single_measurement(
    model,
    model_label,
    config,
    dataset_list,
    device,
    split_number,
    collect_memory,
):
    dataset_name = dataset_list[0][0]
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(f"Expected one evaluation position, got {eval_positions}")

    configured_eval_position = eval_positions[0]
    effective_eval_position = benchmark.real_eval_position(
        dataset_list, configured_eval_position
    )
    before = begin_memory_measurement(device) if collect_memory else None

    benchmark.synchronize(device)
    start = time.perf_counter()
    result = benchmark.evaluate(
        datasets=dataset_list,
        bptt=benchmark.BPTT,
        eval_positions=eval_positions,
        metric_used=benchmark.METRIC_USED,
        model=model,
        device=device,
        method_name=MODELS[model_label]["method_name"],
        max_time=300,
        split_number=split_number,
        jrt_prompt=False,
        random_premutation=False,
        single_evaluation_prompt=False,
        permutation_bagging=1,
        sample_bagging=0,
        num_loops=MODELS[model_label]["num_loops"],
    )
    benchmark.synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    memory = (
        end_memory_measurement(device, before)
        if collect_memory
        else {}
    )
    return {
        "elapsed": elapsed_seconds,
        "inference": benchmark.extract_model_inference_time(
            result, dataset_name, configured_eval_position
        ),
        "aucroc": benchmark.extract_metric(result),
        "configured_eval_position": configured_eval_position,
        "real_eval_position": effective_eval_position,
        **memory,
    }


def dataset_metadata(dataset_list):
    dataset = dataset_list[0]
    _, x, _, _, _, _ = dataset
    return dataset[0], int(x.shape[0]), int(x.shape[1])


def make_failure_row(
    did,
    dataset_list,
    model_label,
    split_number,
    repetition,
    status,
    error,
    model_parameter_mib,
):
    dataset_name, num_samples, num_features = dataset_metadata(dataset_list)
    return {
        "did": did,
        "dataset_name": dataset_name,
        "model": model_label,
        "num_loops": MODELS[model_label]["num_loops"],
        "num_samples": num_samples,
        "num_features": num_features,
        "configured_eval_position": np.nan,
        "real_eval_position": np.nan,
        "split_number": split_number,
        "timing_repetition": repetition,
        "status": status,
        "error": error,
        "model_parameter_mib": model_parameter_mib,
    }


def make_success_row(
    did,
    dataset_list,
    model_label,
    split_number,
    repetition,
    measurement,
    model_parameter_mib,
):
    dataset_name, num_samples, num_features = dataset_metadata(dataset_list)
    return {
        "did": did,
        "dataset_name": dataset_name,
        "model": model_label,
        "num_loops": MODELS[model_label]["num_loops"],
        "num_samples": num_samples,
        "num_features": num_features,
        "configured_eval_position": measurement["configured_eval_position"],
        "real_eval_position": measurement["real_eval_position"],
        "split_number": split_number,
        "timing_repetition": repetition,
        "status": "ok",
        "error": "",
        "synchronized_elapsed_seconds": measurement["elapsed"],
        "synchronized_model_inference_seconds": measurement["inference"],
        "aucroc": measurement["aucroc"],
        "model_parameter_mib": model_parameter_mib,
        "gpu_allocated_before_mib": measurement["gpu_allocated_before_mib"],
        "gpu_peak_allocated_mib": measurement["gpu_peak_allocated_mib"],
        "gpu_incremental_peak_allocated_mib": measurement[
            "gpu_incremental_peak_allocated_mib"
        ],
        "gpu_reserved_before_mib": measurement["gpu_reserved_before_mib"],
        "gpu_peak_reserved_mib": measurement["gpu_peak_reserved_mib"],
    }


def run_warmups(models, datasets, args):
    skipped = set()
    for did, dataset in datasets.items():
        for model_label, (model, config) in models.items():
            for warmup_idx in range(args.warmup_runs):
                try:
                    run_single_measurement(
                        model,
                        model_label,
                        config,
                        dataset,
                        args.device,
                        split_number=args.splits[0],
                        collect_memory=False,
                    )
                except RuntimeError as error:
                    if not benchmark.is_oom_error(error):
                        raise
                    benchmark.clear_device_cache(args.device)
                    skipped.add((did, model_label))
                    print(
                        f"skip warmup did={did} model={model_label} "
                        f"error={benchmark.format_error(error)}",
                        flush=True,
                    )
                    break
                print(
                    f"warmup did={did} model={model_label} run={warmup_idx + 1}",
                    flush=True,
                )
    return skipped


def run_timed_measurements(models, datasets, args, skipped, parameter_mib):
    rng = random.Random(args.seed)
    rows = []
    failed = set(skipped)

    for did, model_label in sorted(skipped):
        rows.append(
            make_failure_row(
                did,
                datasets[did],
                model_label,
                args.splits[0],
                0,
                "skipped_warmup_oom",
                "warmup CUDA out of memory",
                parameter_mib,
            )
        )

    for did, dataset in datasets.items():
        for split_number in args.splits:
            for repetition_idx in range(args.timed_runs):
                model_labels = list(models)
                rng.shuffle(model_labels)
                for model_label in model_labels:
                    if (did, model_label) in failed:
                        continue
                    model, config = models[model_label]
                    try:
                        measurement = run_single_measurement(
                            model,
                            model_label,
                            config,
                            dataset,
                            args.device,
                            split_number,
                            collect_memory=True,
                        )
                    except RuntimeError as error:
                        if not benchmark.is_oom_error(error):
                            raise
                        benchmark.clear_device_cache(args.device)
                        failed.add((did, model_label))
                        rows.append(
                            make_failure_row(
                                did,
                                dataset,
                                model_label,
                                split_number,
                                repetition_idx + 1,
                                "timed_oom",
                                benchmark.format_error(error),
                                parameter_mib,
                            )
                        )
                        print(
                            f"skip timed did={did} model={model_label} "
                            f"split={split_number} error={benchmark.format_error(error)}",
                            flush=True,
                        )
                        continue

                    rows.append(
                        make_success_row(
                            did,
                            dataset,
                            model_label,
                            split_number,
                            repetition_idx + 1,
                            measurement,
                            parameter_mib,
                        )
                    )
                    print(
                        f"timed did={did} depth={MODELS[model_label]['num_loops']} "
                        f"split={split_number} repetition={repetition_idx + 1} "
                        f"aucroc={measurement['aucroc']:.6f} "
                        f"inference_seconds={measurement['inference']:.6f} "
                        f"peak_allocated_mib="
                        f"{measurement['gpu_peak_allocated_mib']:.2f}",
                        flush=True,
                    )

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw_df, reference_depth):
    ok = raw_df[raw_df["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(
            [
                "did",
                "dataset_name",
                "model",
                "num_loops",
                "num_samples",
                "num_features",
                "configured_eval_position",
                "real_eval_position",
            ]
        )
        .agg(
            aucroc_mean=("aucroc", "mean"),
            aucroc_std=("aucroc", "std"),
            inference_mean_seconds=(
                "synchronized_model_inference_seconds",
                "mean",
            ),
            inference_median_seconds=(
                "synchronized_model_inference_seconds",
                "median",
            ),
            inference_std_seconds=(
                "synchronized_model_inference_seconds",
                "std",
            ),
            end_to_end_mean_seconds=("synchronized_elapsed_seconds", "mean"),
            end_to_end_median_seconds=("synchronized_elapsed_seconds", "median"),
            gpu_peak_allocated_mean_mib=("gpu_peak_allocated_mib", "mean"),
            gpu_peak_allocated_max_mib=("gpu_peak_allocated_mib", "max"),
            gpu_incremental_peak_mean_mib=(
                "gpu_incremental_peak_allocated_mib",
                "mean",
            ),
            gpu_peak_reserved_mean_mib=("gpu_peak_reserved_mib", "mean"),
            model_parameter_mib=("model_parameter_mib", "first"),
            measurement_count=("aucroc", "count"),
        )
        .reset_index()
    )

    reference = summary[summary["num_loops"] == reference_depth][
        [
            "did",
            "aucroc_mean",
            "inference_mean_seconds",
            "end_to_end_mean_seconds",
            "gpu_peak_allocated_mean_mib",
            "gpu_incremental_peak_mean_mib",
        ]
    ].rename(
        columns={
            "aucroc_mean": "reference_aucroc",
            "inference_mean_seconds": "reference_inference_seconds",
            "end_to_end_mean_seconds": "reference_end_to_end_seconds",
            "gpu_peak_allocated_mean_mib": "reference_peak_allocated_mib",
            "gpu_incremental_peak_mean_mib": "reference_incremental_peak_mib",
        }
    )
    summary = summary.merge(reference, on="did", how="left")
    summary["reference_num_loops"] = reference_depth
    summary["aucroc_delta_vs_reference"] = (
        summary["aucroc_mean"] - summary["reference_aucroc"]
    )
    summary["inference_saving_vs_reference"] = 1.0 - (
        summary["inference_mean_seconds"]
        / summary["reference_inference_seconds"]
    )
    summary["end_to_end_saving_vs_reference"] = 1.0 - (
        summary["end_to_end_mean_seconds"]
        / summary["reference_end_to_end_seconds"]
    )
    summary["peak_allocated_saving_vs_reference"] = 1.0 - (
        summary["gpu_peak_allocated_mean_mib"]
        / summary["reference_peak_allocated_mib"]
    )
    summary["incremental_peak_saving_vs_reference"] = 1.0 - (
        summary["gpu_incremental_peak_mean_mib"]
        / summary["reference_incremental_peak_mib"]
    )
    return summary


def print_depth_summary(summary_df):
    if summary_df.empty:
        print("No successful measurements.")
        return

    depth_summary = (
        summary_df.groupby("num_loops")
        .agg(
            datasets=("did", "nunique"),
            aucroc_mean=("aucroc_mean", "mean"),
            inference_mean_seconds=("inference_mean_seconds", "mean"),
            inference_saving=("inference_saving_vs_reference", "mean"),
            peak_allocated_mean_mib=("gpu_peak_allocated_mean_mib", "mean"),
            peak_memory_saving=("peak_allocated_saving_vs_reference", "mean"),
        )
        .reset_index()
    )
    print("\nAggregate means across datasets:")
    print(depth_summary.to_string(index=False))


def main():
    args = parse_args()
    args.depths = list(dict.fromkeys(args.depths))
    args.splits = list(dict.fromkeys(args.splits))
    args.device = benchmark.normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not is_cuda(args.device):
        print(
            "CUDA is unavailable; GPU memory columns will be empty.",
            flush=True,
        )

    eval_helper = benchmark.EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = benchmark.prepare_datasets(dids)

    selected_names = [model_name(depth) for depth in args.depths]
    loaded_model = benchmark.load_model(selected_names[0], args.device)
    model, _ = loaded_model
    models = {name: loaded_model for name in selected_names}
    parameter_mib = to_mib(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    )
    print(
        f"Shared checkpoint parameter storage: {parameter_mib:.2f} MiB",
        flush=True,
    )

    skipped = run_warmups(models, datasets, args)
    raw_df = run_timed_measurements(
        models, datasets, args, skipped, parameter_mib
    )
    reference_depth = max(args.depths)
    summary_df = summarize(raw_df, reference_depth)

    benchmark.write_csv(raw_df, args.raw_csv)
    benchmark.write_csv(summary_df, args.summary_csv)

    print(f"\nReference depth for savings: {reference_depth}")
    print(f"Saved raw measurements to {args.raw_csv}")
    print(f"Saved per-dataset summary to {args.summary_csv}")
    print_depth_summary(summary_df)


if __name__ == "__main__":
    main()
