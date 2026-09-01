"""Measure synchronized model-only inference time on synthetic long contexts."""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_script_speed as benchmark


RAW_COLUMNS = [
    "model",
    "table_size",
    "num_features",
    "eval_position",
    "run",
    "status",
    "error",
    "synchronized_model_inference_seconds",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized model-only inference time for Hybrid 8L, "
            "Hydra 9L (16M), Hydra Small, and TabPFN on synthetic tables."
        )
    )
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(benchmark.MODEL_PATHS),
        choices=list(benchmark.MODEL_PATHS),
    )
    parser.add_argument(
        "--table-sizes",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "evaluation_model_only_speed_raw.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "evaluation_model_only_speed_summary.csv",
        ),
    )
    return parser.parse_args()


def measure_model_inference(model, model_name, dataset, table_size, device):
    _, model_inference_seconds = benchmark.run_single_measurement(
        model,
        model_name,
        dataset,
        table_size,
        device,
    )
    if model_inference_seconds is None:
        raise RuntimeError(
            f"Model-only inference time was not returned for {model_name}."
        )
    return model_inference_seconds


def run_warmups(models, datasets, args):
    print("Running warmup passes...", flush=True)
    skipped_pairs = set()
    for table_size, dataset in datasets.items():
        for model_name, model in models.items():
            for warmup_idx in range(args.warmup_runs):
                try:
                    measure_model_inference(
                        model,
                        model_name,
                        dataset,
                        table_size,
                        args.device,
                    )
                except RuntimeError as error:
                    if not benchmark.is_oom_error(error):
                        raise
                    benchmark.clear_device_cache(args.device)
                    skipped_pairs.add((model_name, table_size))
                    print(
                        f"skip warmup model={model_name} table_size={table_size} "
                        f"run={warmup_idx + 1} error={benchmark.format_error(error)}",
                        flush=True,
                    )
                    break
                print(
                    f"warmup model={model_name} table_size={table_size} "
                    f"run={warmup_idx + 1}",
                    flush=True,
                )
    return skipped_pairs


def make_failure_row(model_name, table_size, args, run, status, error):
    return {
        "model": model_name,
        "table_size": table_size,
        "num_features": args.num_features,
        "eval_position": table_size // 2,
        "run": run,
        "status": status,
        "error": error,
        "synchronized_model_inference_seconds": np.nan,
    }


def run_timed_measurements(models, datasets, args, skipped_pairs):
    rng = random.Random(args.seed)
    rows = [
        make_failure_row(
            model_name,
            table_size,
            args,
            run=0,
            status="skipped_warmup_oom",
            error="warmup CUDA out of memory",
        )
        for model_name, table_size in sorted(skipped_pairs)
    ]
    failed_pairs = set(skipped_pairs)

    for table_size, dataset in datasets.items():
        for run_idx in range(args.timed_runs):
            model_names = list(models)
            rng.shuffle(model_names)
            for model_name in model_names:
                if (model_name, table_size) in failed_pairs:
                    continue
                try:
                    inference_seconds = measure_model_inference(
                        models[model_name],
                        model_name,
                        dataset,
                        table_size,
                        args.device,
                    )
                except RuntimeError as error:
                    if not benchmark.is_oom_error(error):
                        raise
                    benchmark.clear_device_cache(args.device)
                    failed_pairs.add((model_name, table_size))
                    rows.append(
                        make_failure_row(
                            model_name,
                            table_size,
                            args,
                            run=run_idx + 1,
                            status="timed_oom",
                            error=benchmark.format_error(error),
                        )
                    )
                    print(
                        f"skip timed model={model_name} table_size={table_size} "
                        f"run={run_idx + 1} "
                        f"error={benchmark.format_error(error)}",
                        flush=True,
                    )
                    continue

                rows.append(
                    {
                        "model": model_name,
                        "table_size": table_size,
                        "num_features": args.num_features,
                        "eval_position": table_size // 2,
                        "run": run_idx + 1,
                        "status": "ok",
                        "error": "",
                        "synchronized_model_inference_seconds": inference_seconds,
                    }
                )
                print(
                    f"timed model={model_name} table_size={table_size} "
                    f"run={run_idx + 1} "
                    f"model_inference_seconds={inference_seconds:.6f}",
                    flush=True,
                )

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def summarize(raw_df):
    successful = raw_df[raw_df["status"] == "ok"]
    return (
        successful.groupby(
            ["model", "table_size", "num_features", "eval_position"]
        )
        .agg(
            inference_mean=("synchronized_model_inference_seconds", "mean"),
            inference_median=("synchronized_model_inference_seconds", "median"),
            inference_std=("synchronized_model_inference_seconds", "std"),
            inference_min=("synchronized_model_inference_seconds", "min"),
            inference_max=("synchronized_model_inference_seconds", "max"),
            count=("synchronized_model_inference_seconds", "count"),
        )
        .reset_index()
    )


def write_csv(data, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data.to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = benchmark.normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    datasets = {
        table_size: benchmark.make_dummy_dataset(
            table_size,
            args.num_features,
            args.num_classes,
            args.seed + table_size,
        )
        for table_size in args.table_sizes
    }
    models = {
        model_name: benchmark.load_model(model_name, args.device)[0]
        for model_name in args.models
    }

    skipped_pairs = run_warmups(models, datasets, args)
    raw_df = run_timed_measurements(models, datasets, args, skipped_pairs)
    summary_df = summarize(raw_df)

    write_csv(raw_df, args.raw_csv)
    write_csv(summary_df, args.summary_csv)

    print("\nSynchronized model-only inference seconds summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw model-only timings to {args.raw_csv}")
    print(f"Saved model-only summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
