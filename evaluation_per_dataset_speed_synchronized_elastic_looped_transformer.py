"""Benchmark every inference depth of the elastic 12-loop Transformer."""

import argparse
import os
import random
import time

import evaluation_per_dataset_speed_synchronized as benchmark
import numpy as np
import torch


MAX_LOOPS = 12
MODEL_PATH = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_1physical_core12x_latest.cpkt"
)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized per-dataset inference speed and AUC-ROC at "
            "each requested depth of the elastic one-physical-layer Transformer."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=list(range(1, MAX_LOOPS + 1)),
        choices=range(1, MAX_LOOPS + 1),
        metavar="N",
        help="Loop depths to benchmark (default: every depth from 1 through 12).",
    )
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
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_elastic_looped_transformer_raw.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_elastic_looped_transformer_summary.csv",
        ),
    )
    return parser.parse_args()


def run_single_measurement(model, model_label, config, dataset_list, device, split_number):
    """Run the shared benchmark at the loop budget encoded by model_label."""
    dataset_name = dataset_list[0][0]
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(f"Expected one evaluation position, got {eval_positions}")
    configured_eval_position = eval_positions[0]
    effective_eval_position = benchmark.real_eval_position(
        dataset_list, configured_eval_position
    )

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

    return (
        elapsed_seconds,
        benchmark.extract_model_inference_time(
            result, dataset_name, configured_eval_position
        ),
        benchmark.extract_metric(result),
        configured_eval_position,
        effective_eval_position,
    )


def add_depth_column(frame):
    """Add an explicit numeric depth column while retaining shared CSV columns."""
    frame.insert(
        frame.columns.get_loc("model") + 1,
        "num_loops",
        frame["model"].map(lambda name: MODELS[name]["num_loops"]),
    )
    return frame


# Reuse the benchmark mechanics while replacing its model registry and CLI.
benchmark.MODELS = MODELS
benchmark.parse_args = parse_args
benchmark.run_single_measurement = run_single_measurement


def main():
    args = parse_args()
    args.device = benchmark.normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = benchmark.EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = benchmark.prepare_datasets(dids)
    selected_names = [model_name(depth) for depth in args.depths]

    # Every depth shares the same parameters. Load the checkpoint once and use
    # the per-label num_loops value only to control recurrent compute.
    loaded_model = benchmark.load_model(selected_names[0], args.device)
    models = {name: loaded_model for name in selected_names}

    skipped = benchmark.run_warmups(models, datasets, args)
    raw_df = benchmark.run_timed_measurements(models, datasets, args, skipped)
    summary_df = benchmark.summarize(raw_df)
    raw_df = add_depth_column(raw_df)
    summary_df = add_depth_column(summary_df)

    benchmark.write_csv(raw_df, args.raw_csv)
    benchmark.write_csv(summary_df, args.summary_csv)

    print("\nSynchronized per-dataset inference summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved summary to {args.summary_csv}")

    print("\nAverage AUC-ROC by inference depth:")
    if summary_df.empty:
        print("No successful measurements.")
    else:
        average_metrics = summary_df.groupby("num_loops")["mean_metric"].mean()
        for num_loops, average_metric in average_metrics.items():
            print(f"depth {num_loops:2d}: {average_metric:.6f}")


if __name__ == "__main__":
    main()
