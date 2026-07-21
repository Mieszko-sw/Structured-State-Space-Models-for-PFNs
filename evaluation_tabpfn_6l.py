import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from scipy import stats

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.model_builder_custom import load_model_only_inference


DEFAULT_CHECKPOINT = "tabpfn/models_diff/callback_original_transformer_6l_latest.cpkt"
DEFAULT_SCORE_CSV = os.path.join("result_csvs", "tabpfn_6l_eval.csv")
DEFAULT_TIME_CSV = os.path.join("result_csvs", "tabpfn_6l_eval_inference_time.csv")

EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}
CONFIDENCE_LEVEL = 0.95


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the current 6-layer TabPFN Transformer checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the 6L TabPFN checkpoint to evaluate.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--evaluation-type",
        default="openmlcc18",
        help="Eval set name: openmlcc18, openmlcc18_large, test, dummy, or a single OpenML DID.",
    )
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--permutation-bagging", type=int, default=1)
    parser.add_argument("--sample-bagging", type=int, default=0)
    parser.add_argument("--jrt-prompt", action="store_true")
    parser.add_argument("--single-evaluation-prompt", action="store_true")
    parser.add_argument("--score-csv", default=DEFAULT_SCORE_CSV)
    parser.add_argument("--time-csv", default=DEFAULT_TIME_CSV)
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    return device


def print_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"tabpfn_6l parameters: {total_params:,} total ({trainable_params:,} trainable)",
        flush=True,
    )


def extract_metric(split_result):
    metric = split_result["mean_metric"]
    if hasattr(metric, "item"):
        return metric.item()
    return float(metric)


def extract_inference_time(split_result):
    time_keys = tuple(f"_time_at_{pos}" for pos in split_result["eval_positions"])
    return sum(
        value
        for key, value in split_result.items()
        if key.endswith(time_keys)
    )


def calc_moe(values):
    if len(values) < 2:
        return 0.0
    sem = stats.sem(values)
    degrees_of_freedom = len(values) - 1
    t_score = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, degrees_of_freedom)
    return t_score * sem


def run_evaluation(model, config, args):
    eval_helper = EvalHelper()
    return eval_helper.do_evaluation_custom(
        model,
        bptt=args.bptt,
        eval_positions=config["eval_positions"],
        metric=tabular_metrics.auc_metric,
        device=args.device,
        method_name="transformer",
        evaluation_type=args.evaluation_type,
        max_classes=args.max_classes,
        max_features=args.max_features,
        max_time=args.max_time,
        split_numbers=args.splits,
        jrt_prompt=args.jrt_prompt,
        single_evaluation_prompt=args.single_evaluation_prompt,
        permutation_bagging=args.permutation_bagging,
        sample_bagging=args.sample_bagging,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )


def summarize_results(result, splits):
    split_score_means = []
    split_time_means = []

    for split_index in range(len(splits)):
        split_scores = [
            extract_metric(split_results[split_index])
            for split_results in result.values()
        ]
        split_times = [
            extract_inference_time(split_results[split_index])
            for split_results in result.values()
        ]
        split_score_means.append(sum(split_scores) / len(split_scores))
        split_time_means.append(sum(split_times) / len(split_times))

    score_summary = {
        "mean_metric": sum(split_score_means) / len(split_score_means),
        "moe": calc_moe(split_score_means),
    }
    time_summary = {
        "mean_seconds": sum(split_time_means) / len(split_time_means),
        "moe_seconds": calc_moe(split_time_means),
    }

    return score_summary, time_summary


def dataset_rows(result, splits, value_fn, value_column):
    rows = []
    for did, split_results in result.items():
        split_values = [value_fn(split_result) for split_result in split_results]
        row = {
            "did": did,
            value_column: sum(split_values) / len(split_values),
        }
        for split_number, split_value in zip(splits, split_values):
            row[f"split_{split_number}_{value_column}"] = split_value
        rows.append(row)
    return rows


def write_csv(rows, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    print(f"Loading tabpfn_6l checkpoint from {checkpoint}", flush=True)
    loaded, config = load_model_only_inference(
        ".",
        str(checkpoint),
        args.device,
        model_name="transformer",
    )
    model = loaded[2]
    print_parameter_count(model)
    print(
        f"Config: nlayers={config.get('nlayers')} emsize={config.get('emsize')} "
        f"eval_positions={config.get('eval_positions')}",
        flush=True,
    )

    result = run_evaluation(model, config, args)
    score_summary, time_summary = summarize_results(result, args.splits)

    score_rows = dataset_rows(result, args.splits, extract_metric, "mean_metric")
    time_rows = dataset_rows(result, args.splits, extract_inference_time, "mean_seconds")
    write_csv(score_rows, args.score_csv)
    write_csv(time_rows, args.time_csv)

    print(
        f"tabpfn_6l mean_metric={score_summary['mean_metric']:.6f} "
        f"moe={score_summary['moe']:.6f}",
        flush=True,
    )
    print(
        f"tabpfn_6l mean_seconds={time_summary['mean_seconds']:.6f} "
        f"moe_seconds={time_summary['moe_seconds']:.6f}",
        flush=True,
    )
    print(f"Saved scores to {args.score_csv}", flush=True)
    print(f"Saved inference times to {args.time_csv}", flush=True)


if __name__ == "__main__":
    main()
