import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from scipy import stats

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.model_builder_custom import load_model_only_inference


EVALUATION_TYPE = "openmlcc18"
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}

METRIC_USED = tabular_metrics.auc_metric
SPLIT_NUMBERS = [1, 2, 3, 4, 5]
BPTT = 1024
CONFIDENCE_LEVEL = 0.95
JRT_PROMPT = False
SINGLE_EVAL_PROMPT = False
PERMUTATION_BAGGING = 1
SAMPLE_BAGGING = 0

CHECKPOINT_SERIES = {
    "hybrid_8_layers": {
        "checkpoint_pattern": "tabpfn/models_diff/callback_hybrid_8_layers_epoch_{epoch}.cpkt",
        "final": "tabpfn/models_diff/hybrid_8_layers_8l.cpkt",
        "latest": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
    },
    "hybrid_12_layers_lr1e4": {
        "checkpoint_pattern": "tabpfn/models_diff/callback_alternating_hydra_tabpfn_12_layers_512e_lr0p0001_epoch_{epoch}.cpkt",
        "final": "tabpfn/models_diff/alternating_hydra_tabpfn_12_layers_512e_lr0p0001_12l.cpkt",
        "latest": "tabpfn/models_diff/callback_alternating_hydra_tabpfn_12_layers_512e_lr0p0001_latest.cpkt",
    },
    "hybrid_12_layers_lr1e5_old": {
        "checkpoint_pattern": "tabpfn/models_diff/callback_hybrid_6hydra_6transformer_epoch_{epoch}.cpkt",
        "final": "tabpfn/models_diff/hybrid_6hydra_6transformer_12l.cpkt",
        "latest": "tabpfn/models_diff/callback_hybrid_6hydra_6transformer_latest.cpkt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy across checkpoints for 8L and 12L hybrid models."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(CHECKPOINT_SERIES),
        choices=list(CHECKPOINT_SERIES),
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=list(range(20, 201, 20)),
        help="Numbered checkpoint epochs to evaluate.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Do not evaluate the final saved model files after the numbered checkpoints.",
    )
    parser.add_argument("--include-latest", action="store_true")
    parser.add_argument(
        "--summary-csv",
        default=os.path.join("result_csvs", "hybrid_checkpoint_accuracy_summary.csv"),
    )
    parser.add_argument(
        "--dataset-csv",
        default=os.path.join("result_csvs", "hybrid_checkpoint_accuracy_by_dataset.csv"),
    )
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    return device


def checkpoint_specs(model_names, epochs, include_final, include_latest):
    specs = []
    for model_name in model_names:
        series = CHECKPOINT_SERIES[model_name]
        for epoch in epochs:
            checkpoint_path = series["checkpoint_pattern"].format(epoch=epoch)
            specs.append(
                {
                    "model": model_name,
                    "checkpoint": f"epoch_{epoch}",
                    "epoch": epoch,
                    "path": checkpoint_path,
                }
            )
        if include_latest:
            specs.append(
                {
                    "model": model_name,
                    "checkpoint": "latest",
                    "epoch": None,
                    "path": series["latest"],
                }
            )
        if include_final:
            specs.append(
                {
                    "model": model_name,
                    "checkpoint": "final",
                    "epoch": None,
                    "path": series["final"],
                }
            )
    return specs


def print_parameter_count(model_name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} parameters: {total_params:,} total ({trainable_params:,} trainable)")


def calc_moe(data):
    sem = stats.sem(data)
    degrees_of_freedom = len(data) - 1
    t_score = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, degrees_of_freedom)
    return t_score * sem


def extract_metric(split_result):
    metric = split_result["mean_metric"]
    if hasattr(metric, "item"):
        return metric.item()
    return float(metric)


def run_model_evaluation(model, config, device):
    eval_helper = EvalHelper()
    return eval_helper.do_evaluation_custom(
        model,
        bptt=BPTT,
        eval_positions=config["eval_positions"],
        metric=METRIC_USED,
        device=device,
        method_name="transformer",
        evaluation_type=EVALUATION_TYPE,
        split_numbers=SPLIT_NUMBERS,
        jrt_prompt=JRT_PROMPT,
        single_evaluation_prompt=SINGLE_EVAL_PROMPT,
        permutation_bagging=PERMUTATION_BAGGING,
        sample_bagging=SAMPLE_BAGGING,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )


def evaluate_checkpoint(spec, device):
    checkpoint_path = Path(spec["path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    loaded, config = load_model_only_inference(
        ".",
        str(checkpoint_path),
        device,
        model_name="hybrid",
    )
    model = loaded[2]
    print_parameter_count(f"{spec['model']} {spec['checkpoint']}", model)
    try:
        return run_model_evaluation(model, config, device)
    finally:
        del model
        del loaded
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def rows_for_checkpoint(spec, result):
    split_means = []
    for split_idx in range(len(SPLIT_NUMBERS)):
        split_metrics = [extract_metric(split_results[split_idx]) for split_results in result.values()]
        split_means.append(sum(split_metrics) / len(split_metrics))

    summary_row = {
        "model": spec["model"],
        "checkpoint": spec["checkpoint"],
        "epoch": spec["epoch"],
        "path": spec["path"],
        "mean_metric": sum(split_means) / len(split_means),
        "moe": calc_moe(split_means),
    }
    for split_number, split_mean in zip(SPLIT_NUMBERS, split_means):
        summary_row[f"split_{split_number}_mean_metric"] = split_mean

    dataset_rows = []
    for did, split_results in result.items():
        split_metrics = [extract_metric(split_result) for split_result in split_results]
        row = {
            "model": spec["model"],
            "checkpoint": spec["checkpoint"],
            "epoch": spec["epoch"],
            "path": spec["path"],
            "did": did,
            "mean_metric": sum(split_metrics) / len(split_metrics),
        }
        for split_number, split_metric in zip(SPLIT_NUMBERS, split_metrics):
            row[f"split_{split_number}_metric"] = split_metric
        dataset_rows.append(row)

    return summary_row, dataset_rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)

    specs = checkpoint_specs(args.models, args.epochs, not args.skip_final, args.include_latest)
    summary_rows = []
    dataset_rows = []

    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] Evaluating {spec['model']} {spec['checkpoint']} from {spec['path']}",
            flush=True,
        )
        result = evaluate_checkpoint(spec, args.device)
        summary_row, checkpoint_dataset_rows = rows_for_checkpoint(spec, result)
        summary_rows.append(summary_row)
        dataset_rows.extend(checkpoint_dataset_rows)
        write_csv(summary_rows, args.summary_csv)
        write_csv(dataset_rows, args.dataset_csv)
        print(
            f"{spec['model']} {spec['checkpoint']} mean_metric={summary_row['mean_metric']:.6f} "
            f"moe={summary_row['moe']:.6f}",
            flush=True,
        )

    print(f"Saved checkpoint summary to {args.summary_csv}")
    print(f"Saved per-dataset checkpoint results to {args.dataset_csv}")


if __name__ == "__main__":
    main()
