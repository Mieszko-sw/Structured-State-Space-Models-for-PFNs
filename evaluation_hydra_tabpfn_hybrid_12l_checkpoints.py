import argparse
import os
import re
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

CHECKPOINT_DIR = Path("tabpfn/models_diff")
CHECKPOINT_MODEL_NAME = "hydra_tabpfn_hybrid_12_layers_512e_lr0p0001"
MODEL_LABEL = "hydra_tabpfn_hybrid_12l_512e_lr1e4"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current Hydra-TabPFN hybrid 12L training checkpoints "
            "on the same OpenML-CC18 benchmark setup used by evaluation_script.py."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(CHECKPOINT_DIR),
        help="Directory containing callback checkpoint .cpkt files.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=CHECKPOINT_MODEL_NAME,
        help="Checkpoint name used by epoch_callback, without the callback_ prefix.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=None,
        help="Specific numbered checkpoint epochs to evaluate. Defaults to all discovered epochs.",
    )
    parser.add_argument(
        "--skip-latest",
        action="store_true",
        help="Do not evaluate the callback latest checkpoint.",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also evaluate the final non-callback model file if it exists.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip requested checkpoints that are not present instead of failing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints already present in the summary CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the checkpoints that would be evaluated, then exit.",
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "hydra_tabpfn_hybrid_12l_checkpoint_benchmark_summary.csv",
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        default=os.path.join(
            "result_csvs",
            "hydra_tabpfn_hybrid_12l_checkpoint_benchmark_by_dataset.csv",
        ),
    )
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    return device


def discover_checkpoint_epochs(checkpoint_dir, checkpoint_name):
    pattern = re.compile(rf"callback_{re.escape(checkpoint_name)}_epoch_(\d+)\.cpkt$")
    epochs = []
    for checkpoint_path in checkpoint_dir.glob(f"callback_{checkpoint_name}_epoch_*.cpkt"):
        match = pattern.match(checkpoint_path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def checkpoint_specs(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_name = args.checkpoint_name
    epochs = args.epochs
    if epochs is None:
        epochs = discover_checkpoint_epochs(checkpoint_dir, checkpoint_name)

    specs = []
    for epoch in epochs:
        specs.append(
            {
                "model": MODEL_LABEL,
                "checkpoint": f"epoch_{epoch}",
                "epoch": epoch,
                "path": str(checkpoint_dir / f"callback_{checkpoint_name}_epoch_{epoch}.cpkt"),
            }
        )

    if not args.skip_latest:
        specs.append(
            {
                "model": MODEL_LABEL,
                "checkpoint": "latest",
                "epoch": None,
                "path": str(checkpoint_dir / f"callback_{checkpoint_name}_latest.cpkt"),
            }
        )

    if args.include_final:
        specs.append(
            {
                "model": MODEL_LABEL,
                "checkpoint": "final",
                "epoch": None,
                "path": str(checkpoint_dir / f"{checkpoint_name}_12l.cpkt"),
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


def extract_inference_time(split_result):
    time_suffixes = tuple(f"_time_at_{pos}" for pos in split_result["eval_positions"])
    return sum(
        float(value)
        for key, value in split_result.items()
        if key.endswith(time_suffixes)
    )


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
    split_metric_means = []
    split_time_means = []
    for split_idx in range(len(SPLIT_NUMBERS)):
        split_metrics = [extract_metric(split_results[split_idx]) for split_results in result.values()]
        split_times = [extract_inference_time(split_results[split_idx]) for split_results in result.values()]
        split_metric_means.append(sum(split_metrics) / len(split_metrics))
        split_time_means.append(sum(split_times) / len(split_times))

    summary_row = {
        "model": spec["model"],
        "checkpoint": spec["checkpoint"],
        "epoch": spec["epoch"],
        "path": spec["path"],
        "mean_metric": sum(split_metric_means) / len(split_metric_means),
        "metric_moe": calc_moe(split_metric_means),
        "mean_inference_time_seconds": sum(split_time_means) / len(split_time_means),
        "inference_time_moe_seconds": calc_moe(split_time_means),
    }
    for split_number, split_metric, split_time in zip(
        SPLIT_NUMBERS,
        split_metric_means,
        split_time_means,
    ):
        summary_row[f"split_{split_number}_mean_metric"] = split_metric
        summary_row[f"split_{split_number}_mean_inference_time_seconds"] = split_time

    dataset_rows = []
    for did, split_results in result.items():
        split_metrics = [extract_metric(split_result) for split_result in split_results]
        split_times = [extract_inference_time(split_result) for split_result in split_results]
        row = {
            "model": spec["model"],
            "checkpoint": spec["checkpoint"],
            "epoch": spec["epoch"],
            "path": spec["path"],
            "did": did,
            "mean_metric": sum(split_metrics) / len(split_metrics),
            "mean_inference_time_seconds": sum(split_times) / len(split_times),
        }
        for split_number, split_metric, split_time in zip(
            SPLIT_NUMBERS,
            split_metrics,
            split_times,
        ):
            row[f"split_{split_number}_metric"] = split_metric
            row[f"split_{split_number}_inference_time_seconds"] = split_time
        dataset_rows.append(row)

    return summary_row, dataset_rows


def read_existing_rows(summary_csv, dataset_csv):
    summary_rows = []
    dataset_rows = []
    if Path(summary_csv).is_file():
        summary_rows = pd.read_csv(summary_csv).to_dict("records")
    if Path(dataset_csv).is_file():
        dataset_rows = pd.read_csv(dataset_csv).to_dict("records")
    return summary_rows, dataset_rows


def row_key(row):
    return str(row["path"])


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)

    specs = checkpoint_specs(args)
    if not specs:
        raise RuntimeError(
            f"No checkpoints found for {args.checkpoint_name} in {args.checkpoint_dir}"
        )

    if args.skip_missing:
        specs = [spec for spec in specs if Path(spec["path"]).is_file()]
        if not specs:
            raise RuntimeError("No requested checkpoints exist after applying --skip-missing.")
    else:
        missing = [spec["path"] for spec in specs if not Path(spec["path"]).is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    if args.resume:
        summary_rows, dataset_rows = read_existing_rows(args.summary_csv, args.dataset_csv)
        completed_paths = {row_key(row) for row in summary_rows}
        specs = [spec for spec in specs if spec["path"] not in completed_paths]
        if not specs:
            print("All requested checkpoints are already present in the summary CSV.")
            return
    else:
        summary_rows = []
        dataset_rows = []

    if args.dry_run:
        for spec in specs:
            print(f"{spec['checkpoint']}: {spec['path']}")
        return

    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] Evaluating {spec['checkpoint']} from {spec['path']}",
            flush=True,
        )
        result = evaluate_checkpoint(spec, args.device)
        summary_row, checkpoint_dataset_rows = rows_for_checkpoint(spec, result)
        summary_rows.append(summary_row)
        dataset_rows.extend(checkpoint_dataset_rows)
        write_csv(summary_rows, args.summary_csv)
        write_csv(dataset_rows, args.dataset_csv)
        print(
            f"{spec['checkpoint']} mean_metric={summary_row['mean_metric']:.6f} "
            f"metric_moe={summary_row['metric_moe']:.6f} "
            f"mean_inference_time_seconds={summary_row['mean_inference_time_seconds']:.6f}",
            flush=True,
        )

    print(f"Saved checkpoint summary to {args.summary_csv}")
    print(f"Saved per-dataset checkpoint results to {args.dataset_csv}")


if __name__ == "__main__":
    main()
