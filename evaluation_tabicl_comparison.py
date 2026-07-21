"""Compare Hybrid 8L, looped TabPFN, TabPFN, Hydra, and TabICL.

This is a standalone counterpart to ``evaluation_script.py``.  The four local
PyTorch checkpoints use the repository's native evaluation path.  TabICL uses
the same datasets and deterministic train/test splits, but is evaluated through
its scikit-learn ``fit``/``predict_proba`` interface.

Install TabICL before selecting it::

    pip install tabicl

The first TabICL run may download its checkpoint.  That download is outside the
reported per-dataset inference time.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.tabular_evaluation import generate_valid_split
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


MODELS = {
    "hybrid_8l": {
        "path": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
    "looped": {
        "path": "tabpfn/models_diff/new_looped_transformer_6physical_core4x2_10l.cpkt",
        "loader_type": "looped_transformer",
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
    "tabicl": {
        "loader_type": "tabicl",
    },
}

EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}
METRIC_USED = tabular_metrics.auc_metric
BPTT = 1024
DEFAULT_EVAL_POSITIONS = [972]
SPLIT_NUMBERS = [1, 2, 3, 4, 5]
CONFIDENCE_LEVEL = 0.95


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument(
        "--evaluation-type",
        default="openmlcc18",
        help="openmlcc18, openmlcc18_large, test, dummy, or a single OpenML DID",
    )
    parser.add_argument("--splits", nargs="+", type=int, default=SPLIT_NUMBERS)
    parser.add_argument("--tabicl-n-estimators", type=int, default=8)
    parser.add_argument("--tabicl-batch-size", type=int, default=8)
    parser.add_argument("--tabicl-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--result-csv",
        default=os.path.join("result_csvs", "tabicl_comparison_eval.csv"),
    )
    parser.add_argument(
        "--time-csv",
        default=os.path.join("result_csvs", "tabicl_comparison_inference_time.csv"),
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


def print_parameter_count(model_name, model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"{model_name}: {total:,} parameters ({trainable:,} trainable)", flush=True)


def load_native_model(model_name, device):
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
    print_parameter_count(model_name, model)
    return model, config


def run_native_evaluation(eval_helper, model_name, model, config, args):
    return eval_helper.do_evaluation_custom(
        model,
        bptt=BPTT,
        eval_positions=config["eval_positions"],
        metric=METRIC_USED,
        device=args.device,
        method_name=MODELS[model_name]["method_name"],
        evaluation_type=parse_evaluation_type(args.evaluation_type),
        split_numbers=args.splits,
        jrt_prompt=False,
        single_evaluation_prompt=False,
        permutation_bagging=1,
        sample_bagging=0,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )


def parse_evaluation_type(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def evaluation_dids(eval_helper, evaluation_type):
    evaluation_type = parse_evaluation_type(evaluation_type)
    if evaluation_type == "openmlcc18":
        return eval_helper.openml_cc18_dids_small
    if evaluation_type == "openmlcc18_large":
        return eval_helper.openml_cc18_dids_large
    if evaluation_type == "test":
        return eval_helper.test_dids_classification
    if isinstance(evaluation_type, int):
        return [evaluation_type]
    raise ValueError("TabICL supports openmlcc18, openmlcc18_large, test, or one DID")


def prepare_tabicl_datasets(eval_helper, evaluation_type):
    dids = evaluation_dids(eval_helper, evaluation_type)
    eval_helper.check_datasets_data(dids)
    eval_helper.make_limit_datasets(
        max_classes=10,
        max_features=100,
        limit_dids=dids,
        eval_filters=EVALUATION_TYPE_FILTERS,
    )
    return {did: eval_helper.limit_dict[did] for did in dids if did in eval_helper.limit_dict}


def make_tabicl(args):
    try:
        from tabicl import TabICLClassifier
    except ImportError as error:
        raise RuntimeError(
            "TabICL is not installed. Install it with `pip install tabicl`, "
            "or omit `tabicl` from --models."
        ) from error

    kwargs = {
        "device": args.device,
        "n_estimators": args.tabicl_n_estimators,
        "batch_size": args.tabicl_batch_size,
        "random_state": args.seed,
    }
    if args.tabicl_checkpoint:
        kwargs["model_path"] = args.tabicl_checkpoint
        kwargs["allow_auto_download"] = False
    return TabICLClassifier(**kwargs)


def evaluate_tabicl_split(classifier, dataset, configured_eval_position, split_number, args):
    dataset_name, x, y, _, _, _ = dataset
    dataset_bptt = min(len(x), BPTT)
    eval_position = (
        int(dataset_bptt * 0.5)
        if 2 * configured_eval_position > dataset_bptt
        else configured_eval_position
    )
    split_bptt = int(eval_position * 2)
    eval_xs, eval_ys = generate_valid_split(
        x,
        y,
        split_bptt,
        eval_position,
        is_classification=True,
        split_number=split_number,
    )
    if eval_xs is None:
        raise RuntimeError(f"Could not generate split {split_number} for {dataset_name}")

    # Match tabular_evaluation.evaluate_position's contiguous class encoding.
    eval_ys = (eval_ys > torch.unique(eval_ys).unsqueeze(0)).sum(axis=1).unsqueeze(-1)
    x_all = eval_xs[:, 0].cpu().numpy()
    y_all = eval_ys[:, 0, 0].cpu().numpy()
    synchronize(args.device)
    start = time.perf_counter()
    classifier.fit(x_all[:eval_position], y_all[:eval_position])
    probabilities = classifier.predict_proba(x_all[eval_position:])
    synchronize(args.device)
    elapsed = time.perf_counter() - start

    outputs = torch.as_tensor(probabilities, dtype=torch.float32)
    targets = torch.as_tensor(y_all[eval_position:], dtype=torch.long)
    metric = METRIC_USED(targets, outputs)
    return {
        "metric_used": "roc",
        "bptt": BPTT,
        "eval_positions": [configured_eval_position],
        f"{dataset_name}_outputs_at_{configured_eval_position}": outputs,
        f"{dataset_name}_ys_at_{configured_eval_position}": targets.unsqueeze(0),
        f"{dataset_name}_time_at_{configured_eval_position}": elapsed,
        "last_outputs": outputs,
        "mean_metric": metric,
    }


def run_tabicl_evaluation(eval_helper, eval_positions, args):
    if len(eval_positions) != 1:
        raise ValueError(f"Expected one evaluation position, got {eval_positions}")
    datasets = prepare_tabicl_datasets(eval_helper, args.evaluation_type)

    # Materialize/download weights before timing the first dataset. Reusing the
    # estimator also matches the native path, where model loading happens once.
    print("Preparing TabICL checkpoint...", flush=True)
    classifier = make_tabicl(args)
    warmup_x = np.arange(80, dtype=np.float32).reshape(16, 5)
    warmup_y = np.arange(16) % 2
    classifier.fit(warmup_x[:8], warmup_y[:8])
    classifier.predict_proba(warmup_x[8:])
    synchronize(args.device)

    results = {}
    for did, dataset_list in datasets.items():
        results[did] = []
        for split_number in args.splits:
            print(f"TabICL: DID {did}, split {split_number}", flush=True)
            results[did].append(
                evaluate_tabicl_split(
                    classifier,
                    dataset_list[0],
                    eval_positions[0],
                    split_number,
                    args,
                )
            )
    return results


def extract_metric(split_result):
    value = split_result["mean_metric"]
    return value.item() if hasattr(value, "item") else float(value)


def extract_inference_time(split_result):
    suffixes = tuple(f"_time_at_{position}" for position in split_result["eval_positions"])
    return sum(value for key, value in split_result.items() if key.endswith(suffixes))


def margin_of_error(values):
    if len(values) < 2:
        return float("nan")
    return stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, len(values) - 1) * stats.sem(values)


def print_stats(results, methods, splits):
    for method in methods:
        metric_means = []
        time_means = []
        for split_index, _ in enumerate(splits):
            dataset_results = results[method].values()
            metric_means.append(np.mean([extract_metric(value[split_index]) for value in dataset_results]))
            dataset_results = results[method].values()
            time_means.append(np.mean([extract_inference_time(value[split_index]) for value in dataset_results]))
        print(f"\n{method} AUC: {np.mean(metric_means):.6f} (95% MOE {margin_of_error(metric_means):.6f})")
        print(f"{method} inference seconds: {np.mean(time_means):.6f} (95% MOE {margin_of_error(time_means):.6f})")


def save_csv(results, methods, path, extractor):
    common_dids = set.intersection(*(set(results[method]) for method in methods))
    rows = []
    for did in sorted(common_dids):
        row = {"did": did}
        for method in methods:
            row[method] = np.mean([extractor(value) for value in results[method][did]])
        rows.append(row)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(rows, columns=["did", *methods]).to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_helper = EvalHelper()
    results = {}
    tabicl_eval_positions = DEFAULT_EVAL_POSITIONS
    for model_name in args.models:
        if model_name == "tabicl":
            results[model_name] = run_tabicl_evaluation(eval_helper, tabicl_eval_positions, args)
            continue
        model, config = load_native_model(model_name, args.device)
        results[model_name] = run_native_evaluation(eval_helper, model_name, model, config, args)
        tabicl_eval_positions = config["eval_positions"]

    print_stats(results, args.models, args.splits)
    save_csv(results, args.models, args.result_csv, extract_metric)
    save_csv(results, args.models, args.time_csv, extract_inference_time)
    print(f"\nSaved results to {args.result_csv}")
    print(f"Saved inference times to {args.time_csv}")


if __name__ == "__main__":
    main()
