"""Focused diagnostics for the four main models on OpenML DID 1063 (kc2).

The script deliberately uses the same EvalHelper/evaluate path as the OpenML
benchmark scripts in this repository.  In addition to the benchmark AUC, it
exports metrics that help distinguish ranking, thresholding, calibration,
split-shift, and model-disagreement effects.

Example:
    python3 diagnose_did_1063_models.py --device cuda:0
"""

import argparse
import gc
import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.tabular_evaluation import generate_valid_split
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


DID = 1063
MODEL_SPECS = {
    "hybrid_8l": {
        "path": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
        "loader": "custom",
        "model_type": "hybrid",
        "prediction_method": "transformer",
    },
    "hydra_small": {
        "path": "tabpfn/models_diff/hydra_small.cpkt",
        "loader": "hydra_workflow",
        "prediction_method": "hydra",
    },
    "hydra": {
        "path": "tabpfn/models_diff/callback_pure_hydra_12_layers_512e_latest.cpkt",
        "loader": "custom",
        "model_type": "hydra",
        "prediction_method": "hydra",
    },
    "tabpfn": {
        "path": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
        "loader": "transformer_workflow",
        "prediction_method": "transformer",
    },
}

EVAL_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the performance outlier on OpenML DID 1063 for hybrid_8l, "
            "hydra_small, hydra, and TabPFN."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--eval-position", type=int, default=972)
    parser.add_argument("--permutation-bagging", type=int, default=1)
    parser.add_argument("--sample-bagging", type=int, default=0)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("result_csvs", "did_1063_diagnostics"),
    )
    parser.add_argument(
        "--hybrid-8l-path",
        default=MODEL_SPECS["hybrid_8l"]["path"],
    )
    parser.add_argument(
        "--hydra-small-path",
        default=MODEL_SPECS["hydra_small"]["path"],
    )
    parser.add_argument("--hydra-path", default=MODEL_SPECS["hydra"]["path"])
    parser.add_argument("--tabpfn-path", default=MODEL_SPECS["tabpfn"]["path"])
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(torch.device(device))
    return device


def resolved_specs(args):
    paths = {
        "hybrid_8l": args.hybrid_8l_path,
        "hydra_small": args.hydra_small_path,
        "hydra": args.hydra_path,
        "tabpfn": args.tabpfn_path,
    }
    specs = {}
    for model_name, spec in MODEL_SPECS.items():
        specs[model_name] = dict(spec)
        specs[model_name]["path"] = paths[model_name]
        if not os.path.isfile(paths[model_name]):
            raise FileNotFoundError(
                f"Checkpoint for {model_name} does not exist: {paths[model_name]}"
            )
    return specs


def load_model(model_name, spec, device):
    if spec["loader"] == "custom":
        loaded, config = load_model_only_inference(
            ".",
            spec["path"],
            device,
            model_name=spec["model_type"],
        )
    elif spec["loader"] == "hydra_workflow":
        loaded, config, _ = hydra_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=spec["path"],
        )
    elif spec["loader"] == "transformer_workflow":
        loaded, config, _ = transformer_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=spec["path"],
        )
    else:
        raise ValueError(f"Unknown loader for {model_name}: {spec['loader']}")

    model = loaded[2]
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"{model_name}: {parameter_count:,} parameters from {spec['path']}",
        flush=True,
    )
    return model, config, parameter_count


def prepare_dataset(eval_helper):
    eval_helper.check_datasets_data([DID])
    eval_helper.make_limit_datasets(
        max_classes=10,
        max_features=100,
        limit_dids=[DID],
        eval_filters=EVAL_FILTERS,
    )
    if DID not in eval_helper.limit_dict:
        raise RuntimeError(f"DID {DID} was removed by the evaluation filters")
    return eval_helper.limit_dict[DID][0]


def dataset_profile(dataset):
    dataset_name, x, y, categorical_features, _, _ = dataset
    x_np = torch.as_tensor(x).cpu().numpy()
    y_np = torch.as_tensor(y).cpu().numpy().reshape(-1)
    x_frame = pd.DataFrame(x_np)
    class_values, class_counts = np.unique(y_np, return_counts=True)

    duplicate_feature_rows = int(x_frame.duplicated(keep=False).sum())
    conflicting_duplicate_rows = 0
    if duplicate_feature_rows:
        combined = x_frame.copy()
        combined["_target"] = y_np
        group_columns = list(x_frame.columns)
        conflicting = combined.groupby(group_columns, dropna=False)["_target"].transform(
            "nunique"
        )
        conflicting_duplicate_rows = int((conflicting > 1).sum())

    profile = {
        "did": DID,
        "dataset": dataset_name,
        "n_rows_after_filters": len(x_np),
        "n_features_after_filters": x_np.shape[1],
        "n_classes": len(class_values),
        "class_values": "|".join(str(value) for value in class_values),
        "class_counts": "|".join(str(int(value)) for value in class_counts),
        "minority_fraction": float(class_counts.min() / class_counts.sum()),
        "categorical_feature_count": len(categorical_features or []),
        "rows_with_nan": int(np.isnan(x_np).any(axis=1).sum()),
        "constant_feature_count": int(
            sum(pd.Series(x_np[:, index]).nunique(dropna=True) <= 1 for index in range(x_np.shape[1]))
        ),
        "duplicate_feature_rows": duplicate_feature_rows,
        "conflicting_duplicate_rows": conflicting_duplicate_rows,
    }
    return pd.DataFrame([profile])


def actual_split(dataset, bptt, requested_eval_position, split_number):
    _, x, y, _, _, _ = dataset
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    dataset_bptt = min(len(x), bptt)
    eval_position = (
        dataset_bptt // 2
        if 2 * requested_eval_position > dataset_bptt
        else requested_eval_position
    )
    split_bptt = 2 * eval_position
    eval_x, eval_y = generate_valid_split(
        x,
        y,
        split_bptt,
        eval_position,
        is_classification=True,
        split_number=split_number,
    )
    if eval_x is None:
        raise RuntimeError(f"Could not generate benchmark split {split_number}")
    return eval_x[:, 0].cpu(), eval_y[:, 0].cpu(), eval_position


def encoded_labels(y):
    unique = torch.unique(y)
    return (y.unsqueeze(-1) > unique.unsqueeze(0)).sum(dim=1).long()


def split_profile_rows(dataset, args):
    rows = []
    for split_number in args.splits:
        x, y_original, eval_position = actual_split(
            dataset,
            args.bptt,
            args.eval_position,
            split_number,
        )
        y = encoded_labels(y_original)
        train_x = x[:eval_position].float()
        test_x = x[eval_position:].float()
        train_y = y[:eval_position]
        test_y = y[eval_position:]

        feature_scale = torch.std(x.float(), dim=0, unbiased=False)
        valid_scale = torch.isfinite(feature_scale) & (feature_scale > 0)
        standardized_shift = torch.zeros_like(feature_scale)
        standardized_shift[valid_scale] = (
            torch.mean(train_x[:, valid_scale], dim=0)
            - torch.mean(test_x[:, valid_scale], dim=0)
        ).abs() / feature_scale[valid_scale]
        finite_shift = standardized_shift[torch.isfinite(standardized_shift)]

        train_tuples = {tuple(row) for row in train_x.numpy().tolist()}
        cross_split_matches = sum(
            tuple(row) in train_tuples for row in test_x.numpy().tolist()
        )
        rows.append(
            {
                "did": DID,
                "split": split_number,
                "ordered_original_split": split_number == 1,
                "train_size": len(train_y),
                "test_size": len(test_y),
                "train_positive_fraction": float(train_y.float().mean()),
                "test_positive_fraction": float(test_y.float().mean()),
                "positive_fraction_gap": float(
                    (train_y.float().mean() - test_y.float().mean()).abs()
                ),
                "feature_shift_rms": float(
                    torch.sqrt(torch.mean(finite_shift.square()))
                ),
                "feature_shift_max": float(torch.max(finite_shift)),
                "exact_test_rows_present_in_train": int(cross_split_matches),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(eval_helper, model, prediction_method, args):
    result = eval_helper.do_evaluation_custom(
        model,
        bptt=args.bptt,
        eval_positions=[args.eval_position],
        metric=tabular_metrics.auc_metric,
        device=args.device,
        method_name=prediction_method,
        evaluation_type=DID,
        max_classes=10,
        max_features=100,
        max_time=args.max_time,
        split_numbers=args.splits,
        jrt_prompt=False,
        single_evaluation_prompt=False,
        permutation_bagging=args.permutation_bagging,
        sample_bagging=args.sample_bagging,
        eval_filters=EVAL_FILTERS,
        return_whole_output=True,
    )
    return result[DID]


def tensor_for_key(split_result, marker):
    keys = [key for key in split_result if marker in key]
    if len(keys) != 1:
        raise RuntimeError(f"Expected one result key containing {marker!r}, got {keys}")
    return torch.as_tensor(split_result[keys[0]]).detach().cpu()


def normalized_outputs_and_targets(split_result):
    probabilities = tensor_for_key(split_result, "_outputs_at_").float()
    targets = tensor_for_key(split_result, "_ys_at_").long()
    probabilities = probabilities.squeeze()
    targets = targets.squeeze()
    if probabilities.ndim != 2 or targets.ndim != 1:
        raise RuntimeError(
            "Unexpected result shapes after removing singleton batch dimensions: "
            f"outputs={tuple(probabilities.shape)}, targets={tuple(targets.shape)}"
        )
    if len(probabilities) != len(targets):
        raise RuntimeError(
            f"Prediction/target length mismatch: {len(probabilities)} != {len(targets)}"
        )
    probabilities = probabilities.clamp_min(1e-12)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    return probabilities.numpy(), targets.numpy()


def inference_seconds(split_result):
    values = [
        float(value)
        for key, value in split_result.items()
        if "_time_at_" in key
    ]
    return sum(values)


def expected_calibration_error(targets, probabilities, bins=10):
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def metric_row(model_name, split_number, split_result, parameter_count):
    probabilities, targets = normalized_outputs_and_targets(split_result)
    predicted = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    matrix = confusion_matrix(targets, predicted, labels=list(range(n_classes)))
    result = {
        "did": DID,
        "model": model_name,
        "parameters": parameter_count,
        "split": split_number,
        "n_test": len(targets),
        "auc": roc_auc_score(
            targets,
            probabilities[:, 1] if n_classes == 2 else probabilities,
            multi_class="ovo",
        ),
        "accuracy": accuracy_score(targets, predicted),
        "balanced_accuracy": balanced_accuracy_score(targets, predicted),
        "average_precision": (
            average_precision_score(targets, probabilities[:, 1])
            if n_classes == 2
            else np.nan
        ),
        "log_loss": log_loss(
            targets,
            probabilities,
            labels=list(range(n_classes)),
        ),
        "brier": (
            float(np.mean((probabilities[:, 1] - targets) ** 2))
            if n_classes == 2
            else float(
                np.mean(
                    np.sum(
                        (
                            probabilities
                            - np.eye(n_classes, dtype=float)[targets]
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        ),
        "ece_10_bin": expected_calibration_error(targets, probabilities),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "mean_predictive_entropy": float(
            np.mean(-np.sum(probabilities * np.log(probabilities), axis=1))
        ),
        "predicted_positive_fraction": (
            float((predicted == 1).mean()) if n_classes == 2 else np.nan
        ),
        "inference_seconds": inference_seconds(split_result),
    }
    if n_classes == 2:
        tn, fp, fn, tp = matrix.ravel()
        result.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result


def prediction_rows(model_name, split_number, split_result):
    probabilities, targets = normalized_outputs_and_targets(split_result)
    predicted = probabilities.argmax(axis=1)
    rows = []
    for test_index, (target, prediction, probability) in enumerate(
        zip(targets, predicted, probabilities)
    ):
        row = {
            "did": DID,
            "model": model_name,
            "split": split_number,
            "test_index": test_index,
            "target": int(target),
            "predicted_class": int(prediction),
            "confidence": float(probability.max()),
            "correct": bool(prediction == target),
        }
        for class_index, value in enumerate(probability):
            row[f"probability_class_{class_index}"] = float(value)
        rows.append(row)
    return rows


def pairwise_rows(prediction_frame, split_numbers):
    rows = []
    for split_number in split_numbers:
        split_frame = prediction_frame[prediction_frame["split"] == split_number]
        models = sorted(split_frame["model"].unique())
        for model_a, model_b in itertools.combinations(models, 2):
            a = split_frame[split_frame["model"] == model_a].sort_values("test_index")
            b = split_frame[split_frame["model"] == model_b].sort_values("test_index")
            if not np.array_equal(a["target"].to_numpy(), b["target"].to_numpy()):
                raise RuntimeError(
                    f"Targets differ for {model_a} and {model_b}, split {split_number}"
                )
            pred_a = a["predicted_class"].to_numpy()
            pred_b = b["predicted_class"].to_numpy()
            correct_a = a["correct"].to_numpy(dtype=bool)
            correct_b = b["correct"].to_numpy(dtype=bool)
            p1_a = a["probability_class_1"].to_numpy()
            p1_b = b["probability_class_1"].to_numpy()
            rows.append(
                {
                    "did": DID,
                    "split": split_number,
                    "model_a": model_a,
                    "model_b": model_b,
                    "prediction_disagreement_fraction": float(
                        np.mean(pred_a != pred_b)
                    ),
                    "both_correct_fraction": float(np.mean(correct_a & correct_b)),
                    "only_model_a_correct_fraction": float(
                        np.mean(correct_a & ~correct_b)
                    ),
                    "only_model_b_correct_fraction": float(
                        np.mean(~correct_a & correct_b)
                    ),
                    "both_wrong_fraction": float(np.mean(~correct_a & ~correct_b)),
                    "positive_probability_correlation": float(
                        np.corrcoef(p1_a, p1_b)[0, 1]
                    ),
                    "mean_absolute_positive_probability_gap": float(
                        np.mean(np.abs(p1_a - p1_b))
                    ),
                }
            )
    return rows


def summary_frame(metric_frame):
    metric_columns = [
        "auc",
        "accuracy",
        "balanced_accuracy",
        "average_precision",
        "log_loss",
        "brier",
        "ece_10_bin",
        "mean_confidence",
        "mean_predictive_entropy",
        "predicted_positive_fraction",
        "inference_seconds",
    ]
    summary = metric_frame.groupby(["model", "parameters"])[metric_columns].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    return summary.reset_index()


def write_csv(frame, output_dir, filename):
    path = os.path.join(output_dir, filename)
    frame.to_csv(path, index=False)
    print(f"Saved {path}", flush=True)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)
    specs = resolved_specs(args)
    os.makedirs(args.output_dir, exist_ok=True)

    eval_helper = EvalHelper()
    dataset = prepare_dataset(eval_helper)
    profile_frame = dataset_profile(dataset)
    split_profile_frame = split_profile_rows(dataset, args)

    metric_rows = []
    predictions = []
    for model_name, spec in specs.items():
        print(f"\nEvaluating {model_name} on OpenML DID {DID}...", flush=True)
        model, _, parameter_count = load_model(model_name, spec, args.device)
        split_results = evaluate_model(
            eval_helper,
            model,
            spec["prediction_method"],
            args,
        )
        if len(split_results) != len(args.splits):
            raise RuntimeError(
                f"{model_name} returned {len(split_results)} results for "
                f"{len(args.splits)} requested splits"
            )
        for split_number, split_result in zip(args.splits, split_results):
            metric_rows.append(
                metric_row(
                    model_name,
                    split_number,
                    split_result,
                    parameter_count,
                )
            )
            predictions.extend(
                prediction_rows(model_name, split_number, split_result)
            )

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    metric_frame = pd.DataFrame(metric_rows)
    prediction_frame = pd.DataFrame(predictions)
    pairwise_frame = pd.DataFrame(pairwise_rows(prediction_frame, args.splits))
    summary = summary_frame(metric_frame)

    write_csv(profile_frame, args.output_dir, "dataset_profile.csv")
    write_csv(split_profile_frame, args.output_dir, "split_profile.csv")
    write_csv(metric_frame, args.output_dir, "model_metrics_by_split.csv")
    write_csv(summary, args.output_dir, "model_summary.csv")
    write_csv(prediction_frame, args.output_dir, "predictions.csv")
    write_csv(pairwise_frame, args.output_dir, "pairwise_disagreement.csv")

    display_columns = [
        "model",
        "parameters",
        "auc_mean",
        "auc_std",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "log_loss_mean",
        "brier_mean",
        "ece_10_bin_mean",
    ]
    print("\nDataset profile:", flush=True)
    print(profile_frame.to_string(index=False), flush=True)
    print("\nModel summary (mean over splits):", flush=True)
    print(
        summary.sort_values("auc_mean", ascending=False)[display_columns].to_string(
            index=False
        ),
        flush=True,
    )
    print(
        "\nInspect split_profile.csv for ordered-split/class/feature shift, "
        "model_metrics_by_split.csv for variance and calibration, and "
        "pairwise_disagreement.csv plus predictions.csv for model-specific errors.",
        flush=True,
    )


if __name__ == "__main__":
    main()
