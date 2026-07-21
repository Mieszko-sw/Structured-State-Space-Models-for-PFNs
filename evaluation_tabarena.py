import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    get_params_from_config as hydra_get_params_from_config,
)
from tabpfn.scripts.hydra_prediction_interface import hydra_predict
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.transformer_prediction_interface import (
    get_params_from_config as transformer_get_params_from_config,
)
from tabpfn.scripts.transformer_prediction_interface import transformer_predict
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


TABARENA_V01_TASK_IDS = [
    363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
    363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
    363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
    363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
    363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
    363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
    363708, 363711, 363712,
]

MODEL_SPECS = {
    "hybrid_8l": {
        "loader": "custom",
        "path": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
        "model_type": "hybrid",
        "prediction_method": "transformer",
    },
    "tabpfn": {
        "loader": "transformer_workflow",
        "path": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
        "prediction_method": "transformer",
    },
    "hydra": {
        "loader": "custom",
        "path": "tabpfn/models_diff/callback_pure_hydra_12_layers_512e_latest.cpkt",
        "model_type": "hydra",
        "prediction_method": "hydra",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model from evaluation_script.py on TabArena OpenML tasks."
    )
    parser.add_argument("--model", choices=list(MODEL_SPECS), default=None, help="Evaluate a single model instead of all default models.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=None, help="Evaluate selected models. Defaults to all models in MODEL_SPECS.")
    parser.add_argument("--model-path", default=None, help="Override the checkpoint path for --model.")
    parser.add_argument("--model-type", default=None, help="Override custom loader type, e.g. hybrid, hydra, transformer.")
    parser.add_argument("--prediction-method", choices=["transformer", "hydra"], default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-ids", nargs="+", type=int, default=None)
    parser.add_argument("--task-csv", default=None, help="Optional CSV with a task_id column.")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--n-ensembles", type=int, default=1)
    parser.add_argument("--batch-size-inference", type=int, default=16)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Untimed prediction runs before each timed fold (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--multiclass-auc", choices=["ovr", "ovo"], default="ovr")
    parser.add_argument("--skip-missing-values", action="store_true")
    parser.add_argument("--fold-csv", default=os.path.join("result_csvs", "tabarena_eval_folds.csv"))
    parser.add_argument("--summary-csv", default=os.path.join("result_csvs", "tabarena_eval_summary.csv"))
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


def task_ids_from_args(args):
    task_ids = list(args.task_ids) if args.task_ids else list(TABARENA_V01_TASK_IDS)
    if args.task_csv:
        task_df = pd.read_csv(args.task_csv)
        if "task_id" not in task_df.columns:
            raise ValueError("--task-csv must contain a task_id column")
        task_ids = task_df["task_id"].astype(int).tolist()
    if args.limit_tasks:
        task_ids = task_ids[: args.limit_tasks]
    return task_ids


def selected_model_names(args):
    if args.model and args.models:
        raise ValueError("Use either --model or --models, not both")
    if args.model:
        return [args.model]
    if args.models:
        return args.models
    return list(MODEL_SPECS)


def load_model(args, model_name):
    spec = dict(MODEL_SPECS[model_name])
    if args.model_path:
        spec["path"] = args.model_path
    if args.model_type:
        spec["model_type"] = args.model_type
    if args.prediction_method:
        spec["prediction_method"] = args.prediction_method

    if spec["loader"] == "custom":
        loaded, config = load_model_only_inference(
            ".",
            spec["path"],
            args.device,
            model_name=spec["model_type"],
        )
    elif spec["loader"] == "transformer_workflow":
        loaded, config, _ = transformer_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=args.device,
            eval_addition="",
            only_inference=True,
            model_path_custom=spec["path"],
        )
    elif spec["loader"] == "hydra_workflow":
        loaded, config, _ = hydra_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=args.device,
            eval_addition="",
            only_inference=True,
            model_path_custom=spec["path"],
        )
    else:
        raise ValueError(f"Unknown loader: {spec['loader']}")

    model = loaded[2]
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} from {spec['path']}")
    print(f"Parameters: {total_params:,}")
    return model, config, spec["prediction_method"], spec["path"]


def is_classification_task(task):
    try:
        from openml.tasks import TaskType

        return task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION
    except Exception:
        task_type = str(getattr(task, "task_type", "")).lower()
        return "classification" in task_type


def encode_features(X_df, categorical_indicator, max_features):
    X_df = X_df.iloc[:, :max_features].copy()
    categorical_indicator = list(categorical_indicator or [])[: X_df.shape[1]]
    columns = []
    categorical_feats = []

    for idx, column in enumerate(X_df.columns):
        series = X_df[column]
        is_categorical = (
            idx < len(categorical_indicator) and categorical_indicator[idx]
        ) or not pd.api.types.is_numeric_dtype(series)
        if is_categorical:
            codes = pd.Categorical(series).codes.astype(np.float32)
            codes[codes < 0] = np.nan
            columns.append(codes)
            categorical_feats.append(idx)
        else:
            columns.append(pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32))

    if not columns:
        raise ValueError("Dataset has no usable feature columns after filtering")
    return np.column_stack(columns).astype(np.float32), categorical_feats


def load_task_data(task_id, max_features):
    import openml

    try:
        task = openml.tasks.get_task(task_id, download_splits=False)
    except TypeError:
        task = openml.tasks.get_task(task_id)
    if not is_classification_task(task):
        return None, f"task {task_id} is not supervised classification"

    try:
        dataset = task.get_dataset(download_data=False)
    except TypeError:
        dataset = task.get_dataset()
    X_df, y_raw, categorical_indicator, _ = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    valid_target_mask = pd.Series(y_raw).notna().to_numpy()
    X_df = X_df.loc[valid_target_mask].reset_index(drop=True)
    y_raw = pd.Series(y_raw).loc[valid_target_mask].reset_index(drop=True)

    X, categorical_feats = encode_features(X_df, categorical_indicator, max_features=max_features)
    y = LabelEncoder().fit_transform(y_raw.to_numpy())
    return {
        "task": task,
        "dataset_name": dataset.name,
        "X": X,
        "y": y.astype(np.int64),
        "categorical_feats": categorical_feats,
    }, None


def openml_split(task, fold):
    try:
        return task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
    except Exception:
        return None


def stratified_limit(indices, y, max_samples, seed):
    indices = np.asarray(indices)
    if max_samples is None or len(indices) <= max_samples:
        return indices
    try:
        limited, _ = train_test_split(
            indices,
            train_size=max_samples,
            stratify=y[indices],
            random_state=seed,
        )
        return np.asarray(limited)
    except ValueError:
        rng = np.random.default_rng(seed)
        return rng.choice(indices, size=max_samples, replace=False)


def iter_splits(task, X, y, folds, seed):
    used_openml = False
    for fold in range(folds):
        split = openml_split(task, fold)
        if split is None:
            break
        used_openml = True
        train_idx, test_idx = split
        yield fold, np.asarray(train_idx), np.asarray(test_idx), "openml"

    if used_openml:
        return

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        yield fold, train_idx, test_idx, "stratified_kfold"


def predict_proba(model, config, prediction_method, X_train, y_train, X_test, categorical_feats, args):
    X_full = np.concatenate([X_train, X_test], axis=0).astype(np.float32)
    y_full = np.concatenate([y_train, np.zeros(len(X_test), dtype=y_train.dtype)], axis=0)

    eval_xs = torch.tensor(X_full, device=args.device).float().unsqueeze(1)
    eval_ys = torch.tensor(y_full, device=args.device).float().unsqueeze(1)
    eval_position = len(X_train)

    if prediction_method == "hydra":
        predict_fn = hydra_predict
        params = hydra_get_params_from_config(config)
    else:
        predict_fn = transformer_predict
        params = transformer_get_params_from_config(config)

    synchronize(args.device)
    start = time.perf_counter()
    outputs, _internal_unsynchronized_seconds = predict_fn(
        model,
        eval_xs,
        eval_ys,
        eval_position,
        device=args.device,
        inference_mode=True,
        categorical_feats=categorical_feats,
        metric_used=tabular_metrics.auc_metric,
        N_ensemble_configurations=args.n_ensembles,
        batch_size_inference=args.batch_size_inference,
        seed=args.seed,
        **params,
    )
    synchronize(args.device)
    inference_seconds = time.perf_counter() - start
    proba = outputs.squeeze(0).detach().cpu().numpy()
    if proba.ndim != 2:
        raise ValueError(f"Expected 2D probability output, got shape {proba.shape}")
    return proba, inference_seconds


def score_predictions(y_true, proba, multiclass_auc):
    labels = np.arange(proba.shape[1])
    pred = np.argmax(proba, axis=1)
    result = {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, proba, labels=labels),
    }
    if proba.shape[1] == 2:
        result["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
    else:
        result["roc_auc"] = roc_auc_score(
            y_true,
            proba,
            labels=labels,
            multi_class=multiclass_auc,
        )
    return result


def write_csv(df, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_names = selected_model_names(args)
    if len(model_names) > 1 and (args.model_path or args.model_type or args.prediction_method):
        raise ValueError("--model-path, --model-type, and --prediction-method require a single --model")

    fold_rows = []
    task_ids = task_ids_from_args(args)

    for model_name in model_names:
        model, config, prediction_method, model_path = load_model(args, model_name)
        for task_id in task_ids:
            print(f"\nLoading TabArena/OpenML task {task_id}", flush=True)
            task_data, skip_reason = load_task_data(task_id, args.max_features)
            if skip_reason:
                print(f"Skipping task {task_id}: {skip_reason}", flush=True)
                fold_rows.append({"model": model_name, "model_path": model_path, "prediction_method": prediction_method, "task_id": task_id, "status": "skipped", "error": skip_reason})
                continue
    
            X = task_data["X"]
            y = task_data["y"]
            dataset_name = task_data["dataset_name"]
            categorical_feats = task_data["categorical_feats"]
            n_classes = len(np.unique(y))
    
            if n_classes > args.max_classes:
                error = f"{n_classes} classes exceeds --max-classes={args.max_classes}"
                print(f"Skipping {dataset_name}: {error}", flush=True)
                fold_rows.append({"model": model_name, "model_path": model_path, "prediction_method": prediction_method, "task_id": task_id, "dataset": dataset_name, "status": "skipped", "error": error})
                continue
            if args.skip_missing_values and np.isnan(X).any():
                error = "contains missing feature values"
                print(f"Skipping {dataset_name}: {error}", flush=True)
                fold_rows.append({"model": model_name, "model_path": model_path, "prediction_method": prediction_method, "task_id": task_id, "dataset": dataset_name, "status": "skipped", "error": error})
                continue
    
            for fold, train_idx, test_idx, split_source in iter_splits(
                task_data["task"], X, y, args.folds, args.seed
            ):
                train_idx = stratified_limit(train_idx, y, args.max_train_samples, args.seed + fold)
                test_idx = stratified_limit(test_idx, y, args.max_test_samples, args.seed + 1000 + fold)
                if len(np.unique(y[train_idx])) < n_classes:
                    error = "train split does not contain all dataset classes after subsampling"
                    status = "skipped"
                    scores = {"accuracy": np.nan, "log_loss": np.nan, "roc_auc": np.nan}
                    inference_seconds = np.nan
                    elapsed_seconds = np.nan
                else:
                    timed_start = None
                    try:
                        for _ in range(args.warmup_runs):
                            predict_proba(
                                model,
                                config,
                                prediction_method,
                                X[train_idx],
                                y[train_idx],
                                X[test_idx],
                                categorical_feats,
                                args,
                            )
                        timed_start = time.perf_counter()
                        proba, inference_seconds = predict_proba(
                            model,
                            config,
                            prediction_method,
                            X[train_idx],
                            y[train_idx],
                            X[test_idx],
                            categorical_feats,
                            args,
                        )
                        scores = score_predictions(y[test_idx], proba, args.multiclass_auc)
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        status = "failed"
                        error = str(exc).splitlines()[0]
                        scores = {"accuracy": np.nan, "log_loss": np.nan, "roc_auc": np.nan}
                        inference_seconds = np.nan
                    elapsed_seconds = (
                        time.perf_counter() - timed_start
                        if timed_start is not None
                        else np.nan
                    )
    
                row = {
                    "model": model_name,
                    "model_path": model_path,
                    "prediction_method": prediction_method,
                    "task_id": task_id,
                    "dataset": dataset_name,
                    "fold": fold,
                    "split_source": split_source,
                    "status": status,
                    "error": error,
                    "num_rows": len(y),
                    "num_features": X.shape[1],
                    "num_classes": n_classes,
                    "train_samples": len(train_idx),
                    "test_samples": len(test_idx),
                    "accuracy": scores["accuracy"],
                    "log_loss": scores["log_loss"],
                    "roc_auc": scores["roc_auc"],
                    "inference_seconds": inference_seconds,
                    "elapsed_seconds": elapsed_seconds if status != "skipped" else np.nan,
                }
                fold_rows.append(row)
                print(
                    f"{dataset_name} fold={fold} status={status} "
                    f"roc_auc={row['roc_auc']} accuracy={row['accuracy']} error={error}",
                    flush=True,
                )

    fold_df = pd.DataFrame(fold_rows)
    ok_df = fold_df[fold_df["status"] == "ok"].copy() if len(fold_df) else fold_df
    summary_df = (
        ok_df.groupby(["model", "task_id", "dataset"])[["roc_auc", "accuracy", "log_loss", "inference_seconds", "elapsed_seconds"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    ) if len(ok_df) else pd.DataFrame()

    write_csv(fold_df, args.fold_csv)
    write_csv(summary_df, args.summary_csv)
    print(f"\nSaved fold results to {args.fold_csv}")
    print(f"Saved task summary to {args.summary_csv}")
    if len(ok_df):
        print("\nOverall mean metrics over successful folds by model:")
        print(
            ok_df.groupby("model")
            [["roc_auc", "accuracy", "log_loss", "inference_seconds", "elapsed_seconds"]]
            .mean()
            .to_string()
        )


if __name__ == "__main__":
    main()
