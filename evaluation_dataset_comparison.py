import argparse
import os
import sys

import torch

import pandas as pd

NANOTABPFN_IMPORT_PATH = os.path.join(os.path.dirname(__file__), "nanoTabPFN")
if NANOTABPFN_IMPORT_PATH not in sys.path:
    sys.path.insert(0, NANOTABPFN_IMPORT_PATH)

from model import NanoTabPFNModel

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


DEFAULT_MODELS = [
    "hybrid_8l",
    "tabpfn",
    "hydra",
]
DEFAULT_HYBRID_MODEL = "tabpfn/models_diff/callback_hybrid_6hydra_6transformer_epoch_200.cpkt"
DEFAULT_HYBRID_8_LAYERS_MODEL = "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt"
DEFAULT_LOOPED_TRANSFORMER_DEPTH3_MODEL = (
    "tabpfn/models_diff/callback_looped_transformer_6physical_core4x3_latest.cpkt"
)
DEFAULT_NEW_LOOPED_12_LAYERS_MODEL = (
    "tabpfn/models_diff/looped_transformer_6physical_core4_mixed_depth_2_3_3_2_12l.cpkt"
)
DEFAULT_TABPFN_MODEL = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
DEFAULT_HYDRA_MODEL = "tabpfn/models_diff/hydra_small.cpkt"
DEFAULT_RECENT_HYDRA_25M_MODEL = "tabpfn/models_diff/callback_pure_hydra_12_layers_512e_latest.cpkt"
DEFAULT_NANOTABPFN_MODEL = "nanoTabPFN/nanotabpfn_trained.pt"

MODEL_SPECS = {
    "original_hydra": {
        "loader": "hydra_workflow",
        "path_arg": "hydra_model",
        "prediction_method": "hydra",
    },
    "recent_hydra_25m": {
        "loader": "custom",
        "path_arg": "recent_hydra_25m_model",
        "model_type": "hydra",
        "prediction_method": "hydra",
    },
    "original_tabpfn": {
        "loader": "transformer_workflow",
        "path_arg": "tabpfn_model",
        "prediction_method": "transformer",
    },
    "hybrid_8_layers": {
        "loader": "custom",
        "path_arg": "hybrid_8_layers_model",
        "model_type": "hybrid",
        "prediction_method": "transformer",
    },
    "hybrid_8l": {
        "loader": "custom",
        "path_arg": "hybrid_8_layers_model",
        "model_type": "hybrid",
        "prediction_method": "transformer",
    },
    "new_looped_12_layers": {
        "loader": "custom",
        "path_arg": "new_looped_12_layers_model",
        "model_type": "looped_transformer",
        "prediction_method": "transformer",
    },
    "nanotabpfn": {
        "loader": "nanotabpfn",
        "path_arg": "nanotabpfn_model",
        "prediction_method": "transformer",
    },
    "hydra": {
        "loader": "custom",
        "path_arg": "recent_hydra_25m_model",
        "model_type": "hydra",
        "prediction_method": "hydra",
    },
    "hybrid": {
        "loader": "custom",
        "path_arg": "hybrid_model",
        "model_type": "hybrid",
        "prediction_method": "transformer",
    },
    "looped_transformer_depth3": {
        "loader": "custom",
        "path_arg": "looped_transformer_depth3_model",
        "model_type": "looped_transformer",
        "prediction_method": "transformer",
    },
    "tabpfn": {
        "loader": "transformer_workflow",
        "path_arg": "tabpfn_model",
        "prediction_method": "transformer",
    },
}

METRICS = {
    "auc": tabular_metrics.auc_metric,
    "accuracy": tabular_metrics.accuracy_metric,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hydra, hybrid, and TabPFN model quality on specific OpenML datasets "
            "and report which model wins per dataset."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs. Defaults to the small OpenML-CC18 subset used in EvalHelper.",
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=DEFAULT_MODELS)
    parser.add_argument("--metric", choices=list(METRICS), default="auc")
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--hybrid-model", default=DEFAULT_HYBRID_MODEL)
    parser.add_argument("--hybrid-8-layers-model", default=DEFAULT_HYBRID_8_LAYERS_MODEL)
    parser.add_argument(
        "--looped-transformer-depth3-model",
        default=DEFAULT_LOOPED_TRANSFORMER_DEPTH3_MODEL,
    )
    parser.add_argument(
        "--new-looped-12-layers-model",
        default=DEFAULT_NEW_LOOPED_12_LAYERS_MODEL,
    )
    parser.add_argument("--tabpfn-model", default=DEFAULT_TABPFN_MODEL)
    parser.add_argument("--hydra-model", default=DEFAULT_HYDRA_MODEL)
    parser.add_argument("--recent-hydra-25m-model", default=DEFAULT_RECENT_HYDRA_25M_MODEL)
    parser.add_argument("--nanotabpfn-model", default=DEFAULT_NANOTABPFN_MODEL)
    parser.add_argument("--permutation-bagging", type=int, default=1)
    parser.add_argument("--sample-bagging", type=int, default=0)
    parser.add_argument("--jrt-prompt", action="store_true")
    parser.add_argument("--single-eval-prompt", action="store_true")
    parser.add_argument("--exclude-categorical", action="store_true")
    parser.add_argument("--keep-nans", action="store_true")
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument(
        "--result-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison.csv"),
    )
    parser.add_argument(
        "--split-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_splits.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_summary.csv"),
    )
    parser.add_argument(
        "--dataset-table-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_table.csv"),
    )
    parser.add_argument(
        "--model-stats-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_model_stats.csv"),
    )
    parser.add_argument(
        "--hydra-wins-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_hydra_wins.csv"),
    )
    parser.add_argument(
        "--hybrid-wins-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_hybrid_wins.csv"),
    )
    parser.add_argument(
        "--non-tabpfn-wins-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_non_tabpfn_wins.csv"),
    )
    return parser.parse_args()



class NanoTabPFNEvaluationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = None

    def forward(self, src, single_eval_pos):
        _, eval_xs, eval_ys = src
        x = eval_xs.transpose(0, 1).contiguous()
        y = eval_ys[:single_eval_pos].squeeze(-1).transpose(0, 1).contiguous()
        output = self.model((x, y), train_test_split_index=single_eval_pos)
        return output.transpose(0, 1).contiguous()


def load_nanotabpfn_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = NanoTabPFNModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = checkpoint.get("evaluation_config", {})
    return NanoTabPFNEvaluationWrapper(model), config


def print_parameter_count(model_name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} parameters: {total_params:,} total ({trainable_params:,} trainable)")


def load_model(model_name, args):
    spec = MODEL_SPECS[model_name]
    model_path = getattr(args, spec["path_arg"])

    if spec["loader"] == "nanotabpfn":
        model, config = load_nanotabpfn_model(model_path)
        model.eval()
        print_parameter_count(model_name, model)
        return model, config

    if spec["loader"] == "custom":
        loaded, config = load_model_only_inference(
            ".",
            model_path,
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
            model_path_custom=model_path,
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
            model_path_custom=model_path,
        )
    else:
        raise ValueError(f"Unknown loader for model: {model_name}")

    model = loaded[2]
    model.eval()
    print_parameter_count(model_name, model)
    return model, config


def prediction_method(model_name):
    return MODEL_SPECS[model_name]["prediction_method"]


def eval_filters(args):
    return {
        "categorical": not args.exclude_categorical,
        "nans": not args.keep_nans,
        "multiclass": not args.binary_only,
    }


def as_float(value):
    if hasattr(value, "item"):
        return value.item()
    return float(value)


def evaluate_dataset(eval_helper, did, model_name, model, config, args):
    eval_helper.limit_dict = {}
    result = eval_helper.do_evaluation_custom(
        model,
        bptt=config.get("bptt", args.bptt),
        eval_positions=config.get("eval_positions", [config.get("bptt", args.bptt) // 2]),
        metric=METRICS[args.metric],
        device=args.device,
        method_name=prediction_method(model_name),
        evaluation_type=did,
        max_classes=config.get("max_num_classes", args.max_classes),
        max_features=config.get("max_num_features", args.max_features),
        max_time=args.max_time,
        split_numbers=args.splits,
        jrt_prompt=args.jrt_prompt,
        single_evaluation_prompt=args.single_eval_prompt,
        permutation_bagging=args.permutation_bagging,
        sample_bagging=args.sample_bagging,
        eval_filters=eval_filters(args),
    )

    scores = [as_float(score) for score in result[did]]
    return scores


def best_model_for_scores(scores_by_model):
    ranked = sorted(scores_by_model.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    margin = best_score - second_score if second_score is not None else None
    return best_name, best_score, margin


def score_stats(scores):
    series = pd.Series(scores)
    return {
        "mean": series.mean(),
        "std": series.std(ddof=1) if len(series) > 1 else 0.0,
        "min": series.min(),
        "max": series.max(),
        "median": series.median(),
    }


def write_csv(df, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def make_dataset_table(dataset_df, model_names):
    columns = ["did", "best_model", "best_score", "margin_to_second"]
    for model_name in model_names:
        columns.extend([f"{model_name}_mean", f"{model_name}_std"])
    return dataset_df[columns]


def make_model_stats_table(dataset_df, model_names):
    rows = []
    for model_name in model_names:
        per_dataset_means = dataset_df[f"{model_name}_mean"]
        rows.append(
            {
                "model": model_name,
                "mean_across_datasets": per_dataset_means.mean(),
                "std_across_datasets": (
                    per_dataset_means.std(ddof=1) if len(per_dataset_means) > 1 else 0.0
                ),
                "num_datasets": len(per_dataset_means),
                "num_dataset_wins": int((dataset_df["best_model"] == model_name).sum()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    eval_helper = EvalHelper()
    dids = args.dids or eval_helper.openml_cc18_dids_small
    models = {model_name: load_model(model_name, args) for model_name in args.models}

    dataset_rows = []
    split_rows = []

    for did in dids:
        print(f"\nEvaluating OpenML dataset {did}", flush=True)
        split_scores_by_model = {}

        for model_name, (model, config) in models.items():
            scores = evaluate_dataset(eval_helper, did, model_name, model, config, args)
            split_scores_by_model[model_name] = scores
            print(f"{model_name}: split scores={scores}, mean={sum(scores) / len(scores)}", flush=True)

        mean_scores = {
            model_name: sum(scores) / len(scores)
            for model_name, scores in split_scores_by_model.items()
        }
        best_name, best_score, margin = best_model_for_scores(mean_scores)

        dataset_row = {
            "did": did,
            "best_model": best_name,
            "best_score": best_score,
            "margin_to_second": margin,
        }
        for model_name, scores in split_scores_by_model.items():
            stats = score_stats(scores)
            for stat_name, stat_value in stats.items():
                dataset_row[f"{model_name}_{stat_name}"] = stat_value
            dataset_row[model_name] = stats["mean"]
        dataset_rows.append(dataset_row)

        for split_idx, split_number in enumerate(args.splits):
            split_score_row = {
                "did": did,
                "split": split_number,
            }
            split_score_row.update(
                {
                    model_name: scores[split_idx]
                    for model_name, scores in split_scores_by_model.items()
                }
            )
            split_best_name, split_best_score, split_margin = best_model_for_scores(
                {
                    model_name: scores[split_idx]
                    for model_name, scores in split_scores_by_model.items()
                }
            )
            split_score_row["best_model"] = split_best_name
            split_score_row["best_score"] = split_best_score
            split_score_row["margin_to_second"] = split_margin
            split_rows.append(split_score_row)

    dataset_df = pd.DataFrame(dataset_rows)
    split_df = pd.DataFrame(split_rows)
    dataset_table_df = make_dataset_table(dataset_df, list(models))
    model_stats_df = make_model_stats_table(dataset_df, list(models))
    summary_df = (
        dataset_df["best_model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="num_dataset_wins")
    )
    hydra_wins_df = dataset_df[
        dataset_df["best_model"].str.contains("hydra", case=False, na=False)
    ].copy()
    hybrid_wins_df = dataset_df[
        dataset_df["best_model"].str.contains("hybrid", case=False, na=False)
    ].copy()
    non_tabpfn_wins_df = dataset_df[
        ~dataset_df["best_model"].str.contains("tabpfn", case=False, na=False)
    ].copy()

    write_csv(dataset_df, args.result_csv)
    write_csv(split_df, args.split_csv)
    write_csv(dataset_table_df, args.dataset_table_csv)
    write_csv(model_stats_df, args.model_stats_csv)
    write_csv(summary_df, args.summary_csv)
    write_csv(hydra_wins_df, args.hydra_wins_csv)
    write_csv(hybrid_wins_df, args.hybrid_wins_csv)
    write_csv(non_tabpfn_wins_df, args.non_tabpfn_wins_csv)

    print("\nDataset winner summary:")
    print(summary_df.to_string(index=False))
    print("\nAll datasets by model (mean +/- std over splits):")
    print(dataset_table_df.to_string(index=False))
    print("\nModel score summary across dataset means:")
    print(model_stats_df.to_string(index=False))
    print(f"\nHydra-prevalent datasets: {hydra_wins_df['did'].tolist()}")
    print(f"Hybrid-prevalent datasets: {hybrid_wins_df['did'].tolist()}")
    print(f"Non-TabPFN prevalent datasets: {non_tabpfn_wins_df['did'].tolist()}")
    print(f"\nSaved per-dataset comparison to {args.result_csv}")
    print(f"Saved per-split comparison to {args.split_csv}")
    print(f"Saved all-datasets table to {args.dataset_table_csv}")
    print(f"Saved model score summary to {args.model_stats_csv}")
    print(f"Saved winner summary to {args.summary_csv}")
    print(f"Saved Hydra wins to {args.hydra_wins_csv}")
    print(f"Saved hybrid wins to {args.hybrid_wins_csv}")
    print(f"Saved non-TabPFN wins to {args.non_tabpfn_wins_csv}")


if __name__ == "__main__":
    main()
