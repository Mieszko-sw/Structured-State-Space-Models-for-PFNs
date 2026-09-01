"""Compare elastic and standard 12x looped models at every inference depth.

Both checkpoints contain one physical Transformer block and have a maximum
effective depth of 12 recurrent passes.  The elastic checkpoint was trained to
support shortcut depths, while the standard checkpoint was trained at depth 12.
For a fair compute comparison, both models are evaluated with the same
``num_loops`` value at each depth.

Scores are averaged across splits within each dataset.  The depth-level score
is then a macro-average across datasets, so every dataset receives equal weight.
Two CSV files are produced:

* one row per dataset and depth, including every split score for both models;
* one row per depth, including both macro averages, their difference, and
  per-dataset win counts.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


MAX_LOOPS = 12
ELASTIC_MODEL = "elastic_looped_12x"
LOOPED_MODEL = "looped_12x"
ELASTIC_PATH = (
    "tabpfn/models_diff/"
    "callback_elastic_looped_transformer_1physical_core12x_latest.cpkt"
)
LOOPED_PATH = (
    "tabpfn/models_diff/"
    "callback_new_looped_transformer_1physical_core12x_latest.cpkt"
)

MODELS = {
    ELASTIC_MODEL: {
        "path": ELASTIC_PATH,
        "loader_type": "looped_transformer",
        "method_name": "transformer",
    },
    LOOPED_MODEL: {
        "path": LOOPED_PATH,
        "loader_type": "looped_transformer",
        "method_name": "transformer",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        choices=range(1, MAX_LOOPS + 1),
        default=list(range(1, MAX_LOOPS + 1)),
        metavar="N",
        help="Inference depths to evaluate (default: 1 through 12).",
    )
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs (default: the project's 30-dataset list).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        choices=benchmark.SPLIT_NUMBERS,
        default=benchmark.SPLIT_NUMBERS,
        help="Evaluation splits (default: 1 2 3 4 5).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_12x_depths_per_dataset.csv",
        ),
    )
    parser.add_argument(
        "--average-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_vs_looped_12x_depths_average.csv",
        ),
    )
    return parser.parse_args()


def evaluate_one(model, config, model_name, dataset_list, device, split, depth):
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(
            f"{model_name}: expected one evaluation position, got {eval_positions}"
        )

    with torch.no_grad():
        result = benchmark.evaluate(
            datasets=dataset_list,
            bptt=benchmark.BPTT,
            eval_positions=eval_positions,
            metric_used=benchmark.METRIC_USED,
            model=model,
            device=device,
            method_name=MODELS[model_name]["method_name"],
            max_time=300,
            split_number=split,
            jrt_prompt=False,
            random_premutation=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            num_loops=depth,
        )
    return benchmark.extract_metric(result)


def score_summary(scores):
    return {
        "mean": float(np.mean(scores)),
        "std": (
            float(np.std(scores, ddof=1))
            if len(scores) > 1
            else np.nan
        ),
    }


def evaluate_dataset_depth(models, did, dataset_list, depth, args):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    scores = {model_name: [] for model_name in MODELS}

    for split in args.splits:
        for model_name in (ELASTIC_MODEL, LOOPED_MODEL):
            model, config = models[model_name]
            score = evaluate_one(
                model,
                config,
                model_name,
                dataset_list,
                args.device,
                split,
                depth,
            )
            scores[model_name].append(score)
            print(
                f"model={model_name} depth={depth} did={did} "
                f"split={split} auc_roc={score:.6f}",
                flush=True,
            )

    elastic = score_summary(scores[ELASTIC_MODEL])
    looped = score_summary(scores[LOOPED_MODEL])
    row = {
        "depth": depth,
        "effective_transformer_layers": depth,
        "did": did,
        "dataset_name": dataset_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "split_count": len(args.splits),
        "elastic_mean_auc_roc": elastic["mean"],
        "elastic_std_auc_roc": elastic["std"],
        "looped_mean_auc_roc": looped["mean"],
        "looped_std_auc_roc": looped["std"],
        "elastic_minus_looped_auc_roc": elastic["mean"] - looped["mean"],
    }
    for split, elastic_score, looped_score in zip(
        args.splits,
        scores[ELASTIC_MODEL],
        scores[LOOPED_MODEL],
    ):
        row[f"split_{split}_elastic_auc_roc"] = elastic_score
        row[f"split_{split}_looped_auc_roc"] = looped_score
        row[f"split_{split}_elastic_minus_looped_auc_roc"] = (
            elastic_score - looped_score
        )
    return row


def aggregate_by_depth(per_dataset):
    rows = []
    for depth, depth_df in per_dataset.groupby("depth", sort=True):
        delta = depth_df["elastic_minus_looped_auc_roc"]
        rows.append(
            {
                "depth": int(depth),
                "effective_transformer_layers": int(depth),
                "dataset_count": int(depth_df["did"].nunique()),
                "splits_per_dataset": int(depth_df["split_count"].iloc[0]),
                "elastic_average_auc_roc": float(
                    depth_df["elastic_mean_auc_roc"].mean()
                ),
                "elastic_std_auc_roc_across_datasets": float(
                    depth_df["elastic_mean_auc_roc"].std(ddof=1)
                ),
                "looped_average_auc_roc": float(
                    depth_df["looped_mean_auc_roc"].mean()
                ),
                "looped_std_auc_roc_across_datasets": float(
                    depth_df["looped_mean_auc_roc"].std(ddof=1)
                ),
                "elastic_minus_looped_average_auc_roc": float(delta.mean()),
                "elastic_dataset_wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "looped_dataset_wins": int((delta < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def ensure_parent_directory(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    args.depths = list(dict.fromkeys(args.depths))
    args.splits = list(dict.fromkeys(args.splits))
    args.device = benchmark.normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = EvalHelper()
    dids = (
        args.dids
        if args.dids is not None
        else eval_helper.openml_cc18_dids_small
    )
    datasets = benchmark.prepare_datasets(dids)
    if len(datasets) != len(dids):
        raise RuntimeError(
            f"Requested {len(dids)} datasets but only {len(datasets)} "
            "passed preparation."
        )

    # benchmark.load_model reads this registry from its own module.
    benchmark.MODELS = MODELS
    models = {
        model_name: benchmark.load_model(model_name, args.device)
        for model_name in (ELASTIC_MODEL, LOOPED_MODEL)
    }

    rows = []
    for depth in args.depths:
        for did, dataset_list in datasets.items():
            rows.append(
                evaluate_dataset_depth(
                    models,
                    did,
                    dataset_list,
                    depth,
                    args,
                )
            )

    per_dataset = pd.DataFrame(rows).sort_values(["depth", "did"])
    average = aggregate_by_depth(per_dataset)

    ensure_parent_directory(args.per_dataset_csv)
    ensure_parent_directory(args.average_csv)
    per_dataset.to_csv(args.per_dataset_csv, index=False)
    average.to_csv(args.average_csv, index=False)

    print("\nMacro-average AUC-ROC across datasets:")
    print(average.to_string(index=False))
    print(f"\nSaved per-dataset results to {args.per_dataset_csv}")
    print(f"Saved depth averages to {args.average_csv}")


if __name__ == "__main__":
    main()
