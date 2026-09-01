"""Compare elastic 6x3 depths 1-6 with TabPFN on 30 OpenML datasets.

Each model/depth is evaluated on splits 1-5. Scores are first averaged across
splits for each dataset and then macro-averaged across datasets, giving every
dataset equal weight. The script writes both a compact model-level summary and
a per-dataset table containing the individual split scores.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


MAX_LOOPS = 6
ELASTIC_PATH = (
    "tabpfn/models_diff/"
    "elastic_looped_transformer_3physical_core6x_18l.cpkt"
)
TABPFN_PATH = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
TABPFN_NAME = "tabpfn"


def elastic_name(num_loops):
    return f"elastic_6x3_depth_{num_loops}"


MODELS = {
    **{
        elastic_name(num_loops): {
            "path": ELASTIC_PATH,
            "loader_type": "looped_transformer",
            "method_name": "transformer",
            "num_loops": num_loops,
        }
        for num_loops in range(1, MAX_LOOPS + 1)
    },
    TABPFN_NAME: {
        "path": TABPFN_PATH,
        "loader_type": "tabpfn",
        "method_name": "transformer",
        "num_loops": None,
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
        help="Elastic ABC-stack repetitions to evaluate (default: 1 2 3 4 5 6).",
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
        help="Splits averaged within each dataset (default: 1 2 3 4 5).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_6x3_depths_vs_tabpfn_openml_average.csv",
        ),
    )
    parser.add_argument(
        "--per-dataset-csv",
        default=os.path.join(
            "result_csvs",
            "elastic_6x3_depths_vs_tabpfn_openml_per_dataset.csv",
        ),
    )
    return parser.parse_args()


def evaluate_one(
    model,
    config,
    model_label,
    dataset_list,
    device,
    split_number,
):
    info = MODELS[model_label]
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(
            f"{model_label}: expected one evaluation position, got {eval_positions}"
        )

    optional_depth = (
        {} if info["num_loops"] is None else {"num_loops": info["num_loops"]}
    )
    with torch.no_grad():
        result = benchmark.evaluate(
            datasets=dataset_list,
            bptt=benchmark.BPTT,
            eval_positions=eval_positions,
            metric_used=benchmark.METRIC_USED,
            model=model,
            device=device,
            method_name=info["method_name"],
            max_time=300,
            split_number=split_number,
            jrt_prompt=False,
            random_premutation=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            **optional_depth,
        )
    return benchmark.extract_metric(result)


def evaluate_model(model, config, model_label, datasets, args):
    dataset_means = []
    per_dataset_rows = []
    for did, dataset_list in datasets.items():
        dataset_name, x, _, _, _, _ = dataset_list[0]
        split_scores = []
        for split_number in args.splits:
            score = evaluate_one(
                model,
                config,
                model_label,
                dataset_list,
                args.device,
                split_number,
            )
            split_scores.append(score)
            print(
                f"model={model_label} did={did} split={split_number} "
                f"auc_roc={score:.6f}",
                flush=True,
            )

        dataset_mean = float(np.mean(split_scores))
        dataset_means.append(dataset_mean)
        num_loops = MODELS[model_label]["num_loops"]
        row = {
            "model": "elastic_6x3" if num_loops is not None else TABPFN_NAME,
            "num_loops": num_loops if num_loops is not None else np.nan,
            "block_configuration": (
                f"({num_loops},{num_loops},{num_loops})"
                if num_loops is not None
                else "fixed"
            ),
            "did": did,
            "dataset_name": dataset_name,
            "num_samples": int(x.shape[0]),
            "num_features": int(x.shape[1]),
            "split_count": len(split_scores),
            "mean_auc_roc": dataset_mean,
            "std_auc_roc": (
                float(np.std(split_scores, ddof=1))
                if len(split_scores) > 1
                else np.nan
            ),
        }
        for split_number, score in zip(args.splits, split_scores):
            row[f"split_{split_number}_auc_roc"] = score
        per_dataset_rows.append(row)

    if len(dataset_means) != len(datasets):
        raise RuntimeError(
            f"{model_label}: evaluated {len(dataset_means)} of {len(datasets)} datasets."
        )
    return float(np.mean(dataset_means)), per_dataset_rows


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
            f"Requested {len(dids)} datasets but only {len(datasets)} passed preparation."
        )

    benchmark.MODELS = MODELS

    # All elastic depths reuse one loaded set of weights. TabPFN is loaded once.
    elastic_model, elastic_config = benchmark.load_model(
        elastic_name(args.depths[0]), args.device
    )
    tabpfn_model, tabpfn_config = benchmark.load_model(TABPFN_NAME, args.device)

    rows = []
    all_per_dataset_rows = []
    for num_loops in args.depths:
        model_label = elastic_name(num_loops)
        average_score, per_dataset_rows = evaluate_model(
            elastic_model,
            elastic_config,
            model_label,
            datasets,
            args,
        )
        all_per_dataset_rows.extend(per_dataset_rows)
        rows.append(
            {
                "model": "elastic_6x3",
                "num_loops": num_loops,
                "block_configuration": f"({num_loops},{num_loops},{num_loops})",
                "effective_transformer_layers": 3 * num_loops,
                "dataset_count": len(datasets),
                "splits_per_dataset": len(args.splits),
                "average_auc_roc": average_score,
            }
        )

    tabpfn_average, tabpfn_per_dataset_rows = evaluate_model(
        tabpfn_model,
        tabpfn_config,
        TABPFN_NAME,
        datasets,
        args,
    )
    all_per_dataset_rows.extend(tabpfn_per_dataset_rows)
    rows.append(
        {
            "model": TABPFN_NAME,
            "num_loops": np.nan,
            "block_configuration": "fixed",
            "effective_transformer_layers": tabpfn_config.get("nlayers", np.nan),
            "dataset_count": len(datasets),
            "splits_per_dataset": len(args.splits),
            "average_auc_roc": tabpfn_average,
        }
    )

    summary = pd.DataFrame(rows)
    per_dataset = pd.DataFrame(all_per_dataset_rows)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    per_dataset_output_dir = os.path.dirname(args.per_dataset_csv)
    if per_dataset_output_dir:
        os.makedirs(per_dataset_output_dir, exist_ok=True)
    per_dataset.to_csv(args.per_dataset_csv, index=False)

    print("\nAverage AUC-ROC across OpenML datasets:")
    print(summary.to_string(index=False))
    print(f"\nSaved aggregate results to {args.output_csv}")
    print(f"Saved per-dataset results to {args.per_dataset_csv}")


if __name__ == "__main__":
    main()
