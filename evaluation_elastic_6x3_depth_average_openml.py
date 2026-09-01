"""Evaluate elastic 6x3 at depths 1-6 and report one OpenML mean per depth.

For every depth, each of the 30 datasets is evaluated on splits 1-5. Scores
are first averaged across splits within each dataset and then macro-averaged
across datasets, so every OpenML dataset has equal weight. The only output CSV
is the compact depth-level table; no per-run or per-dataset CSV is written.
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
MODEL_PATH = (
    "tabpfn/models_diff/"
    "elastic_looped_transformer_3physical_core6x_18l.cpkt"
)


def model_name(num_loops):
    return f"elastic_6x3_depth_{num_loops}"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        choices=range(1, MAX_LOOPS + 1),
        default=list(range(1, MAX_LOOPS + 1)),
        metavar="N",
        help="Complete ABC-stack repetitions to evaluate (default: 1 2 3 4 5 6).",
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
        help="Evaluation splits to average within each dataset (default: 1 2 3 4 5).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(
            "result_csvs", "elastic_6x3_depth_average_openml.csv"
        ),
    )
    return parser.parse_args()


def evaluate_one(model, config, dataset_list, device, split_number, num_loops):
    eval_positions = config["eval_positions"]
    if len(eval_positions) != 1:
        raise ValueError(f"Expected one evaluation position, got {eval_positions}")

    with torch.no_grad():
        result = benchmark.evaluate(
            datasets=dataset_list,
            bptt=benchmark.BPTT,
            eval_positions=eval_positions,
            metric_used=benchmark.METRIC_USED,
            model=model,
            device=device,
            method_name="transformer",
            max_time=300,
            split_number=split_number,
            jrt_prompt=False,
            random_premutation=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            num_loops=num_loops,
        )
    return benchmark.extract_metric(result)


def main():
    args = parse_args()
    args.depths = list(dict.fromkeys(args.depths))
    args.splits = list(dict.fromkeys(args.splits))
    args.device = benchmark.normalize_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    eval_helper = EvalHelper()
    default_dids = eval_helper.openml_cc18_dids_small
    dids = args.dids if args.dids is not None else default_dids
    datasets = benchmark.prepare_datasets(dids)
    if len(datasets) != len(dids):
        raise RuntimeError(
            f"Requested {len(dids)} datasets but only {len(datasets)} passed preparation."
        )

    # load_model reads its model description from benchmark.MODELS. All depths
    # share the same checkpoint, so load it once and vary only num_loops.
    benchmark.MODELS = MODELS
    model, config = benchmark.load_model(model_name(args.depths[0]), args.device)

    rows = []
    for num_loops in args.depths:
        dataset_means = []
        for did, dataset_list in datasets.items():
            split_scores = []
            for split_number in args.splits:
                score = evaluate_one(
                    model,
                    config,
                    dataset_list,
                    args.device,
                    split_number,
                    num_loops,
                )
                split_scores.append(score)
                print(
                    f"depth={num_loops} did={did} split={split_number} "
                    f"auc_roc={score:.6f}",
                    flush=True,
                )

            dataset_means.append(float(np.mean(split_scores)))

        rows.append(
            {
                "num_loops": num_loops,
                "block_configuration": f"({num_loops},{num_loops},{num_loops})",
                "effective_block_applications": 3 * num_loops,
                "dataset_count": len(dataset_means),
                "splits_per_dataset": len(args.splits),
                "average_auc_roc": float(np.mean(dataset_means)),
            }
        )

    summary = pd.DataFrame(rows)
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    print("\nAverage AUC-ROC across OpenML datasets:")
    print(summary.to_string(index=False))
    print(f"\nSaved {args.output_csv}")


if __name__ == "__main__":
    main()
