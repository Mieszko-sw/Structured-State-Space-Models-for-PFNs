"""Synchronized per-dataset evaluation for the TTHHHHTT and THHHHHHT models."""

import argparse
import os

import evaluation_per_dataset_speed_synchronized as benchmark


MODELS = {
    "TTHHHHTT": {
        "path": (
            "tabpfn/models_diff/"
            "callback_tabpfn2_hydra4_tabpfn2_hybrid_8_layers_512e_"
            "lr0p0001_latest.cpkt"
        ),
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
    "THHHHHHT": {
        "path": (
            "tabpfn/models_diff/"
            "callback_tabpfn1_hydra6_tabpfn1_hybrid_8_layers_512e_"
            "lr0p0001_latest.cpkt"
        ),
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare synchronized per-dataset inference speed and predictive "
            "performance for the recently trained TTHHHHTT and THHHHHHT "
            "models on the real OpenML datasets used by evaluation_script.py."
        )
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        choices=list(MODELS),
    )
    parser.add_argument("--dids", nargs="+", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=15,
        help="Timed repetitions per benchmark split (splits 1-5 match evaluation_script.py).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--raw-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_tthhhhtt_thhhhhht_raw.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_tthhhhtt_thhhhhht_summary.csv",
        ),
    )
    return parser.parse_args()


# Reuse the benchmark mechanics while replacing its model registry and CLI.
benchmark.MODELS = MODELS
benchmark.parse_args = parse_args


if __name__ == "__main__":
    benchmark.main()
