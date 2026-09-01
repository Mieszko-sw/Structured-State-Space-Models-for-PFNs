"""Synchronized per-dataset evaluation for recently trained Tydra models."""

import argparse
import os

import evaluation_per_dataset_speed_synchronized as benchmark


MODELS = {
    "hydra_16M": {
        "path": "tabpfn/models_diff/callback_pure_hydra_9_layers_latest.cpkt",
        "loader_type": "hydra",
        "method_name": "hydra",
    },
    "HHTTTTHH": {
        "path": (
            "tabpfn/models_diff/"
            "callback_hydra2_tabpfn4_hydra2_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
        ),
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
    "HTTTTTTH": {
        "path": (
            "tabpfn/models_diff/"
            "callback_hydra1_tabpfn6_hydra1_hybrid_8_layers_512e_lr0p0001_latest.cpkt"
        ),
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare synchronized per-dataset inference speed and predictive "
            "performance for hydra_16M, HHTTTTHH, and HTTTTTTH on the real "
            "OpenML datasets used by evaluation_script.py."
        )
    )
    parser.add_argument("--device", default="cuda:4")
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
            "per_dataset_speed_synchronized_hydra16m_hhtttthh_htttttth_raw.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_hydra16m_hhtttthh_htttttth_summary.csv",
        ),
    )
    return parser.parse_args()


# Reuse the benchmark mechanics while replacing its model registry and CLI.
benchmark.MODELS = MODELS
benchmark.parse_args = parse_args


if __name__ == "__main__":
    benchmark.main()
