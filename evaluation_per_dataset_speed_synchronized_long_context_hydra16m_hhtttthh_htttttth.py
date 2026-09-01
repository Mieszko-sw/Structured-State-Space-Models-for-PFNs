"""Long-context synchronized evaluation for recently trained Tydra models."""

import argparse
import os

import evaluation_per_dataset_speed_synchronized_long_context as benchmark


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
            "Compare synchronized inference speed and predictive performance "
            "for hydra_16M, HHTTTTHH, and HTTTTTTH at 32k context length on "
            "TabArena-v0.1 classification datasets with N * d > 200,000 and "
            "at most 100 features."
        )
    )
    parser.add_argument("--device", default="cuda:7")
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
            "per_dataset_speed_synchronized_long_context_"
            "hydra16m_hhtttthh_htttttth_raw_tabarena_16.csv",
        ),
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(
            "result_csvs",
            "per_dataset_speed_synchronized_long_context_"
            "hydra16m_hhtttthh_htttttth_summary_tabarena_16.csv",
        ),
    )
    return parser.parse_args()


# Reuse the 32k benchmark mechanics while replacing its model registry and CLI.
benchmark.MODELS = MODELS
benchmark.parse_args = parse_args


if __name__ == "__main__":
    benchmark.main()
