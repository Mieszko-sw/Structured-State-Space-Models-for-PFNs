import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_dataset_performance_inference import MODEL_SPECS, read_inference, read_performance


REFERENCE_MODEL = "original_tabpfn"
DEFAULT_COMPARISON_MODELS = ["hybrid_8_layers", "recent_hydra_25m"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Hybrid 8L and Hydra 22M deltas against Transformer 12L "
            "for per-dataset performance and inference time."
        )
    )
    parser.add_argument(
        "--performance-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison.csv"),
    )
    parser.add_argument(
        "--inference-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval_inference_time.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("result_csvs", "dataset_delta_stripes_vs_transformer_12l.png"),
    )
    parser.add_argument("--metric", default="AUC")
    parser.add_argument(
        "--comparison-models",
        nargs="+",
        choices=["hybrid_8_layers", "recent_hydra_25m"],
        default=DEFAULT_COMPARISON_MODELS,
    )
    return parser.parse_args()


def inference_pivot(inference_df, dids):
    return (
        inference_df.pivot_table(
            index="did",
            columns="model",
            values="mean_inference_time_seconds",
            aggfunc="mean",
        )
        .reindex(dids)
    )


def plot_delta_stripes(performance_df, inference_df, args):
    dids = performance_df["did"].tolist()
    x = np.arange(len(dids))
    stripe_width = 0.72
    offsets = (
        np.arange(len(args.comparison_models)) - (len(args.comparison_models) - 1) / 2.0
    ) * (stripe_width / max(len(args.comparison_models), 1))

    ref_score = performance_df[MODEL_SPECS[REFERENCE_MODEL]["score_column"]]
    time_df = inference_pivot(inference_df, dids)
    if REFERENCE_MODEL not in time_df.columns:
        raise ValueError("Transformer 12L timing column is missing after model-name normalization.")
    ref_time = time_df[REFERENCE_MODEL]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(13.0, 0.48 * len(dids) + 4.0), 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    for model_idx, model in enumerate(args.comparison_models):
        spec = MODEL_SPECS[model]
        score_delta = performance_df[spec["score_column"]] - ref_score
        axes[0].bar(
            x + offsets[model_idx],
            score_delta,
            width=stripe_width / 2.5,
            color=spec["color"],
            label=spec["label"],
        )

        if model not in time_df.columns:
            continue
        time_delta = time_df[model] - ref_time
        axes[1].bar(
            x + offsets[model_idx],
            time_delta,
            width=stripe_width / 2.5,
            color=spec["color"],
            label=spec["label"],
        )

    axes[0].axhline(0.0, color="#222222", linewidth=1.0)
    axes[0].set_ylabel(f"Δ {args.metric}")
    axes[0].set_title("Performance difference vs Transformer 12L")

    axes[1].axhline(0.0, color="#222222", linewidth=1.0)
    axes[1].set_ylabel("Δ inference seconds")
    axes[1].set_title("Inference-time difference vs Transformer 12L")
    axes[1].set_xlabel("OpenML dataset ID")

    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(ncol=len(args.comparison_models), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(did) for did in dids], rotation=45, ha="right")

    fig.text(
        0.01,
        0.01,
        "Positive Δ AUC means better than Transformer 12L. Negative Δ inference seconds means faster.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    performance_df = read_performance(args.performance_csv)
    inference_df = read_inference(args.inference_csv)
    if inference_df is None:
        raise FileNotFoundError(f"Per-dataset inference CSV not found: {args.inference_csv}")

    plot_delta_stripes(performance_df, inference_df, args)
    print(f"Saved delta stripe plot to {args.output}")


if __name__ == "__main__":
    main()
