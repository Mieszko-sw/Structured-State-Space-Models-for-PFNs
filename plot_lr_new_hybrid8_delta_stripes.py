import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot lr_new_hybrid_8l deltas against the actual Transformer 12L reference."
    )
    parser.add_argument(
        "--transformer-score-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval.csv"),
    )
    parser.add_argument(
        "--transformer-time-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval_inference_time.csv"),
    )
    parser.add_argument(
        "--hybrid-score-csv",
        default=os.path.join("result_csvs", "lr_new_hybrid_8l_eval.csv"),
    )
    parser.add_argument(
        "--hybrid-time-csv",
        default=os.path.join("result_csvs", "lr_new_hybrid_8l_eval_inference_time.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            "result_csvs",
            "dataset_delta_stripes_lr_new_hybrid8_vs_actual_transformer_12l.png",
        ),
    )
    parser.add_argument("--metric", default="AUC")
    return parser.parse_args()


def main():
    args = parse_args()
    transformer_score = pd.read_csv(args.transformer_score_csv)
    transformer_time = pd.read_csv(args.transformer_time_csv)
    hybrid_score = pd.read_csv(args.hybrid_score_csv)
    hybrid_time = pd.read_csv(args.hybrid_time_csv)

    df = (
        transformer_score[["did", "original_transformer_12l"]]
        .merge(transformer_time[["did", "original_transformer_12l"]], on="did", suffixes=("_score", "_time"))
        .merge(hybrid_score[["did", "lr_new_hybrid_8l"]], on="did")
        .merge(hybrid_time[["did", "lr_new_hybrid_8l"]], on="did", suffixes=("_score", "_time"))
        .sort_values("did")
    )

    dids = df["did"].tolist()
    x = np.arange(len(dids))
    score_delta = df["lr_new_hybrid_8l_score"] - df["original_transformer_12l_score"]
    time_delta = df["lr_new_hybrid_8l_time"] - df["original_transformer_12l_time"]

    fig, axes = plt.subplots(2, 1, figsize=(max(13.0, 0.48 * len(dids) + 4.0), 8.2), sharex=True)
    color = "#54A24B"
    axes[0].bar(x, score_delta, width=0.62, color=color, label="lr_new Hybrid 8L")
    axes[1].bar(x, time_delta, width=0.62, color=color, label="lr_new Hybrid 8L")

    axes[0].set_title("lr_new Hybrid 8L performance difference vs actual Transformer 12L")
    axes[0].set_ylabel(f"Delta {args.metric}")
    axes[1].set_title("lr_new Hybrid 8L inference-time difference vs actual Transformer 12L")
    axes[1].set_ylabel("Delta inference seconds")
    axes[1].set_xlabel("OpenML dataset ID")

    for ax in axes:
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(did) for did in dids], rotation=45, ha="right")
    fig.text(
        0.01,
        0.01,
        "Positive Delta AUC means better than Transformer 12L. Negative Delta inference seconds means faster.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {args.output}")
    print(f"mean_delta_auc={score_delta.mean():.6f}")
    print(f"mean_delta_seconds={time_delta.mean():.6f}")
    print(f"median_delta_seconds={time_delta.median():.6f}")


if __name__ == "__main__":
    main()
