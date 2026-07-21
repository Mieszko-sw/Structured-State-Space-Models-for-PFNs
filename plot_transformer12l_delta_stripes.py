import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REFERENCE_SCORE = "original_transformer_12l"
REFERENCE_TIME = "original_transformer_12l"
COMPARISON_SCORE = "hybrid_8_layers_latest"
COMPARISON_TIME = "hybrid_8_layers_latest"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Hybrid 8L deltas relative to the actual Transformer 12L columns."
    )
    parser.add_argument(
        "--score-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval.csv"),
    )
    parser.add_argument(
        "--time-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval_inference_time.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("result_csvs", "dataset_delta_stripes_hybrid8_vs_actual_transformer_12l.png"),
    )
    parser.add_argument("--metric", default="AUC")
    return parser.parse_args()


def require_columns(df, path, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def main():
    args = parse_args()
    score_df = pd.read_csv(args.score_csv).sort_values("did")
    time_df = pd.read_csv(args.time_csv).sort_values("did")

    require_columns(score_df, args.score_csv, ["did", REFERENCE_SCORE, COMPARISON_SCORE])
    require_columns(time_df, args.time_csv, ["did", REFERENCE_TIME, COMPARISON_TIME])

    df = score_df[["did", REFERENCE_SCORE, COMPARISON_SCORE]].merge(
        time_df[["did", REFERENCE_TIME, COMPARISON_TIME]],
        on="did",
        suffixes=("_score", "_time"),
    )

    dids = df["did"].tolist()
    x = np.arange(len(dids))
    score_delta = df[f"{COMPARISON_SCORE}_score"] - df[f"{REFERENCE_SCORE}_score"]
    time_delta = df[f"{COMPARISON_TIME}_time"] - df[f"{REFERENCE_TIME}_time"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(13.0, 0.48 * len(dids) + 4.0), 8.2),
        sharex=True,
    )

    color = "#54A24B"
    axes[0].bar(x, score_delta, width=0.62, color=color, label="Hybrid 8L")
    axes[1].bar(x, time_delta, width=0.62, color=color, label="Hybrid 8L")

    axes[0].set_title("Hybrid 8L performance difference vs actual Transformer 12L")
    axes[0].set_ylabel(f"Delta {args.metric}")
    axes[1].set_title("Hybrid 8L inference-time difference vs actual Transformer 12L")
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

    print(f"Saved corrected delta stripe plot to {args.output}")
    print(f"mean_delta_auc={score_delta.mean():.6f}")
    print(f"mean_delta_seconds={time_delta.mean():.6f}")


if __name__ == "__main__":
    main()
