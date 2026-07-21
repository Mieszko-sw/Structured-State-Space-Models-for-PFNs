import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot lr_new Hybrid 8L and Hydra deltas against the actual Transformer 12L reference."
        )
    )
    parser.add_argument(
        "--reference-score-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval.csv"),
    )
    parser.add_argument(
        "--reference-time-csv",
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
            "dataset_delta_stripes_lr_new_hybrid8_hydra22m_vs_transformer_12l.png",
        ),
    )
    parser.add_argument("--metric", default="AUC")
    return parser.parse_args()


def main():
    args = parse_args()
    reference_score = pd.read_csv(args.reference_score_csv)
    reference_time = pd.read_csv(args.reference_time_csv)
    hybrid_score = pd.read_csv(args.hybrid_score_csv)
    hybrid_time = pd.read_csv(args.hybrid_time_csv)

    df = (
        reference_score[["did", "original_transformer_12l", "pure_hydra_12_layers_512e"]]
        .merge(
            reference_time[["did", "original_transformer_12l", "pure_hydra_12_layers_512e"]],
            on="did",
            suffixes=("_score", "_time"),
        )
        .merge(hybrid_score[["did", "lr_new_hybrid_8l"]], on="did")
        .merge(hybrid_time[["did", "lr_new_hybrid_8l"]], on="did", suffixes=("_score", "_time"))
        .sort_values("did")
    )

    dids = df["did"].tolist()
    x = np.arange(len(dids))
    width = 0.32
    offsets = [-width / 1.8, width / 1.8]

    series = [
        {
            "label": "lr_new Hybrid 8L",
            "color": "#54A24B",
            "score_delta": df["lr_new_hybrid_8l_score"] - df["original_transformer_12l_score"],
            "time_delta": df["lr_new_hybrid_8l_time"] - df["original_transformer_12l_time"],
        },
        {
            "label": "Hydra 22M",
            "color": "#72B7B2",
            "score_delta": df["pure_hydra_12_layers_512e_score"]
            - df["original_transformer_12l_score"],
            "time_delta": df["pure_hydra_12_layers_512e_time"]
            - df["original_transformer_12l_time"],
        },
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(13.0, 0.48 * len(dids) + 4.0), 8.2),
        sharex=True,
    )

    for idx, item in enumerate(series):
        axes[0].bar(
            x + offsets[idx],
            item["score_delta"],
            width=width,
            color=item["color"],
            label=item["label"],
        )
        axes[1].bar(
            x + offsets[idx],
            item["time_delta"],
            width=width,
            color=item["color"],
            label=item["label"],
        )

    axes[0].set_title("Performance difference vs actual Transformer 12L")
    axes[0].set_ylabel(f"Delta {args.metric}")
    axes[1].set_title("Inference-time difference vs actual Transformer 12L")
    axes[1].set_ylabel("Delta inference seconds")
    axes[1].set_xlabel("OpenML dataset ID")

    for ax in axes:
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
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
    for item in series:
        print(f"{item['label']} mean_delta_auc={item['score_delta'].mean():.6f}")
        print(f"{item['label']} mean_delta_seconds={item['time_delta'].mean():.6f}")
        print(f"{item['label']} median_delta_seconds={item['time_delta'].median():.6f}")


if __name__ == "__main__":
    main()
