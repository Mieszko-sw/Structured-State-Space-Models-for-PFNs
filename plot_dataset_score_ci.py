import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NON_MODEL_COLUMNS = {"did", "split", "best_model", "best_score", "margin_to_second"}
MODEL_LABELS = {
    "original_hydra": "Original Hydra",
    "recent_hydra_25m": "Hydra 25M",
    "original_tabpfn": "Original TabPFN",
    "hybrid_8_layers": "Hybrid 8L",
    "new_looped_12_layers": "Looped 12L",
    "nanotabpfn": "NanoTabPFN",
}
MODEL_COLORS = {
    "original_hydra": "#4C78A8",
    "recent_hydra_25m": "#72B7B2",
    "original_tabpfn": "#F58518",
    "hybrid_8_layers": "#54A24B",
    "new_looped_12_layers": "#B279A2",
    "nanotabpfn": "#E45756",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-dataset model scores as mean +/- confidence interval over splits."
    )
    parser.add_argument(
        "--split-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_splits.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("result_csvs", "dataset_score_ci.png"),
    )
    parser.add_argument("--metric", default="AUC")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="Model columns to omit from the plot.",
    )
    return parser.parse_args()


def model_columns(df):
    return [column for column in df.columns if column not in NON_MODEL_COLUMNS]


def t_critical(confidence, n):
    if n <= 1:
        return 0.0
    try:
        from scipy import stats

        return stats.t.ppf((1.0 + confidence) / 2.0, n - 1)
    except Exception:
        return 1.96


def summarize_scores(df, models, confidence):
    rows = []
    for did, did_df in df.groupby("did", sort=True):
        for model in models:
            scores = did_df[model].dropna().astype(float)
            n = len(scores)
            mean = scores.mean()
            std = scores.std(ddof=1) if n > 1 else 0.0
            ci = t_critical(confidence, n) * std / np.sqrt(n) if n > 1 else 0.0
            rows.append({"did": did, "model": model, "mean": mean, "ci": ci, "n": n})
    return pd.DataFrame(rows)


def plot_score_ci(summary_df, models, output, metric):
    dids = list(summary_df["did"].drop_duplicates())
    x = np.arange(len(dids))
    width = 0.82 / max(len(models), 1)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width

    fig_width = max(12.0, 0.45 * len(dids) + 4.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    for model_idx, model in enumerate(models):
        model_df = summary_df[summary_df["model"] == model].set_index("did").loc[dids]
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model)
        ax.errorbar(
            x + offsets[model_idx],
            model_df["mean"],
            yerr=model_df["ci"],
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.0,
            elinewidth=1.0,
            color=color,
            label=label,
        )

    ax.set_title(f"Per-dataset {metric}: mean +/- 95% CI over splits")
    ax.set_xlabel("OpenML dataset ID")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels([str(did) for did in dids], rotation=45, ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_csv(args.split_csv)
    excluded_models = set(args.exclude_models)
    models = [model for model in model_columns(df) if model not in excluded_models]
    summary_df = summarize_scores(df, models, args.confidence)
    plot_score_ci(summary_df, models, args.output, args.metric)
    print(f"Saved per-dataset score plot to {args.output}")


if __name__ == "__main__":
    main()
