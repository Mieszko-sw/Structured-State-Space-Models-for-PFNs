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
    "hybrid_8_layers": "#54A24B",
    "new_looped_12_layers": "#B279A2",
    "nanotabpfn": "#E45756",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot paired per-dataset score deltas against a baseline model."
    )
    parser.add_argument(
        "--split-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison_splits.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("result_csvs", "dataset_delta_vs_original_tabpfn.png"),
    )
    parser.add_argument("--baseline", default="original_tabpfn")
    parser.add_argument("--metric", default="AUC")
    parser.add_argument("--confidence", type=float, default=0.95)
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


def summarize_paired_deltas(df, models, baseline, confidence):
    rows = []
    for did, did_df in df.groupby("did", sort=True):
        baseline_scores = did_df[baseline].astype(float)
        for model in models:
            if model == baseline:
                continue
            deltas = did_df[model].astype(float) - baseline_scores
            n = len(deltas)
            mean = deltas.mean()
            std = deltas.std(ddof=1) if n > 1 else 0.0
            ci = t_critical(confidence, n) * std / np.sqrt(n) if n > 1 else 0.0
            rows.append({"did": did, "model": model, "mean_delta": mean, "ci": ci, "n": n})
    return pd.DataFrame(rows)


def dataset_order(df, baseline):
    other_models = [column for column in model_columns(df) if column != baseline]
    mean_deltas = (
        df.groupby("did")[other_models]
        .mean()
        .sub(df.groupby("did")[baseline].mean(), axis=0)
        .abs()
        .max(axis=1)
    )
    return list(mean_deltas.sort_values(ascending=False).index)


def plot_delta(summary_df, models, dids, baseline, output, metric):
    compared_models = [model for model in models if model != baseline]
    y = np.arange(len(dids))
    height = 0.80 / max(len(compared_models), 1)
    offsets = (np.arange(len(compared_models)) - (len(compared_models) - 1) / 2.0) * height

    fig_height = max(8.0, 0.34 * len(dids) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))

    for model_idx, model in enumerate(compared_models):
        model_df = summary_df[summary_df["model"] == model].set_index("did").loc[dids]
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model)
        ax.errorbar(
            model_df["mean_delta"],
            y + offsets[model_idx],
            xerr=model_df["ci"],
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.0,
            elinewidth=1.0,
            color=color,
            label=label,
        )

    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(
        f"Paired per-dataset {metric} delta vs {MODEL_LABELS.get(baseline, baseline)}"
    )
    ax.set_xlabel(f"Mean {metric} delta over matched splits")
    ax.set_ylabel("OpenML dataset ID")
    ax.set_yticks(y)
    ax.set_yticklabels([str(did) for did in dids])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_csv(args.split_csv)
    models = model_columns(df)
    if args.baseline not in models:
        raise ValueError(f"Baseline '{args.baseline}' is not a model column in {args.split_csv}")

    summary_df = summarize_paired_deltas(df, models, args.baseline, args.confidence)
    dids = dataset_order(df, args.baseline)
    plot_delta(summary_df, models, dids, args.baseline, args.output, args.metric)
    print(f"Saved paired-delta plot to {args.output}")


if __name__ == "__main__":
    main()
