"""Plot per-dataset AUC and inference confidence intervals for four models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SUMMARY_CSV = Path("result_csvs/all_runs_15.csv")
MAIN_RAW_CSV = Path("result_csvs/per_dataset_speed_synchronized_raw_15.csv")
HYDRA16_RAW_CSV = Path(
    "result_csvs/per_dataset_speed_synchronized_hydra16m_hhtttthh_htttttth_raw.csv"
)
OUTPUT_DIR = Path("result_csvs")

MODELS = ["hybrid_8l", "hydra_16M", "hydra_small", "tabpfn"]
MODEL_LABELS = {
    "hybrid_8l": "Hybrid 8L (ours)",
    "hydra_16M": "Hydra 16M",
    "hydra_small": "Hydra 160M",
    "tabpfn": "TabPFN",
}

# Exact colors from the supplied PGFPlots example; gray is used for TabPFN.
MODEL_COLORS = {
    "hybrid_8l": "#1F77B4",
    "hydra_16M": "#D9A05B",
    "hydra_small": "#FF7F0E",
    "tabpfn": "#555555",
}

# Preserve the dataset ordering from the supplied plot.
DATASET_ORDER = [
    31,
    50,
    1494,
    1049,
    40966,
    1050,
    23,
    1068,
    40975,
    1462,
    54,
    37,
    469,
    15,
    1464,
    458,
    11,
    29,
    188,
    14,
    22,
    18,
    40982,
    16,
    1063,
    40994,
    1480,
    1510,
    6332,
    23381,
]


def t_interval_half_width(std, n, confidence=0.95):
    """Return a two-sided Student-t confidence-interval half-width."""
    std = pd.to_numeric(std, errors="coerce")
    n = pd.to_numeric(n, errors="coerce")
    critical = stats.t.ppf((1.0 + confidence) / 2.0, n - 1)
    return critical * std / np.sqrt(n)


def load_auc_summary(summary):
    """Compute per-dataset AUC intervals across five evaluation splits."""
    main_raw = pd.read_csv(MAIN_RAW_CSV)
    hydra16_raw = pd.read_csv(HYDRA16_RAW_CSV)

    raw = pd.concat(
        [
            main_raw[
                main_raw["model"].isin(["hybrid_8l", "hydra_small", "tabpfn"])
            ],
            hydra16_raw[hydra16_raw["model"] == "hydra_16M"],
        ],
        ignore_index=True,
    )
    raw = raw[raw["status"] == "ok"].copy()

    # AUC is repeated for every timing repetition. Collapse repetitions first
    # so that the five data splits, rather than 75 timing rows, are the units.
    split_scores = (
        raw.groupby(["did", "model", "split_number"], as_index=False)["mean_metric"]
        .mean()
    )
    auc_ci = (
        split_scores.groupby(["did", "model"])["mean_metric"]
        .agg(std="std", n="count")
        .reset_index()
    )
    auc_ci["ci"] = t_interval_half_width(auc_ci["std"], auc_ci["n"])

    # Use the exact means reported by all_runs_15.csv.
    means = summary[["did", "model", "mean_metric"]].rename(
        columns={"mean_metric": "mean"}
    )
    return means.merge(auc_ci[["did", "model", "ci", "n"]], on=["did", "model"])


def load_inference_summary(summary):
    """Compute per-dataset inference intervals from the summarized 75 timings."""
    result = summary[
        ["did", "model", "inference_mean", "inference_std", "count"]
    ].copy()
    result["mean"] = 1000.0 * result["inference_mean"]
    result["ci"] = 1000.0 * t_interval_half_width(
        result["inference_std"], result["count"]
    )
    return result.rename(columns={"count": "n"})[
        ["did", "model", "mean", "ci", "n"]
    ]


def plot_dataset_ci(data, ylabel, title, output_stem, higher_is_better):
    """Match the marker-and-whisker style of plot_dataset_score_ci.py."""
    x = np.arange(len(DATASET_ORDER))
    width = 0.82 / len(MODELS)
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * width

    fig_width = max(12.0, 0.45 * len(DATASET_ORDER) + 4.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    for model_idx, model in enumerate(MODELS):
        model_data = (
            data[data["model"] == model]
            .set_index("did")
            .loc[DATASET_ORDER]
        )
        ax.errorbar(
            x + offsets[model_idx],
            model_data["mean"],
            yerr=model_data["ci"],
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.0,
            elinewidth=1.0,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )

    ax.set_title(title)
    ax.set_xlabel("OpenML dataset ID")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(did) for did in DATASET_ORDER],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    ax.text(
        0.995,
        0.985,
        "higher is better" if higher_is_better else "lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#666666",
    )

    if higher_is_better:
        ax.set_ylim(0.4, 1.03)
    else:
        ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        destination = OUTPUT_DIR / f"{output_stem}{suffix}"
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        print(f"Saved {destination}")
    plt.close(fig)


def main():
    summary = pd.read_csv(SUMMARY_CSV)
    summary = summary[summary["model"].isin(MODELS)].copy()

    auc = load_auc_summary(summary)
    inference = load_inference_summary(summary)

    plot_dataset_ci(
        auc,
        ylabel="AUC-ROC",
        title="Per-dataset AUC-ROC: mean ± 95% CI over splits",
        output_stem="all_runs_15_four_models_auc_ci",
        higher_is_better=True,
    )
    plot_dataset_ci(
        inference,
        ylabel="Inference time (ms)",
        title="Per-dataset inference time: mean ± 95% CI",
        output_stem="all_runs_15_four_models_inference_ci",
        higher_is_better=False,
    )


if __name__ == "__main__":
    main()
