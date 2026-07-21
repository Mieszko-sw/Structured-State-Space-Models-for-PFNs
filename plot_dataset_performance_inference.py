import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SPECS = {
    "original_tabpfn": {
        "label": "Transformer 12L",
        "score_column": "original_tabpfn_mean",
        "time_names": ["original_tabpfn", "tabpfn", "transformer_12l", "original_transformer_12l"],
        "color": "#F58518",
    },
    "hybrid_8_layers": {
        "label": "Hybrid 8L",
        "score_column": "hybrid_8_layers_mean",
        "time_names": ["hybrid_8_layers", "hybrid_8l", "hybrid_8_layers_latest"],
        "color": "#54A24B",
    },
    "recent_hydra_25m": {
        "label": "Hydra 22M",
        "score_column": "recent_hydra_25m_mean",
        "time_names": ["recent_hydra_25m", "hydra", "hydra_22m"],
        "color": "#72B7B2",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-dataset performance for Transformer 12L, Hybrid 8L, and Hydra 22M, "
            "optionally with per-dataset inference times."
        )
    )
    parser.add_argument(
        "--performance-csv",
        default=os.path.join("result_csvs", "dataset_model_comparison.csv"),
    )
    parser.add_argument(
        "--inference-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval_inference_time.csv"),
        help=(
            "CSV with columns model,did,mean_inference_time_seconds. "
            "Rows may also include split_*_inference_time_seconds."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("result_csvs", "dataset_performance_inference_three_models.png"),
    )
    parser.add_argument("--metric", default="AUC")
    parser.add_argument(
        "--allow-missing-inference",
        action="store_true",
        help="Create the performance-only plot if the per-dataset inference CSV is missing.",
    )
    return parser.parse_args()


def read_performance(path):
    df = pd.read_csv(path)
    required = ["did"] + [spec["score_column"] for spec in MODEL_SPECS.values()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing performance columns in {path}: {missing}")
    return df.sort_values("did").reset_index(drop=True)


def normalize_time_model_name(model_name):
    model_name = str(model_name)
    for canonical, spec in MODEL_SPECS.items():
        if model_name == canonical or model_name in spec["time_names"]:
            return canonical
    return model_name


def read_inference(path):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if {"model", "did", "mean_inference_time_seconds"}.issubset(df.columns):
        long_df = df[["model", "did", "mean_inference_time_seconds"]].copy()
        long_df["model"] = long_df["model"].map(normalize_time_model_name)
        long_df = long_df[long_df["model"].isin(MODEL_SPECS)]
        if long_df.empty:
            raise ValueError(f"No matching model rows found in {path}")
        return long_df

    if "did" not in df.columns:
        raise ValueError(f"Missing did column in {path}")

    rows = []
    for source_column in df.columns:
        if source_column == "did":
            continue

        model = normalize_time_model_name(source_column)
        if model not in MODEL_SPECS:
            continue

        for _, row in df[["did", source_column]].dropna().iterrows():
            rows.append({
                "model": model,
                "did": row["did"],
                "mean_inference_time_seconds": row[source_column],
            })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(f"No matching model timing columns found in {path}")
    return long_df


def plot(performance_df, inference_df, args):
    dids = performance_df["did"].tolist()
    x = np.arange(len(dids))
    width = 0.82 / len(MODEL_SPECS)
    offsets = (np.arange(len(MODEL_SPECS)) - (len(MODEL_SPECS) - 1) / 2.0) * width

    has_inference = inference_df is not None
    fig_height = 8.0 if has_inference else 5.2
    fig, axes = plt.subplots(
        2 if has_inference else 1,
        1,
        figsize=(max(13.0, 0.48 * len(dids) + 4.0), fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]} if has_inference else None,
    )
    if not has_inference:
        axes = [axes]

    score_ax = axes[0]
    for model_idx, (model, spec) in enumerate(MODEL_SPECS.items()):
        score_ax.scatter(
            x + offsets[model_idx],
            performance_df[spec["score_column"]],
            s=24,
            color=spec["color"],
            label=spec["label"],
            zorder=3,
        )

    score_ax.set_ylabel(args.metric)
    score_ax.set_title("Per-dataset performance")
    score_ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    score_ax.spines["top"].set_visible(False)
    score_ax.spines["right"].set_visible(False)
    score_ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    if has_inference:
        time_ax = axes[1]
        pivot = inference_df.pivot_table(
            index="did",
            columns="model",
            values="mean_inference_time_seconds",
            aggfunc="mean",
        )
        pivot = pivot.reindex(dids)

        for model_idx, (model, spec) in enumerate(MODEL_SPECS.items()):
            if model not in pivot.columns:
                continue
            time_ax.scatter(
                x + offsets[model_idx],
                pivot[model],
                s=24,
                color=spec["color"],
                label=spec["label"],
                zorder=3,
            )

        time_ax.set_ylabel("Inference seconds")
        time_ax.set_title("Per-dataset inference time")
        time_ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        time_ax.spines["top"].set_visible(False)
        time_ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("OpenML dataset ID")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(did) for did in dids], rotation=45, ha="right")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    performance_df = read_performance(args.performance_csv)
    inference_df = read_inference(args.inference_csv)

    if inference_df is None and not args.allow_missing_inference:
        raise FileNotFoundError(
            f"Per-dataset inference CSV not found: {args.inference_csv}. "
            "Run/create a CSV with columns model,did,mean_inference_time_seconds, "
            "or pass --allow-missing-inference to plot performance only."
        )

    plot(performance_df, inference_df, args)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
