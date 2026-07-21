import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-dataset inference speed deltas against TabPFN."
    )
    parser.add_argument(
        "--time-csv",
        default=os.path.join("result_csvs", "alternating_hybrid_eval_inference_time.csv"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            "result_csvs", "inference_speed_stripes_hybrid8l_hydra22m_vs_tabpfn.png"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.time_csv)

    required_columns = [
        "did",
        "transformer",
        "hybrid_8_layers_latest",
        "pure_hydra_12_layers_512e",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {args.time_csv}: {missing_columns}")

    df = df[required_columns].sort_values("did").copy()
    dids = df["did"].astype(str).tolist()
    x = np.arange(len(dids))
    width = 0.34

    tabpfn_time = df["transformer"]
    series = [
        {
            "label": "Hybrid 8L",
            "color": "#54A24B",
            "time": df["hybrid_8_layers_latest"],
        },
        {
            "label": "Hydra 22M",
            "color": "#72B7B2",
            "time": df["pure_hydra_12_layers_512e"],
        },
    ]

    fig, ax = plt.subplots(figsize=(max(13.0, 0.48 * len(dids) + 4.0), 5.8))
    offsets = [-width / 1.8, width / 1.8]

    for idx, item in enumerate(series):
        speedup_percent = (tabpfn_time / item["time"] - 1.0) * 100.0
        ax.bar(
            x + offsets[idx],
            speedup_percent,
            width=width,
            color=item["color"],
            label=item["label"],
        )
        print(f"{item['label']} mean_speedup_percent={speedup_percent.mean():.2f}")
        print(f"{item['label']} median_speedup_percent={speedup_percent.median():.2f}")

    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Inference Speed vs TabPFN by OpenML Benchmark")
    ax.set_ylabel("Speedup vs TabPFN (%)")
    ax.set_xlabel("OpenML dataset ID")
    ax.set_xticks(x)
    ax.set_xticklabels(dids, rotation=45, ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.text(
        0.01,
        0.01,
        "TabPFN is the zero reference. Positive bars mean faster than TabPFN; negative bars mean slower.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
