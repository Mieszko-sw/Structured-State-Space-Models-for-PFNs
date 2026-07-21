import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    "/root/.codex/attachments/71b78d7a-d03a-4df2-a8d7-99ea1650367f/pasted-text.txt"
)
MODEL_PATTERN = re.compile(r"\s(hybrid_8l|hydra|tabpfn)\s+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot synchronized per-dataset speed deltas against TabPFN."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--parsed-csv",
        default=os.path.join(
            "result_csvs", "per_dataset_speed_synchronized_summary_from_text.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            "result_csvs", "synchronized_speed_stripes_hybrid8l_hydra22m_vs_tabpfn.png"
        ),
    )
    return parser.parse_args()


def parse_summary_text(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            match = MODEL_PATTERN.search(line)
            if not match:
                continue

            prefix = line[: match.start()].strip()
            suffix = line[match.end() :].strip().split()
            prefix_parts = prefix.split(maxsplit=1)
            if len(prefix_parts) != 2:
                continue

            did_text, dataset_name = prefix_parts
            model = match.group(1)
            if len(suffix) < 9:
                continue

            rows.append(
                {
                    "did": int(did_text),
                    "dataset_name": dataset_name,
                    "model": model,
                    "num_samples": int(suffix[0]),
                    "num_features": int(suffix[1]),
                    "eval_position": int(suffix[2]),
                    "mean": float(suffix[3]),
                    "median": float(suffix[4]),
                    "std": float(suffix[5]),
                    "min": float(suffix[6]),
                    "max": float(suffix[7]),
                    "count": int(suffix[8]),
                    "mean_metric": float(suffix[9]) if len(suffix) > 9 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def load_summary(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return parse_summary_text(path)


def main():
    args = parse_args()
    summary = load_summary(args.input)
    if summary.empty:
        raise ValueError(f"No synchronized summary rows parsed from {args.input}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    parsed_dir = os.path.dirname(args.parsed_csv)
    if parsed_dir:
        os.makedirs(parsed_dir, exist_ok=True)
    summary.to_csv(args.parsed_csv, index=False)

    wide = summary.pivot_table(
        index=["did", "dataset_name", "num_samples", "num_features"],
        columns="model",
        values="mean",
        aggfunc="first",
    ).reset_index()

    required = ["tabpfn", "hybrid_8l", "hydra"]
    missing = [column for column in required if column not in wide.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")

    wide = wide.sort_values("did")
    dids = wide["did"].astype(str).tolist()
    x = np.arange(len(wide))
    width = 0.34

    series = [
        ("Hybrid 8L", "hybrid_8l", "#54A24B", -width / 1.8),
        ("Hydra 22M", "hydra", "#72B7B2", width / 1.8),
    ]

    fig, ax = plt.subplots(figsize=(max(13.0, 0.48 * len(wide) + 4.0), 5.8))
    for label, column, color, offset in series:
        speedup_percent = (wide["tabpfn"] / wide[column] - 1.0) * 100.0
        ax.bar(x + offset, speedup_percent, width=width, color=color, label=label)
        print(f"{label} mean_speedup_percent={speedup_percent.mean():.2f}")
        print(f"{label} median_speedup_percent={speedup_percent.median():.2f}")

    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Synchronized Inference Speed vs TabPFN by OpenML Benchmark")
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
        "Source: synchronized elapsed mean timings. TabPFN is zero; positive means faster.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved parsed CSV to {args.parsed_csv}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
