"""Generate a LaTeX AUCROC/inference table for all 8-layer hybrid variants."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SUMMARY_CSV = Path("result_csvs/all_runs_15.csv")
RAW_CSVS = [
    Path("result_csvs/per_dataset_speed_synchronized_raw_15.csv"),
    Path(
        "result_csvs/"
        "per_dataset_speed_synchronized_hydra16m_hhtttthh_htttttth_raw.csv"
    ),
    Path("result_csvs/per_dataset_speed_synchronized_tthhhhtt_thhhhhht_raw.csv"),
    Path("result_csvs/per_dataset_speed_synchronized_thx4_raw.csv"),
]
OUTPUT_TEX = Path("result_csvs/all_runs_15_hybrid_variants_table.tex")

# H denotes a Hydra layer and T denotes a Transformer layer.
MODELS = [
    "hybrid_8l",
    "THx4",
    "HHTTTTHH",
    "HTTTTTTH",
    "TTHHHHTT",
    "THHHHHHT",
]
LABELS = {
    "hybrid_8l": "HTHTHTHT",
    "THx4": "THTHTHTH",
    "HHTTTTHH": "HHTTTTHH",
    "HTTTTTTH": "HTTTTTTH",
    "TTHHHHTT": "TTHHHHTT",
    "THHHHHHT": "THHHHHHT",
}


def ci95(std: pd.Series, count: pd.Series) -> pd.Series:
    """Return a two-sided 95% Student-t CI half-width."""
    return stats.t.ppf(0.975, count - 1) * std / np.sqrt(count)


def format_cell(mean: float, half_width: float, best: bool, digits: int) -> str:
    mean_text = f"{mean:.{digits}f}"
    if best:
        mean_text = rf"\textbf{{{mean_text}}}"
    return rf"{mean_text} $\pm$ {half_width:.{digits}f}"


summary = pd.read_csv(SUMMARY_CSV)
summary = summary[summary["model"].isin(MODELS)].copy()

raw = pd.concat([pd.read_csv(path) for path in RAW_CSVS], ignore_index=True)
raw = raw[(raw["status"] == "ok") & raw["model"].isin(MODELS)].copy()

# Each split's AUCROC is repeated for every timing repetition.
split_auc = (
    raw.groupby(["did", "model", "split_number"], as_index=False)["mean_metric"]
    .mean()
)
auc_ci = (
    split_auc.groupby(["did", "model"])["mean_metric"]
    .agg(auc_std="std", auc_n="count")
    .reset_index()
)
auc_ci["auc_ci"] = ci95(auc_ci["auc_std"], auc_ci["auc_n"])

data = summary.merge(
    auc_ci[["did", "model", "auc_ci", "auc_n"]],
    on=["did", "model"],
    validate="one_to_one",
)
data["inference_ms"] = 1000.0 * data["inference_mean"]
data["inference_ci_ms"] = 1000.0 * ci95(data["inference_std"], data["count"])

model_counts = data.groupby("did")["model"].nunique()
if len(model_counts) != 30 or model_counts.ne(len(MODELS)).any():
    raise RuntimeError("Expected all six hybrid variants for each of 30 datasets.")

header = "DID" + "".join(rf" & \texttt{{{LABELS[m]}}}" for m in MODELS)
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Comparison of the 8-layer hybrid variants on the 30 OpenML",
    r"classification datasets. Values are means with 95\% Student-$t$ confidence",
    r"intervals, computed across five evaluation splits for AUCROC and 75 timed",
    r"measurements for inference. The best result in each row is bold.",
    r"$\mathrm{H}$ denotes a Hydra layer and $\mathrm{T}$ a Transformer layer.}",
    r"\label{tab:hybrid_variants_auc_inference}",
    r"\setlength{\tabcolsep}{2.5pt}",
    r"\renewcommand{\arraystretch}{1.04}",
    r"\scriptsize",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{@{}rcccccc@{\hspace{7pt}}cccccc@{}}",
    r"\toprule",
    r"& \multicolumn{6}{c}{AUCROC $\uparrow$}"
    r" & \multicolumn{6}{c}{Inference (ms) $\downarrow$} \\",
    r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
    header + "".join(rf" & \texttt{{{LABELS[m]}}}" for m in MODELS) + r" \\",
    r"\midrule",
]

for did, group in data.groupby("did", sort=True):
    indexed = group.set_index("model").loc[MODELS]
    best_auc = indexed["mean_metric"].max()
    best_time = indexed["inference_ms"].min()
    auc_cells = [
        format_cell(
            row["mean_metric"],
            row["auc_ci"],
            np.isclose(row["mean_metric"], best_auc),
            4,
        )
        for _, row in indexed.iterrows()
    ]
    time_cells = [
        format_cell(
            row["inference_ms"],
            row["inference_ci_ms"],
            np.isclose(row["inference_ms"], best_time),
            2,
        )
        for _, row in indexed.iterrows()
    ]
    lines.append(
        str(int(did))
        + " & "
        + " & ".join(auc_cells)
        + " & "
        + " & ".join(time_cells)
        + r" \\"
    )

lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        "",
    ]
)

OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUTPUT_TEX}")
