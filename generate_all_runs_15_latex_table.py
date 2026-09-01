"""Generate a LaTeX per-dataset AUCROC/inference table from all_runs_15.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SUMMARY_CSV = Path("result_csvs/all_runs_15.csv")
MAIN_RAW_CSV = Path("result_csvs/per_dataset_speed_synchronized_raw_15.csv")
HYDRA16_RAW_CSV = Path(
    "result_csvs/per_dataset_speed_synchronized_hydra16m_hhtttthh_htttttth_raw.csv"
)
OUTPUT_TEX = Path("result_csvs/all_runs_15_four_models_table.tex")

MODELS = ["hybrid_8l", "hydra_16M", "hydra_small", "tabpfn"]
LABELS = {
    "hybrid_8l": "Hybrid 8L",
    "hydra_16M": "Hydra 16M",
    "hydra_small": "Hydra 160M",
    "tabpfn": "TabPFN",
}


def ci95(std: pd.Series, count: pd.Series) -> pd.Series:
    """Two-sided 95% Student-t confidence-interval half-width."""
    return stats.t.ppf(0.975, count - 1) * std / np.sqrt(count)


def format_value(mean: float, half_width: float, best: bool, digits: int) -> str:
    value = f"{mean:.{digits}f}"
    if best:
        value = rf"\textbf{{{value}}}"
    return rf"{value} $\pm$ {half_width:.{digits}f}"


summary = pd.read_csv(SUMMARY_CSV)
summary = summary[summary["model"].isin(MODELS)].copy()

main_raw = pd.read_csv(MAIN_RAW_CSV)
hydra16_raw = pd.read_csv(HYDRA16_RAW_CSV)
raw = pd.concat(
    [
        main_raw[main_raw["model"].isin(["hybrid_8l", "hydra_small", "tabpfn"])],
        hydra16_raw[hydra16_raw["model"] == "hydra_16M"],
    ],
    ignore_index=True,
)
raw = raw[raw["status"] == "ok"].copy()

# Timing repetitions duplicate each split's AUCROC, so collapse them first.
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

if data.groupby("did")["model"].nunique().ne(len(MODELS)).any():
    raise RuntimeError("At least one dataset does not contain all four models.")

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Results on the 30 OpenML classification datasets. Values are means",
    r"with 95\% Student-$t$ confidence intervals. The best AUCROC (higher is",
    r"better) and inference time (lower is better) in each row are bold.}",
    r"\label{tab:openml_auc_inference}",
    r"\setlength{\tabcolsep}{3.2pt}",
    r"\renewcommand{\arraystretch}{1.05}",
    r"\scriptsize",
    r"\begin{tabular}{@{}rcccc@{\hspace{5pt}}cccc@{}}",
    r"\toprule",
    r"& \multicolumn{4}{c}{AUCROC $\uparrow$}"
    r" & \multicolumn{4}{c}{Inference (ms) $\downarrow$} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"DID"
    + "".join(rf" & {LABELS[model]}" for model in MODELS)
    + "".join(rf" & {LABELS[model]}" for model in MODELS)
    + r" \\",
    r"\midrule",
]

for did, group in data.groupby("did", sort=True):
    indexed = group.set_index("model").loc[MODELS]
    best_auc = indexed["mean_metric"].max()
    best_time = indexed["inference_ms"].min()
    auc_cells = [
        format_value(
            row["mean_metric"],
            row["auc_ci"],
            np.isclose(row["mean_metric"], best_auc),
            4,
        )
        for _, row in indexed.iterrows()
    ]
    time_cells = [
        format_value(
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
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
)

OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUTPUT_TEX}")
