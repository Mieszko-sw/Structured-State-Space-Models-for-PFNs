import os

import pandas as pd
import torch

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.model_builder_custom import load_model_only_inference


CHECKPOINT = "tabpfn/models_diff/lr_new_hybrid_8l.cpkt"
MODEL_NAME = "lr_new_hybrid_8l"
SCORE_CSV = os.path.join("result_csvs", "lr_new_hybrid_8l_eval.csv")
TIME_CSV = os.path.join("result_csvs", "lr_new_hybrid_8l_eval_inference_time.csv")

DEVICE = "cuda:0"
SPLIT_NUMBERS = [1, 2, 3, 4, 5]
EVALUATION_TYPE = "openmlcc18"
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}


def extract_metric(split_result):
    metric = split_result["mean_metric"]
    if hasattr(metric, "item"):
        return metric.item()
    return float(metric)


def extract_inference_time(split_result):
    time_keys = tuple(f"_time_at_{pos}" for pos in split_result["eval_positions"])
    return sum(value for key, value in split_result.items() if key.endswith(time_keys))


def mean(values):
    return sum(values) / len(values)


def write_csv(rows, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(DEVICE))

    loaded, config = load_model_only_inference(".", CHECKPOINT, DEVICE, model_name="hybrid")
    model = loaded[2]
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"{MODEL_NAME} parameters: {params:,}", flush=True)

    eval_helper = EvalHelper()
    result = eval_helper.do_evaluation_custom(
        model,
        bptt=config.get("bptt", 1024),
        eval_positions=config["eval_positions"],
        metric=tabular_metrics.auc_metric,
        device=DEVICE,
        method_name="transformer",
        evaluation_type=EVALUATION_TYPE,
        split_numbers=SPLIT_NUMBERS,
        jrt_prompt=False,
        single_evaluation_prompt=False,
        permutation_bagging=1,
        sample_bagging=0,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )

    score_rows = []
    time_rows = []
    for did, split_results in result.items():
        split_scores = [extract_metric(split_result) for split_result in split_results]
        split_times = [extract_inference_time(split_result) for split_result in split_results]

        score_row = {"did": did, MODEL_NAME: mean(split_scores)}
        time_row = {"did": did, MODEL_NAME: mean(split_times)}
        for split_number, split_score, split_time in zip(SPLIT_NUMBERS, split_scores, split_times):
            score_row[f"split_{split_number}_{MODEL_NAME}"] = split_score
            time_row[f"split_{split_number}_{MODEL_NAME}"] = split_time

        score_rows.append(score_row)
        time_rows.append(time_row)
        print(
            f"did={did} {MODEL_NAME}_auc={score_row[MODEL_NAME]:.6f} "
            f"{MODEL_NAME}_seconds={time_row[MODEL_NAME]:.6f}",
            flush=True,
        )

    write_csv(score_rows, SCORE_CSV)
    write_csv(time_rows, TIME_CSV)
    print(f"Saved scores to {SCORE_CSV}", flush=True)
    print(f"Saved inference times to {TIME_CSV}", flush=True)


if __name__ == "__main__":
    main()
