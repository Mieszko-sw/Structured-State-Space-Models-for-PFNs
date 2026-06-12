import os

import pandas as pd
from scipy import stats

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


# EVALUATION_TYPE = "openmlcc18_large"
EVALUATION_TYPE = "openmlcc18"

# True means to keep matching datasets, False means to omit them.
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}

EVALUATION_METHODS = [
    "alternating",
    "original_transformer_12l",
    "transformer",
    "hydra",
]

ALTERNATING_MODEL_NAME = "tabpfn/models_diff/callback_hybrid_6hydra_6transformer_epoch_200.cpkt"
ORIGINAL_TRANSFORMER_12L_MODEL_NAME = "tabpfn/models_diff/callback_original_transformer_12l_latest.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/hydra_small.cpkt"

METRIC_USED = tabular_metrics.auc_metric
RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "alternating_hybrid_eval.csv")
TIME_CSV_SAVE_DIR = os.path.join("result_csvs", "alternating_hybrid_eval_inference_time.csv")

SPLIT_NUMBERS = [1, 2, 3, 4, 5]

bptt_here = 1024
CONFIDENCE_LEVEL = 0.95

JRT_PROMPT = False
SINGLE_EVAL_PROMPT = False
PERMUTATION_BAGGING = 1
SAMPLE_BAGGING = 0

device = "cuda:0"

eval_helper = EvalHelper()


def print_parameter_count(model_name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} parameters: {total_params:,} total ({trainable_params:,} trainable)")


def calc_moe(data):
    sem = stats.sem(data)
    degrees_of_freedom = len(data) - 1
    t_score = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, degrees_of_freedom)
    return t_score * sem


def run_model_evaluation(model, config, method_name):
    return eval_helper.do_evaluation_custom(
        model,
        bptt=bptt_here,
        eval_positions=config["eval_positions"],
        metric=METRIC_USED,
        device=device,
        method_name=method_name,
        evaluation_type=EVALUATION_TYPE,
        split_numbers=SPLIT_NUMBERS,
        jrt_prompt=JRT_PROMPT,
        single_evaluation_prompt=SINGLE_EVAL_PROMPT,
        permutation_bagging=PERMUTATION_BAGGING,
        sample_bagging=SAMPLE_BAGGING,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )


def evaluate_alternating_model():
    hybrid_loaded, hybrid_config = load_model_only_inference(
        ".",
        ALTERNATING_MODEL_NAME,
        device,
        model_name="hybrid",
    )

    hybrid_model = hybrid_loaded[2]
    print_parameter_count("alternating", hybrid_model)
    return run_model_evaluation(hybrid_model, hybrid_config, method_name="transformer")


def evaluate_transformer_model():
    transformer_loaded, transformer_config, _ = transformer_load_model_workflow(
        2,
        -1,
        add_name="",
        base_path="",
        device=device,
        eval_addition="",
        only_inference=True,
        model_path_custom=TRANSFORMER_MODEL_NAME,
    )

    transformer_model = transformer_loaded[2]
    print_parameter_count("transformer", transformer_model)
    return run_model_evaluation(transformer_model, transformer_config, method_name="transformer")


def evaluate_original_transformer_12l_model():
    transformer_loaded, transformer_config = load_model_only_inference(
        ".",
        ORIGINAL_TRANSFORMER_12L_MODEL_NAME,
        device,
        model_name="transformer",
    )

    transformer_model = transformer_loaded[2]
    print_parameter_count("original_transformer_12l", transformer_model)
    return run_model_evaluation(transformer_model, transformer_config, method_name="transformer")


def evaluate_hydra_model():
    hydra_loaded, hydra_config, _ = hydra_load_model_workflow(
        2,
        -1,
        add_name="",
        base_path="",
        device=device,
        eval_addition="",
        only_inference=True,
        model_path_custom=HYDRA_MODEL_NAME,
    )

    hydra_model = hydra_loaded[2]
    print_parameter_count("hydra", hydra_model)
    return run_model_evaluation(hydra_model, hydra_config, method_name="hydra")


def do_evaluation(eval_list):
    result_dict = {}

    for method_name in eval_list:
        if method_name == "alternating":
            result_dict[method_name] = evaluate_alternating_model()
        elif method_name == "original_transformer_12l":
            result_dict[method_name] = evaluate_original_transformer_12l_model()
        elif method_name == "transformer":
            result_dict[method_name] = evaluate_transformer_model()
        elif method_name == "hydra":
            result_dict[method_name] = evaluate_hydra_model()
        else:
            raise ValueError(f"Unknown evaluation method: {method_name}")

    return result_dict

def print_summary_stats(result_dict):
    for method in EVALUATION_METHODS:
        split_means = []

        for split in range(len(SPLIT_NUMBERS)):
            vals = result_dict[method].values()
            split_errs = [extract_metric(x[split]) for x in vals]
            split_means.append(sum(split_errs) / len(split_errs))

        print(f"{method} Stats: ")
        print(f"Split Means: {split_means}")
        print(f"Mean Overall: {sum(split_means) / len(split_means)}")
        print(f"MOE: {calc_moe(split_means)}")


def print_timing_stats(result_dict):
    for method in EVALUATION_METHODS:
        split_means = []

        for split in range(len(SPLIT_NUMBERS)):
            vals = result_dict[method].values()
            split_times = [extract_inference_time(x[split]) for x in vals]
            split_means.append(sum(split_times) / len(split_times))

        print(f"{method} Inference Time Stats: ")
        print(f"Split Means Seconds: {split_means}")
        print(f"Mean Overall Seconds: {sum(split_means) / len(split_means)}")
        print(f"MOE Seconds: {calc_moe(split_means)}")


def extract_metric(split_result):
    metric = split_result["mean_metric"]
    if hasattr(metric, "item"):
        return metric.item()
    return float(metric)


def extract_inference_time(split_result):
    time_keys = tuple(f"_time_at_{pos}" for pos in split_result["eval_positions"])
    time_values = [
        value
        for key, value in split_result.items()
        if key.endswith(time_keys)
    ]
    return sum(time_values)


def save_results_csv(result_dict):
    header = ["did"] + EVALUATION_METHODS
    result_arr = []
    keys = list(result_dict[list(result_dict.keys())[0]].keys())

    for key in keys:
        to_add = [key]

        for method in EVALUATION_METHODS:
            res = result_dict[method][key]
            to_add.append(sum(extract_metric(x) for x in res) / len(res))

        result_arr.append(to_add)

    os.makedirs(os.path.dirname(RESULT_CSV_SAVE_DIR), exist_ok=True)
    df_out = pd.DataFrame(result_arr, columns=header)
    df_out.to_csv(RESULT_CSV_SAVE_DIR, index=False)


def save_timing_csv(result_dict):
    header = ["did"] + EVALUATION_METHODS
    result_arr = []
    keys = list(result_dict[list(result_dict.keys())[0]].keys())

    for key in keys:
        to_add = [key]

        for method in EVALUATION_METHODS:
            res = result_dict[method][key]
            to_add.append(sum(extract_inference_time(x) for x in res) / len(res))

        result_arr.append(to_add)

    os.makedirs(os.path.dirname(TIME_CSV_SAVE_DIR), exist_ok=True)
    df_out = pd.DataFrame(result_arr, columns=header)
    df_out.to_csv(TIME_CSV_SAVE_DIR, index=False)


if __name__ == "__main__":
    results = do_evaluation(EVALUATION_METHODS)
    print_summary_stats(results)
    print_timing_stats(results)
    save_results_csv(results)
    save_timing_csv(results)
    print(f"Saved results to {RESULT_CSV_SAVE_DIR}")
    print(f"Saved inference times to {TIME_CSV_SAVE_DIR}")
