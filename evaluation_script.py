import os
import sys

import torch

import pandas as pd
from scipy import stats

NANOTABPFN_IMPORT_PATH = os.path.join(os.path.dirname(__file__), "nanoTabPFN")
if NANOTABPFN_IMPORT_PATH not in sys.path:
    sys.path.insert(0, NANOTABPFN_IMPORT_PATH)

from evaluation_helper import EvalHelper
from model import NanoTabPFNModel
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
    "alternating_hydra_tabpfn_latest",
    "alternating_hydra_tabpfn_epoch_200",
    "alternating_hydra_tabpfn_final",
    "new_looped_transformer_6physical_core4x2",
    "pure_hydra_12_layers_512e",
    "hybrid_8_layers_latest",
    "original_transformer_12l",
    "transformer",
    "hydra",
    "nanotabpfn",
]

ALTERNATING_MODEL_NAME = "tabpfn/models_diff/callback_alternating_hydra_tabpfn_12_layers_512e_lr0p0001_epoch_200.cpkt"
ALTERNATING_HYDRA_TABPFN_LATEST_MODEL_NAME = (
    "tabpfn/models_diff/callback_alternating_hydra_tabpfn_12_layers_512e_lr0p0001_latest.cpkt"
)
ALTERNATING_HYDRA_TABPFN_EPOCH_200_MODEL_NAME = (
    "tabpfn/models_diff/callback_alternating_hydra_tabpfn_12_layers_512e_lr0p0001_epoch_200.cpkt"
)
ALTERNATING_HYDRA_TABPFN_FINAL_MODEL_NAME = (
    "tabpfn/models_diff/alternating_hydra_tabpfn_12_layers_512e_lr0p0001_12l.cpkt"
)
NEW_LOOPED_TRANSFORMER_6PHYSICAL_CORE4X2_MODEL_NAME = (
    "tabpfn/models_diff/new_looped_transformer_6physical_core4x2_10l.cpkt"
)
PURE_HYDRA_12_LAYERS_512E_MODEL_NAME = (
    "tabpfn/models_diff/pure_hydra_12_layers_512e_12l.cpkt"
)
HYBRID_8_LAYERS_LATEST_MODEL_NAME = "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt"
ORIGINAL_TRANSFORMER_12L_MODEL_NAME = "tabpfn/models_diff/callback_original_transformer_12l_latest.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/hydra_small.cpkt"
NANOTABPFN_MODEL_NAME = "nanoTabPFN/nanotabpfn_trained.pt"

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


class NanoTabPFNEvaluationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = None

    def forward(self, src, single_eval_pos):
        _, eval_xs, eval_ys = src
        x = eval_xs.transpose(0, 1).contiguous()
        y = eval_ys[:single_eval_pos].squeeze(-1).transpose(0, 1).contiguous()
        output = self.model((x, y), train_test_split_index=single_eval_pos)
        return output.transpose(0, 1).contiguous()


def load_nanotabpfn_model(model_path=NANOTABPFN_MODEL_NAME):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = NanoTabPFNModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return NanoTabPFNEvaluationWrapper(model), checkpoint.get("evaluation_config", {})


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


def evaluate_hybrid_model(model_path, model_label):
    hybrid_loaded, hybrid_config = load_model_only_inference(
        ".",
        model_path,
        device,
        model_name="hybrid",
    )

    hybrid_model = hybrid_loaded[2]
    print_parameter_count(model_label, hybrid_model)
    return run_model_evaluation(hybrid_model, hybrid_config, method_name="transformer")


def evaluate_looped_transformer_model(model_path, model_label):
    looped_loaded, looped_config = load_model_only_inference(
        ".",
        model_path,
        device,
        model_name="looped_transformer",
    )

    looped_model = looped_loaded[2]
    print_parameter_count(model_label, looped_model)
    return run_model_evaluation(looped_model, looped_config, method_name="transformer")


def evaluate_custom_hydra_model(model_path, model_label):
    hydra_loaded, hydra_config = load_model_only_inference(
        ".",
        model_path,
        device,
        model_name="hydra",
    )

    hydra_model = hydra_loaded[2]
    print_parameter_count(model_label, hydra_model)
    return run_model_evaluation(hydra_model, hydra_config, method_name="hydra")


def evaluate_alternating_hydra_tabpfn_latest_model():
    return evaluate_hybrid_model(
        ALTERNATING_HYDRA_TABPFN_LATEST_MODEL_NAME,
        "alternating_hydra_tabpfn_latest",
    )


def evaluate_alternating_hydra_tabpfn_epoch_200_model():
    return evaluate_hybrid_model(
        ALTERNATING_HYDRA_TABPFN_EPOCH_200_MODEL_NAME,
        "alternating_hydra_tabpfn_epoch_200",
    )


def evaluate_alternating_hydra_tabpfn_final_model():
    return evaluate_hybrid_model(
        ALTERNATING_HYDRA_TABPFN_FINAL_MODEL_NAME,
        "alternating_hydra_tabpfn_final",
    )


def evaluate_new_looped_transformer_6physical_core4x2_model():
    return evaluate_looped_transformer_model(
        NEW_LOOPED_TRANSFORMER_6PHYSICAL_CORE4X2_MODEL_NAME,
        "new_looped_transformer_6physical_core4x2",
    )


def evaluate_pure_hydra_12_layers_512e_model():
    return evaluate_custom_hydra_model(
        PURE_HYDRA_12_LAYERS_512E_MODEL_NAME,
        "pure_hydra_12_layers_512e",
    )


def evaluate_hybrid_8_layers_latest_model():
    return evaluate_hybrid_model(
        HYBRID_8_LAYERS_LATEST_MODEL_NAME,
        "hybrid_8_layers_latest",
    )


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


def evaluate_nanotabpfn_model():
    model, config = load_nanotabpfn_model()
    print_parameter_count("nanotabpfn", model)
    return eval_helper.do_evaluation_custom(
        model,
        bptt=config.get("bptt", 150),
        eval_positions=config.get("eval_positions", [75]),
        metric=METRIC_USED,
        device=device,
        method_name="transformer",
        evaluation_type=EVALUATION_TYPE,
        max_classes=config.get("max_num_classes", 2),
        max_features=config.get("max_num_features", 5),
        split_numbers=SPLIT_NUMBERS,
        jrt_prompt=JRT_PROMPT,
        single_evaluation_prompt=SINGLE_EVAL_PROMPT,
        permutation_bagging=PERMUTATION_BAGGING,
        sample_bagging=SAMPLE_BAGGING,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )


def do_evaluation(eval_list):
    result_dict = {}

    for method_name in eval_list:
        if method_name == "alternating":
            result_dict[method_name] = evaluate_alternating_model()
        elif method_name == "alternating_hydra_tabpfn_latest":
            result_dict[method_name] = evaluate_alternating_hydra_tabpfn_latest_model()
        elif method_name == "alternating_hydra_tabpfn_epoch_200":
            result_dict[method_name] = evaluate_alternating_hydra_tabpfn_epoch_200_model()
        elif method_name == "alternating_hydra_tabpfn_final":
            result_dict[method_name] = evaluate_alternating_hydra_tabpfn_final_model()
        elif method_name == "new_looped_transformer_6physical_core4x2":
            result_dict[method_name] = evaluate_new_looped_transformer_6physical_core4x2_model()
        elif method_name == "pure_hydra_12_layers_512e":
            result_dict[method_name] = evaluate_pure_hydra_12_layers_512e_model()
        elif method_name == "hybrid_8_layers_latest":
            result_dict[method_name] = evaluate_hybrid_8_layers_latest_model()
        elif method_name == "original_transformer_12l":
            result_dict[method_name] = evaluate_original_transformer_12l_model()
        elif method_name == "transformer":
            result_dict[method_name] = evaluate_transformer_model()
        elif method_name == "hydra":
            result_dict[method_name] = evaluate_hydra_model()
        elif method_name == "nanotabpfn":
            result_dict[method_name] = evaluate_nanotabpfn_model()
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
