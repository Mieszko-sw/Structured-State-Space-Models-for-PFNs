import argparse
import os

import pandas as pd
import torch

from evaluation_helper import EvalHelper
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.hydra_prediction_interface import (
    load_model_workflow as hydra_load_model_workflow,
)
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)


MODELS = {
    "tabpfn": {
        "path": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
        "loader_type": "tabpfn",
        "method_name": "transformer",
    },
    "hybrid_8l": {
        "path": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
        "loader_type": "hybrid",
        "method_name": "transformer",
    },
    "hydra_160m": {
        "path": "tabpfn/models_diff/hydra_small.cpkt",
        "loader_type": "hydra_workflow",
        "method_name": "hydra",
    },
    "hydra_22m": {
        "path": "tabpfn/models_diff/callback_pure_hydra_12_layers_512e_latest.cpkt",
        "loader_type": "hydra",
        "method_name": "hydra",
    },
}

EVALUATION_TYPE = "openmlcc18"
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}
SPLIT_NUMBERS = [1, 2, 3, 4, 5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect complete outputs of four models on OpenML DID 46937."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--did", type=int, default=46937)
    parser.add_argument(
        "--outputs-csv",
        default=os.path.join(
            "result_csvs", "did_46937_four_models_outputs.csv"
        ),
    )
    parser.add_argument(
        "--time-csv",
        default=os.path.join(
            "result_csvs", "did_46937_four_models_times.csv"
        ),
    )
    parser.add_argument(
        "--score-csv",
        default=os.path.join(
            "result_csvs", "did_46937_four_models_scores.csv"
        ),
    )
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    return device


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


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def load_model(model_name, device):
    info = MODELS[model_name]
    if info["loader_type"] == "tabpfn":
        loaded, config, _ = transformer_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=info["path"],
        )
    elif info["loader_type"] == "hydra_workflow":
        loaded, config, _ = hydra_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=info["path"],
        )
    else:
        loaded, config = load_model_only_inference(
            ".",
            info["path"],
            device,
            model_name=info["loader_type"],
        )

    model = loaded[2]
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {params:,} parameters from {info['path']}", flush=True)
    return model, config


def warmup_model(model, config, model_name, device, runs):
    if runs <= 0:
        return

    eval_helper = EvalHelper()
    for run_idx in range(runs):
        synchronize(device)
        eval_helper.do_evaluation_custom(
            model,
            bptt=32768,
            eval_positions=[16384],
            metric=tabular_metrics.auc_metric,
            device=device,
            method_name=MODELS[model_name]["method_name"],
            evaluation_type="dummy",
            split_numbers=[1],
            jrt_prompt=False,
            single_evaluation_prompt=False,
            permutation_bagging=1,
            sample_bagging=0,
            return_whole_output=True,
        )
        synchronize(device)
        print(f"warmup model={model_name} run={run_idx + 1}", flush=True)


def evaluate_model(model_name, device, warmup_runs, did):
    model, config = load_model(model_name, device)
    warmup_model(model, config, model_name, device, warmup_runs)
    eval_helper = EvalHelper()
    synchronize(device)
    return eval_helper.do_evaluation_custom(
        model,
        bptt=32768,
        eval_positions=[16384],
        metric=tabular_metrics.auc_metric,
        device=device,
        method_name=MODELS[model_name]["method_name"],
        evaluation_type=did,
        split_numbers=SPLIT_NUMBERS,
        jrt_prompt=False,
        single_evaluation_prompt=False,
        permutation_bagging=1,
        sample_bagging=0,
        eval_filters=EVALUATION_TYPE_FILTERS,
        return_whole_output=True,
    )


def extract_prediction_rows(model_name, split_number, split_result):
    output_key = next(key for key in split_result if "_outputs_at_" in key)
    target_key = next(key for key in split_result if "_ys_at_" in key)
    outputs = torch.as_tensor(split_result[output_key]).detach().cpu()
    targets = torch.as_tensor(split_result[target_key]).detach().cpu()

    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)

    rows = []
    for ensemble_index in range(outputs.shape[0]):
        for test_index in range(outputs.shape[1]):
            values = outputs[ensemble_index, test_index]
            row = {
                "did": 46937,
                "model": model_name,
                "split_number": split_number,
                "ensemble_index": ensemble_index,
                "test_index": test_index,
                "target": int(targets[ensemble_index, test_index].item()),
                "predicted_class": int(values.argmax().item()),
            }
            for class_index, value in enumerate(values.tolist()):
                row[f"output_class_{class_index}"] = float(value)
            rows.append(row)
    return rows


def write_csv(rows, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def mean_inference_by_model(time_rows):
    return {
        model_name: mean([row[model_name] for row in time_rows])
        for model_name in MODELS
    }


def main():
    args = parse_args()
    device = normalize_device(args.device)

    model_results = {}
    for model_name in MODELS:
        print(f"Evaluating {model_name} on did={args.did}...", flush=True)
        model_results[model_name] = evaluate_model(
            model_name,
            device,
            args.warmup_runs,
            args.did,
        )
        synchronize(device)

    dids = sorted(next(iter(model_results.values())).keys())
    time_rows = []
    score_rows = []
    output_rows = []

    for did in dids:
        time_row = {"did": did}
        score_row = {"did": did}
        for model_name, result in model_results.items():
            split_results = result[did]
            split_times = [extract_inference_time(x) for x in split_results]
            split_scores = [extract_metric(x) for x in split_results]
            for split_number, split_result in zip(SPLIT_NUMBERS, split_results):
                output_rows.extend(
                    extract_prediction_rows(model_name, split_number, split_result)
                )

            time_row[model_name] = mean(split_times)
            score_row[model_name] = mean(split_scores)
            for split_number, split_time, split_score in zip(
                SPLIT_NUMBERS, split_times, split_scores
            ):
                time_row[f"split_{split_number}_{model_name}"] = split_time
                score_row[f"split_{split_number}_{model_name}"] = split_score
                print(
                    f"benchmark did={did} split={split_number} "
                    f"model={model_name} inference_seconds={split_time:.6f}",
                    flush=True,
                )

        time_rows.append(time_row)
        score_rows.append(score_row)
        print(
            "did={did} ".format(did=did)
            + " ".join(f"{name}_seconds={time_row[name]:.6f}" for name in MODELS),
            flush=True,
        )

    model_mean_times = mean_inference_by_model(time_rows)
    print(
        "mean_across_benchmarks "
        + " ".join(
            f"{name}_seconds={model_mean_times[name]:.6f}" for name in MODELS
        ),
        flush=True,
    )

    write_csv(output_rows, args.outputs_csv)
    write_csv(time_rows, args.time_csv)
    write_csv(score_rows, args.score_csv)
    print(f"Saved complete outputs to {args.outputs_csv}", flush=True)
    print(f"Saved inference times to {args.time_csv}", flush=True)
    print(f"Saved scores to {args.score_csv}", flush=True)


if __name__ == "__main__":
    main()
