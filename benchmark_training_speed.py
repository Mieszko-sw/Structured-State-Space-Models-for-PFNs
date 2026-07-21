import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

NANOTABPFN_IMPORT_PATH = os.path.join(os.path.dirname(__file__), "nanoTabPFN")
if NANOTABPFN_IMPORT_PATH not in sys.path:
    sys.path.insert(0, NANOTABPFN_IMPORT_PATH)

from model import NanoTabPFNModel
from tabpfn.scripts.model_builder_custom import load_model_only_inference
from tabpfn.scripts.transformer_prediction_interface import (
    load_model_workflow as transformer_load_model_workflow,
)
from tabpfn.utils import torch_nanmean


MODEL_SPECS = {
    "hydra_25m": {
        "display_name": "Hydra 25M",
        "loader": "custom",
        "model_type": "hydra",
        "path": "tabpfn/models_diff/callback_pure_hydra_12_layers_512e_latest.cpkt",
        "seq_len": 1024,
        "num_features": 100,
        "num_classes": 10,
        "batch_size": 2,
        "eval_position": 512,
    },
    "original_tabpfn": {
        "display_name": "Original TabPFN",
        "loader": "transformer_workflow",
        "path": "tabpfn/models_diff/tabpfn_transformer_model.cpkt",
        "seq_len": 1024,
        "num_features": 100,
        "num_classes": 10,
        "batch_size": 2,
        "eval_position": 512,
    },
    "nanotabpfn": {
        "display_name": "NanoTabPFN",
        "loader": "nanotabpfn",
        "path": "nanoTabPFN/nanotabpfn_trained.pt",
        "seq_len": 150,
        "num_features": 5,
        "num_classes": 2,
        "batch_size": 32,
        "eval_position": 75,
    },
    "hybrid_8l": {
        "display_name": "Hybrid 8L",
        "loader": "custom",
        "model_type": "hybrid",
        "path": "tabpfn/models_diff/callback_hybrid_8_layers_latest.cpkt",
        "seq_len": 1024,
        "num_features": 100,
        "num_classes": 10,
        "batch_size": 2,
        "eval_position": 512,
    },
}


RAW_COLUMNS = [
    "model",
    "display_name",
    "checkpoint",
    "run",
    "status",
    "error",
    "device",
    "parameters",
    "seq_len",
    "num_features",
    "num_classes",
    "batch_size",
    "eval_position",
    "train_tokens",
    "test_tokens",
    "loss",
    "step_seconds",
    "examples_per_second",
    "tokens_per_second",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure synthetic training-step speed for selected TabPFN-family models."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pf-mini-batch-size", type=int, default=None)
    parser.add_argument("--pf-seq-len", type=int, default=None)
    parser.add_argument("--pf-eval-position", type=int, default=None)
    parser.add_argument("--nano-batch-size", type=int, default=None)
    parser.add_argument("--raw-csv", default=os.path.join("result_csvs", "training_speed_raw.csv"))
    parser.add_argument("--summary-csv", default=os.path.join("result_csvs", "training_speed_summary.csv"))
    return parser.parse_args()


def normalize_device(device):
    device = str(device)
    if device.isdigit():
        device = f"cuda:{device}"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    return device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu"


def synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def clear_device_cache(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def load_pf_model(name, spec, device):
    if spec["loader"] == "custom":
        loaded, config = load_model_only_inference(
            ".",
            spec["path"],
            device,
            model_name=spec["model_type"],
        )
    elif spec["loader"] == "transformer_workflow":
        loaded, config, _ = transformer_load_model_workflow(
            2,
            -1,
            add_name="",
            base_path="",
            device=device,
            eval_addition="",
            only_inference=True,
            model_path_custom=spec["path"],
        )
    else:
        raise ValueError(f"{name} is not a PFN model")

    model = loaded[2].to(device)
    model.train()
    return model, config


def load_nano_model(spec, device):
    checkpoint = torch.load(spec["path"], map_location="cpu")
    model = NanoTabPFNModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.train()
    return model, checkpoint.get("evaluation_config", {})


def make_pf_batch(spec, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        spec["seq_len"],
        spec["batch_size"],
        spec["num_features"],
        generator=generator,
        device="cpu",
    ).to(device)
    y = torch.randint(
        0,
        spec["num_classes"],
        (spec["seq_len"], spec["batch_size"]),
        generator=generator,
        device="cpu",
    ).float().to(device)
    targets = y.long()
    return (None, x, y), targets


def make_nano_batch(spec, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(
        spec["batch_size"],
        spec["seq_len"],
        spec["num_features"],
        generator=generator,
        device="cpu",
    ).to(device)
    y = torch.randint(
        0,
        spec["num_classes"],
        (spec["batch_size"], spec["seq_len"]),
        generator=generator,
        device="cpu",
    ).to(device)
    return x, y


def pf_train_step(model, optimizer, criterion, batch, eval_position, device):
    data, targets = batch
    optimizer.zero_grad(set_to_none=True)
    output = model(data, single_eval_pos=eval_position)
    targets = targets[eval_position:]
    losses = criterion(output.reshape(-1, output.shape[-1]), targets.to(device).flatten())
    losses = losses.view(*output.shape[:2])
    loss = torch_nanmean(losses.mean(0))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def nano_train_step(model, optimizer, criterion, batch, eval_position):
    x, y = batch
    optimizer.zero_grad(set_to_none=True)
    output = model((x, y[:, :eval_position].float()), train_test_split_index=eval_position)
    targets = y[:, eval_position:].reshape(-1).long()
    loss = criterion(output.reshape(-1, output.shape[-1]), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def apply_overrides(spec, args):
    spec = dict(spec)
    if spec["loader"] == "nanotabpfn":
        if args.nano_batch_size is not None:
            spec["batch_size"] = args.nano_batch_size
    else:
        if args.pf_mini_batch_size is not None:
            spec["batch_size"] = args.pf_mini_batch_size
        if args.pf_seq_len is not None:
            spec["seq_len"] = args.pf_seq_len
        if args.pf_eval_position is not None:
            spec["eval_position"] = args.pf_eval_position
        if spec["eval_position"] >= spec["seq_len"]:
            raise ValueError(f"eval_position must be smaller than seq_len for {spec['display_name']}")
    return spec


def make_summary(raw_df):
    ok = raw_df[raw_df["status"] == "ok"]
    return (
        ok.groupby(
            [
                "model",
                "display_name",
                "parameters",
                "seq_len",
                "num_features",
                "num_classes",
                "batch_size",
                "eval_position",
                "train_tokens",
                "test_tokens",
            ]
        )[["step_seconds", "examples_per_second", "tokens_per_second"]]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )


def write_csv(df, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    args.device = normalize_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = []
    for model_name in args.models:
        spec = apply_overrides(MODEL_SPECS[model_name], args)
        print(f"Loading {spec['display_name']} from {spec['path']}", flush=True)
        try:
            if spec["loader"] == "nanotabpfn":
                model, _ = load_nano_model(spec, args.device)
                batch = make_nano_batch(spec, args.device, args.seed)
                criterion = torch.nn.CrossEntropyLoss()
                step_fn = lambda: nano_train_step(model, optimizer, criterion, batch, spec["eval_position"])
            else:
                model, _ = load_pf_model(model_name, spec, args.device)
                batch = make_pf_batch(spec, args.device, args.seed)
                criterion = torch.nn.CrossEntropyLoss(reduction="none")
                step_fn = lambda: pf_train_step(model, optimizer, criterion, batch, spec["eval_position"], args.device)

            params = parameter_count(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
            print(
                f"{spec['display_name']}: {params:,} params, "
                f"seq_len={spec['seq_len']} batch={spec['batch_size']} features={spec['num_features']}",
                flush=True,
            )

            for warmup_idx in range(args.warmup_steps):
                step_fn()
                synchronize(args.device)
                print(f"warmup model={model_name} step={warmup_idx + 1}", flush=True)

            for run_idx in range(args.timed_steps):
                synchronize(args.device)
                start = time.perf_counter()
                loss = step_fn()
                synchronize(args.device)
                elapsed = time.perf_counter() - start
                train_tokens = spec["eval_position"] * spec["batch_size"]
                test_tokens = (spec["seq_len"] - spec["eval_position"]) * spec["batch_size"]
                row = {
                    "model": model_name,
                    "display_name": spec["display_name"],
                    "checkpoint": spec["path"],
                    "run": run_idx + 1,
                    "status": "ok",
                    "error": "",
                    "device": args.device,
                    "parameters": params,
                    "seq_len": spec["seq_len"],
                    "num_features": spec["num_features"],
                    "num_classes": spec["num_classes"],
                    "batch_size": spec["batch_size"],
                    "eval_position": spec["eval_position"],
                    "train_tokens": train_tokens,
                    "test_tokens": test_tokens,
                    "loss": loss,
                    "step_seconds": elapsed,
                    "examples_per_second": spec["batch_size"] / elapsed,
                    "tokens_per_second": (spec["seq_len"] * spec["batch_size"]) / elapsed,
                }
                rows.append(row)
                print(
                    f"timed model={model_name} run={run_idx + 1} "
                    f"seconds={elapsed:.6f} loss={loss:.6f}",
                    flush=True,
                )
        except RuntimeError as error:
            clear_device_cache(args.device)
            rows.append(
                {
                    "model": model_name,
                    "display_name": spec["display_name"],
                    "checkpoint": spec["path"],
                    "run": 0,
                    "status": "failed",
                    "error": str(error).splitlines()[0],
                    "device": args.device,
                    "parameters": np.nan,
                    "seq_len": spec["seq_len"],
                    "num_features": spec["num_features"],
                    "num_classes": spec["num_classes"],
                    "batch_size": spec["batch_size"],
                    "eval_position": spec["eval_position"],
                    "train_tokens": spec["eval_position"] * spec["batch_size"],
                    "test_tokens": (spec["seq_len"] - spec["eval_position"]) * spec["batch_size"],
                    "loss": np.nan,
                    "step_seconds": np.nan,
                    "examples_per_second": np.nan,
                    "tokens_per_second": np.nan,
                }
            )
            print(f"failed model={model_name} error={str(error).splitlines()[0]}", flush=True)

        del model
        clear_device_cache(args.device)

    raw_df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    summary_df = make_summary(raw_df)
    write_csv(raw_df, args.raw_csv)
    write_csv(summary_df, args.summary_csv)

    print("\nTraining step speed summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw timings to {args.raw_csv}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
