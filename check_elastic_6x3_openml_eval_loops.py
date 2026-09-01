"""Measure the loop depth actually used by elastic 6x3 during OpenML evaluation.

The script follows the same dataset preparation and ``evaluate`` call as
``evaluation_per_dataset_speed_synchronized_elastic_looped_vs_tabpfn.py``.
Forward hooks count executions of the three physical core blocks (A, B, C),
then write one row per OpenML dataset to a CSV file.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

import evaluation_per_dataset_speed_synchronized as benchmark
from evaluation_helper import EvalHelper


MODEL_NAME = "elastic_looped_3physical_core6x"
MODEL_INFO = {
    "path": (
        "tabpfn/models_diff/"
        "elastic_looped_transformer_3physical_core6x_18l.cpkt"
    ),
    "loader_type": "looped_transformer",
    "method_name": "transformer",
}

OUTPUT_COLUMNS = [
    "did",
    "dataset_name",
    "num_samples",
    "num_features",
    "status",
    "error",
    "model_forward_calls",
    "block_a_calls",
    "block_b_calls",
    "block_c_calls",
    "block_a_loops_per_forward",
    "block_b_loops_per_forward",
    "block_c_loops_per_forward",
    "complete_stack_loops_per_forward",
    "same_loop_count_for_all_blocks",
    "configured_core_repeats",
    "configured_effective_layers",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dids",
        nargs="+",
        type=int,
        default=None,
        help="OpenML dataset IDs. Default: the project's small CC18 list.",
    )
    parser.add_argument(
        "--split-number",
        type=int,
        default=1,
        choices=benchmark.SPLIT_NUMBERS,
        help="One split is enough to observe the runtime loop depth.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(
            "result_csvs", "elastic_6x3_openml_evaluation_loop_counts.csv"
        ),
    )
    return parser.parse_args()


class LoopExecutionCounter:
    """Count encoder forwards and executions of physical blocks A, B, and C."""

    def __init__(self, model):
        encoder = getattr(model, "transformer_encoder", None)
        if encoder is None or not hasattr(encoder, "core_layers"):
            raise TypeError("Loaded model does not have a looped Transformer encoder.")
        if encoder.core_layers != 3 or len(encoder.layers) != 3:
            raise ValueError(
                "Expected exactly three physical core blocks and no edge blocks; "
                f"got core_layers={encoder.core_layers}, physical_layers={len(encoder.layers)}."
            )

        self.encoder = encoder
        self.forward_calls = 0
        self.block_calls = [0, 0, 0]
        self.handles = [encoder.register_forward_pre_hook(self._count_forward)]
        self.handles.extend(
            block.register_forward_hook(self._make_block_hook(index))
            for index, block in enumerate(encoder.layers)
        )

    def _count_forward(self, _module, _inputs):
        self.forward_calls += 1

    def _make_block_hook(self, index):
        def count_block(_module, _inputs, _output):
            self.block_calls[index] += 1

        return count_block

    def snapshot(self):
        return self.forward_calls, tuple(self.block_calls)

    def delta(self, before):
        before_forwards, before_blocks = before
        return (
            self.forward_calls - before_forwards,
            tuple(
                after - previous
                for after, previous in zip(self.block_calls, before_blocks)
            ),
        )

    def close(self):
        for handle in self.handles:
            handle.remove()


def exact_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    ratio = numerator / denominator
    return int(ratio) if ratio.is_integer() else ratio


def make_row(did, dataset_list, counter, before, status="ok", error=""):
    dataset_name, x, _, _, _, _ = dataset_list[0]
    forward_calls, block_calls = counter.delta(before)
    loops = [exact_ratio(calls, forward_calls) for calls in block_calls]
    finite_loops = [value for value in loops if not pd.isna(value)]
    same_loop_count = len(finite_loops) == 3 and len(set(finite_loops)) == 1
    complete_stack_loops = finite_loops[0] if same_loop_count else np.nan

    return {
        "did": did,
        "dataset_name": dataset_name,
        "num_samples": int(x.shape[0]),
        "num_features": int(x.shape[1]),
        "status": status,
        "error": error,
        "model_forward_calls": forward_calls,
        "block_a_calls": block_calls[0],
        "block_b_calls": block_calls[1],
        "block_c_calls": block_calls[2],
        "block_a_loops_per_forward": loops[0],
        "block_b_loops_per_forward": loops[1],
        "block_c_loops_per_forward": loops[2],
        "complete_stack_loops_per_forward": complete_stack_loops,
        "same_loop_count_for_all_blocks": same_loop_count,
        "configured_core_repeats": counter.encoder.core_repeats,
        "configured_effective_layers": counter.encoder.num_layers,
    }


def main():
    args = parse_args()
    args.device = benchmark.normalize_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Restrict the shared evaluation helper to the elastic model.
    benchmark.MODELS = {MODEL_NAME: MODEL_INFO}

    eval_helper = EvalHelper()
    dids = args.dids if args.dids is not None else eval_helper.openml_cc18_dids_small
    datasets = benchmark.prepare_datasets(dids)
    model, config = benchmark.load_model(MODEL_NAME, args.device)
    counter = LoopExecutionCounter(model)
    rows = []

    try:
        for did, dataset_list in datasets.items():
            before = counter.snapshot()
            try:
                with torch.no_grad():
                    benchmark.run_single_measurement(
                        model,
                        MODEL_NAME,
                        config,
                        dataset_list,
                        args.device,
                        args.split_number,
                    )
                row = make_row(did, dataset_list, counter, before)
            except Exception as exc:
                row = make_row(
                    did,
                    dataset_list,
                    counter,
                    before,
                    status="error",
                    error=benchmark.format_error(exc),
                )
                benchmark.clear_device_cache(args.device)

            rows.append(row)
            print(
                f"did={did} dataset={row['dataset_name']} status={row['status']} "
                f"forwards={row['model_forward_calls']} "
                f"loops(A,B,C)=({row['block_a_loops_per_forward']}, "
                f"{row['block_b_loops_per_forward']}, "
                f"{row['block_c_loops_per_forward']})",
                flush=True,
            )
    finally:
        counter.close()

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    print(f"\nSaved loop-count report to {args.output_csv}")
    if not result.empty:
        print(
            result[
                [
                    "did",
                    "dataset_name",
                    "status",
                    "block_a_loops_per_forward",
                    "block_b_loops_per_forward",
                    "block_c_loops_per_forward",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
