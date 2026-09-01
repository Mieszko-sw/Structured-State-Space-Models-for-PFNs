"""Run the elastic-looped depth benchmark with a cold CUDA allocator cache.

This entry point reuses ``evaluation_elastic_loopedx12_depth_memory.py`` but
changes the beginning of every timed memory measurement to follow this order:

1. synchronize outstanding CUDA work;
2. release unused blocks held by PyTorch's caching allocator;
3. reset CUDA peak-memory statistics;
4. record the live allocated and reserved baselines.

The model remains loaded on the GPU, so absolute peak allocation still includes
its parameters. The incremental peak column subtracts that live baseline.
"""

import sys

import numpy as np
import torch

import evaluation_elastic_loopedx12_depth_memory as benchmark


DEFAULT_RAW_CSV = (
    "result_csvs/elastic_loopedx12_depth_memory_empty_cache_raw.csv"
)
DEFAULT_SUMMARY_CSV = (
    "result_csvs/elastic_loopedx12_depth_memory_empty_cache_summary.csv"
)


def begin_memory_measurement(device):
    """Start a peak-memory window after clearing unused cached CUDA blocks."""
    if not benchmark.is_cuda(device):
        return {
            "allocated_before": np.nan,
            "reserved_before": np.nan,
        }

    cuda_device = torch.device(device)
    torch.cuda.synchronize(cuda_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_device)
    return {
        "allocated_before": torch.cuda.memory_allocated(cuda_device),
        "reserved_before": torch.cuda.memory_reserved(cuda_device),
    }


original_parse_args = benchmark.parse_args


def parse_args():
    """Use separate output files unless the caller supplies explicit paths."""
    args = original_parse_args()
    command_line = sys.argv[1:]
    if not any(
        argument == "--raw-csv" or argument.startswith("--raw-csv=")
        for argument in command_line
    ):
        args.raw_csv = DEFAULT_RAW_CSV
    if not any(
        argument == "--summary-csv" or argument.startswith("--summary-csv=")
        for argument in command_line
    ):
        args.summary_csv = DEFAULT_SUMMARY_CSV
    return args


if __name__ == "__main__":
    benchmark.begin_memory_measurement = begin_memory_measurement
    benchmark.parse_args = parse_args
    benchmark.main()
