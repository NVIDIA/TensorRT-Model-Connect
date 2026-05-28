"""Tensor-parallel bundle helpers for single-engine Torch-TRT models."""

from __future__ import annotations

from .bundle_writer import BundleSection
from ...parallel_config import normalize_parallel_config, rank_engine_section


def build_torch_trt_tp_sections(engine_bytes: bytes, *, parallel_config=None) -> list[BundleSection]:
    """Return rank-addressable engine sections for a Torch-TRT TP bundle."""
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("Torch-TRT TP sections require tensor_parallel mode with tp_size > 1")
    return [
        BundleSection(rank_engine_section(rank), engine_bytes)
        for rank in range(parallel.tp_size)
    ]
