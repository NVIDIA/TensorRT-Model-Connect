# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build fixed TensorRT wrappers for SAM3's PyTorch hard-mask resize packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HARD_MASK_RESIZE_SECTION = "sam3_hard_mask_resize_engine_plan"
HARD_MASK_RESIZE_BATCH2_SECTION = "sam3_hard_mask_resize_batch2_engine_plan"
_GLOBAL_PATTERN = re.compile(
    r"trtmc\.sam3\.tracker_memory\.resize\.b([12])\.fixed\.([0-9a-f]{20})\Z"
)


@dataclass(frozen=True)
class HardMaskResizePlanSpec:
    """Exporter-to-plan handoff for fixed B1/B2 resize packages."""

    plugin_library: Path
    global_name_b1: str
    global_name_b2: str


def _validate_global_name(global_name: str, *, batch_size: int) -> None:
    match = _GLOBAL_PATTERN.fullmatch(global_name)
    if match is None or int(match.group(1)) != batch_size:
        raise ValueError(
            "SAM3 hard-mask resize global must identify the matching "
            f"content-addressed B{batch_size} AOTI package"
        )


def _build_hard_mask_resize_plan(
    global_name: str,
    *,
    batch_size: int,
    verbose: bool,
) -> bytes:
    from . import tracker_builder
    from .tracker_memory_ffi_builder import _create_plugin

    _validate_global_name(global_name, batch_size=batch_size)
    trt, builder, network, config = tracker_builder._new_network(
        enable_tf32=False,
        verbose=verbose,
    )
    tracker_mask = tracker_builder._input(
        network,
        "tracker_mask",
        trt.float32,
        (batch_size, 1, 288, 288),
    )

    # The shared memory plugin keeps a four-position TensorRT ABI. Resize
    # globals consume only position one; scalar constants make the unused
    # positions explicit without embedding a multi-megabyte feature tensor.
    dummy_feature = network.add_constant((1,), np.zeros((1,), dtype=np.float32))
    dummy_score = network.add_constant((1,), np.zeros((1,), dtype=np.float32))
    dummy_suppression = network.add_constant((1,), np.zeros((1,), dtype=np.int32))
    if dummy_feature is None or dummy_score is None or dummy_suppression is None:
        raise RuntimeError("Could not add SAM3 hard-mask resize ABI constants")
    layer = network.add_plugin_v2(
        [
            dummy_feature.get_output(0),
            tracker_mask,
            dummy_score.get_output(0),
            dummy_suppression.get_output(0),
        ],
        _create_plugin(trt, global_name, batch_size=batch_size),
    )
    if layer is None or layer.num_outputs != 1:
        raise RuntimeError("Could not add the SAM3 hard-mask resize TVM-FFI layer")
    tracker_builder._mark(network, layer.get_output(0), "resized_tracker_mask")
    return tracker_builder._serialize(
        builder,
        network,
        config,
        kind=f"hard-mask resize B{batch_size} TVM-FFI",
        verbose=verbose,
    )


def build_sam3_hard_mask_resize_ffi_plans(
    spec: HardMaskResizePlanSpec,
    *,
    verbose: bool = False,
) -> dict[str, bytes]:
    """Build exact fixed B1/B2 hard-mask resize wrapper plans."""

    _validate_global_name(spec.global_name_b1, batch_size=1)
    _validate_global_name(spec.global_name_b2, batch_size=2)
    from .native_plugin_builder import load_native_plugin

    load_native_plugin(spec.plugin_library)
    return {
        HARD_MASK_RESIZE_SECTION: _build_hard_mask_resize_plan(
            spec.global_name_b1, batch_size=1, verbose=verbose
        ),
        HARD_MASK_RESIZE_BATCH2_SECTION: _build_hard_mask_resize_plan(
            spec.global_name_b2, batch_size=2, verbose=verbose
        ),
    }


__all__ = [
    "HARD_MASK_RESIZE_BATCH2_SECTION",
    "HARD_MASK_RESIZE_SECTION",
    "HardMaskResizePlanSpec",
    "build_sam3_hard_mask_resize_ffi_plans",
]
