# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build SAM3 tracker-memory plans around its model-owned TVM-FFI plugin."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRACKER_MEMORY_SECTION = "sam3_tracker_memory_engine_plan"
TRACKER_MEMORY_BATCH2_SECTION = "sam3_tracker_memory_batch2_engine_plan"
TRACKER_HARD_MEMORY_SECTION = "sam3_tracker_hard_memory_engine_plan"
TRACKER_HARD_MEMORY_BATCH2_SECTION = "sam3_tracker_hard_memory_batch2_engine_plan"

_SPATIAL_TOKENS = 72 * 72
_MEMORY_CHANNELS = 64
_PLUGIN_NAME = "Sam3TrackerMemoryFfi"
_PLUGIN_VERSION = "2"
_GLOBAL_PATTERN = re.compile(
    r"trtmc\.sam3\.tracker_memory\.(soft|hard)\.b([12])\.fixed\.([0-9a-f]{20})\Z"
)


@dataclass(frozen=True)
class TrackerMemoryPlanSpec:
    """Exporter-to-plan handoff for the four fixed SAM3 memory packages."""

    plugin_library: Path
    soft_global_b1: str
    soft_global_b2: str
    hard_global_b1: str
    hard_global_b2: str


def _validate_global_name(
    global_name: str,
    *,
    batch_size: int,
    hard_mask: bool,
) -> None:
    match = _GLOBAL_PATTERN.fullmatch(global_name)
    expected_policy = "hard" if hard_mask else "soft"
    if match is None or match.group(1) != expected_policy or int(match.group(2)) != batch_size:
        raise ValueError(
            "SAM3 tracker-memory global must identify the matching content-addressed "
            f"{expected_policy} B{batch_size} fixed AOTI package"
        )


def _plugin_creator(trt):
    registry = trt.get_plugin_registry()
    creator = registry.get_creator(_PLUGIN_NAME, _PLUGIN_VERSION, "")
    if creator is None:
        raise RuntimeError(
            "SAM3 tracker-memory TensorRT plugin is not registered after loading its native DSO"
        )
    return creator


def _create_plugin(trt, global_name: str, *, batch_size: int):
    kernel_name = global_name.encode("utf-8")
    batch_value = np.asarray([batch_size], dtype=np.int32)
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField("kernel_name", kernel_name, trt.PluginFieldType.CHAR),
            trt.PluginField("batch_size", batch_value, trt.PluginFieldType.INT32),
        ]
    )
    plugin = _plugin_creator(trt).create_plugin("sam3_tracker_memory_ffi", fields)
    if plugin is None:
        raise RuntimeError("Could not create the SAM3 tracker-memory TensorRT plugin")
    return plugin


def _reshape_output(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("Could not reshape a SAM3 tracker-memory output")
    layer.reshape_dims = shape
    return layer.get_output(0)


def _slice_plane(network, packed, *, batch_size: int, plane: int):
    if batch_size == 1:
        slice_shape = (1, _SPATIAL_TOKENS, 1, _MEMORY_CHANNELS)
        output_shape = (_SPATIAL_TOKENS, 1, _MEMORY_CHANNELS)
    elif batch_size == 2:
        slice_shape = (1, 2, _SPATIAL_TOKENS, _MEMORY_CHANNELS)
        output_shape = (2, _SPATIAL_TOKENS, _MEMORY_CHANNELS)
    else:
        raise ValueError("SAM3 tracker-memory plans support only B1 and B2")
    layer = network.add_slice(
        packed,
        (plane, 0, 0, 0),
        slice_shape,
        (1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError("Could not slice a SAM3 tracker-memory packed output")
    return _reshape_output(network, layer.get_output(0), output_shape)


def _add_output_contract(network, packed, *, batch_size: int) -> None:
    from .tracker_builder import _mark

    _mark(
        network,
        _slice_plane(network, packed, batch_size=batch_size, plane=0),
        "new_memory_features",
    )
    _mark(
        network,
        _slice_plane(network, packed, batch_size=batch_size, plane=1),
        "new_memory_position",
    )


def _constant_zero_suppression(network, *, batch_size: int):
    layer = network.add_constant(
        (batch_size, 1),
        np.zeros((batch_size, 1), dtype=np.int32),
    )
    if layer is None:
        raise RuntimeError("Could not add SAM3 hard-memory suppression constant")
    return layer.get_output(0)


def _build_tracker_memory_ffi_plan(
    global_name: str,
    *,
    batch_size: int,
    hard_mask: bool,
    verbose: bool,
) -> bytes:
    from . import tracker_builder

    _validate_global_name(global_name, batch_size=batch_size, hard_mask=hard_mask)
    trt, builder, network, config = tracker_builder._new_network(
        enable_tf32=False,
        verbose=verbose,
    )
    feature = tracker_builder._input(
        network,
        "tracker_feature_2",
        trt.float32,
        (1, 256, 72, 72),
    )
    mask_name = "owned_tracker_mask" if hard_mask else "final_mask"
    mask_size = 1008 if hard_mask else 288
    mask = tracker_builder._input(
        network,
        mask_name,
        trt.float32,
        (batch_size, 1, mask_size, mask_size),
    )
    score = tracker_builder._input(
        network,
        "object_score_logits",
        trt.float32,
        (batch_size, 1),
    )
    suppression = (
        _constant_zero_suppression(network, batch_size=batch_size)
        if hard_mask
        else tracker_builder._input(
            network,
            "suppress_area_shrinkage",
            trt.int32,
            (batch_size, 1),
        )
    )
    layer = network.add_plugin_v2(
        [feature, mask, score, suppression],
        _create_plugin(trt, global_name, batch_size=batch_size),
    )
    if layer is None or layer.num_outputs != 1:
        raise RuntimeError("Could not add the SAM3 tracker-memory TensorRT plugin layer")
    _add_output_contract(network, layer.get_output(0), batch_size=batch_size)
    policy = "hard" if hard_mask else "soft"
    return tracker_builder._serialize(
        builder,
        network,
        config,
        kind=f"{policy} B{batch_size} TVM-FFI memory",
        verbose=verbose,
    )


def build_sam3_tracker_memory_ffi_plans(
    spec: TrackerMemoryPlanSpec,
    *,
    verbose: bool = False,
) -> dict[str, bytes]:
    """Build soft/hard B1/B2 plans for exported tracker-memory packages."""

    variants = (
        (TRACKER_MEMORY_SECTION, spec.soft_global_b1, 1, False),
        (TRACKER_MEMORY_BATCH2_SECTION, spec.soft_global_b2, 2, False),
        (TRACKER_HARD_MEMORY_SECTION, spec.hard_global_b1, 1, True),
        (TRACKER_HARD_MEMORY_BATCH2_SECTION, spec.hard_global_b2, 2, True),
    )
    for _, global_name, batch_size, hard_mask in variants:
        _validate_global_name(
            global_name,
            batch_size=batch_size,
            hard_mask=hard_mask,
        )

    from .native_plugin_builder import load_native_plugin

    load_native_plugin(spec.plugin_library)
    return {
        section: _build_tracker_memory_ffi_plan(
            global_name,
            batch_size=batch_size,
            hard_mask=hard_mask,
            verbose=verbose,
        )
        for section, global_name, batch_size, hard_mask in variants
    }


__all__ = [
    "TrackerMemoryPlanSpec",
    "build_sam3_tracker_memory_ffi_plans",
]
