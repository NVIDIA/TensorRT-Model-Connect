# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build SAM3 recurrent tracker-step plans around its model-owned TVM-FFI plugin."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRACKER_STEP_SECTION = "sam3_tracker_step_engine_plan"
TRACKER_STEP_BATCH2_SECTION = "sam3_tracker_step_batch2_engine_plan"

_MASK_SIZE = 288
_MASK_VALUES = _MASK_SIZE * _MASK_SIZE
_POINTER_VALUES = 256
_PACKED_WIDTH = _MASK_VALUES + _POINTER_VALUES + 1 + 1
_PLUGIN_NAME = "Sam3TrackerStepFfi"
_PLUGIN_VERSION = "2"
_GLOBAL_PATTERN = re.compile(r"trtmc\.sam3\.tracker_step\.b([12])\.split_aoti\.([0-9a-f]{20})\Z")


@dataclass(frozen=True)
class TrackerStepPlanSpec:
    """Exporter-to-plan handoff for one content-addressed SAM3 tracker runtime."""

    plugin_library: Path
    global_name_b1: str
    global_name_b2: str


def _validate_global_name(global_name: str, *, batch_size: int) -> None:
    match = _GLOBAL_PATTERN.fullmatch(global_name)
    if match is None or int(match.group(1)) != batch_size:
        raise ValueError(
            "SAM3 tracker-step global must identify the matching content-addressed "
            f"B{batch_size} split AOTI pipeline"
        )


def _plugin_creator(trt):
    registry = trt.get_plugin_registry()
    creator = registry.get_creator(_PLUGIN_NAME, _PLUGIN_VERSION, "")
    if creator is None:
        raise RuntimeError(
            "SAM3 tracker-step TensorRT plugin is not registered after loading its native DSO"
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
    plugin = _plugin_creator(trt).create_plugin("sam3_tracker_step_ffi", fields)
    if plugin is None:
        raise RuntimeError("Could not create the SAM3 tracker-step TensorRT plugin")
    return plugin


def _slice_packed_output(network, packed, *, batch_size: int, offset: int, width: int):
    layer = network.add_slice(
        packed,
        (0, offset),
        (batch_size, width),
        (1, 1),
    )
    if layer is None:
        raise RuntimeError("Could not slice a SAM3 tracker-step packed output")
    return layer.get_output(0)


def _reshape_output(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("Could not reshape a SAM3 tracker-step output")
    layer.reshape_dims = shape
    return layer.get_output(0)


def _add_output_contract(network, packed, *, batch_size: int) -> None:
    # The AOTI package publishes one object-major contiguous row per batch.
    # TensorRT views preserve the established four-output runtime ABI without
    # another device copy.
    mask = _slice_packed_output(
        network,
        packed,
        batch_size=batch_size,
        offset=0,
        width=_MASK_VALUES,
    )
    pointer = _slice_packed_output(
        network,
        packed,
        batch_size=batch_size,
        offset=_MASK_VALUES,
        width=_POINTER_VALUES,
    )
    score = _slice_packed_output(
        network,
        packed,
        batch_size=batch_size,
        offset=_MASK_VALUES + _POINTER_VALUES,
        width=1,
    )
    selected_iou = _slice_packed_output(
        network,
        packed,
        batch_size=batch_size,
        offset=_PACKED_WIDTH - 1,
        width=1,
    )

    from .tracker_builder import _mark

    _mark(
        network,
        _reshape_output(network, mask, (batch_size, 1, _MASK_SIZE, _MASK_SIZE)),
        "pred_masks",
    )
    _mark(
        network,
        _reshape_output(network, pointer, (batch_size, 1, _POINTER_VALUES)),
        "object_pointer",
    )
    _mark(
        network,
        _reshape_output(network, score, (batch_size, 1, 1)),
        "object_score_logits",
    )
    _mark(
        network,
        _reshape_output(network, selected_iou, (batch_size, 1, 1)),
        "selected_iou",
    )


def _build_tracker_step_ffi_plan(
    global_name: str,
    *,
    batch_size: int,
    verbose: bool,
) -> bytes:
    from . import tracker_builder

    _validate_global_name(global_name, batch_size=batch_size)
    trt, builder, network, config = tracker_builder._new_network(
        enable_tf32=False,
        verbose=verbose,
    )
    feature_0 = tracker_builder._input(
        network,
        "tracker_feature_0",
        trt.float32,
        tracker_builder._FEATURE_SHAPES[0],
    )
    feature_1 = tracker_builder._input(
        network,
        "tracker_feature_1",
        trt.float32,
        tracker_builder._FEATURE_SHAPES[1],
    )
    feature_2 = tracker_builder._input(
        network,
        "tracker_feature_2",
        trt.float32,
        tracker_builder._FEATURE_SHAPES[2],
    )
    position_2 = tracker_builder._input(
        network,
        "tracker_position_2",
        trt.float32,
        tracker_builder._POSITION_SHAPE,
    )
    memory_shape = (batch_size, -1, tracker_builder._SPATIAL_TOKENS, 64)
    memory_offset_shape = (batch_size, -1)
    pointer_shape = (batch_size, -1, _POINTER_VALUES)
    pointer_offset_shape = (batch_size, -1)
    memory_features = tracker_builder._input(
        network,
        "memory_features",
        trt.float32,
        memory_shape,
    )
    memory_position = tracker_builder._input(
        network,
        "memory_position",
        trt.float32,
        memory_shape,
    )
    memory_offsets = tracker_builder._input(
        network,
        "memory_temporal_offsets",
        trt.int32,
        memory_offset_shape,
    )
    object_pointers = tracker_builder._input(
        network,
        "object_pointers",
        trt.float32,
        pointer_shape,
    )
    pointer_offsets = tracker_builder._input(
        network,
        "object_pointer_temporal_offsets",
        trt.int32,
        pointer_offset_shape,
    )
    max_pointers = tracker_builder._input(
        network,
        "max_object_pointers_to_use",
        trt.int32,
        (1,),
    )
    inputs = [
        feature_0,
        feature_1,
        feature_2,
        position_2,
        memory_features,
        memory_position,
        memory_offsets,
        object_pointers,
        pointer_offsets,
        max_pointers,
    ]
    layer = network.add_plugin_v2(
        inputs,
        _create_plugin(trt, global_name, batch_size=batch_size),
    )
    if layer is None or layer.num_outputs != 1:
        raise RuntimeError("Could not add the SAM3 tracker-step TensorRT plugin layer")
    packed = layer.get_output(0)
    _add_output_contract(network, packed, batch_size=batch_size)
    tracker_builder._add_step_profile(builder, config, network, batch_size=batch_size)
    kind = "batch2 TVM-FFI step" if batch_size == 2 else "TVM-FFI step"
    return tracker_builder._serialize(
        builder,
        network,
        config,
        kind=kind,
        verbose=verbose,
    )


def build_sam3_tracker_step_ffi_plans(
    spec: TrackerStepPlanSpec,
    *,
    verbose: bool = False,
) -> dict[str, bytes]:
    """Build the required B1/B2 plans for one exported tracker-step runtime."""

    _validate_global_name(spec.global_name_b1, batch_size=1)
    _validate_global_name(spec.global_name_b2, batch_size=2)

    # Registration is process-local and must precede network construction so
    # TensorRT can find the model-owned plugin creator. The native loader keeps
    # the DSO alive for the lifetime of all serialized plans.
    from .native_plugin_builder import load_native_plugin

    load_native_plugin(spec.plugin_library)
    return {
        TRACKER_STEP_SECTION: _build_tracker_step_ffi_plan(
            spec.global_name_b1,
            batch_size=1,
            verbose=verbose,
        ),
        TRACKER_STEP_BATCH2_SECTION: _build_tracker_step_ffi_plan(
            spec.global_name_b2,
            batch_size=2,
            verbose=verbose,
        ),
    }


__all__ = [
    "TrackerStepPlanSpec",
    "build_sam3_tracker_step_ffi_plans",
]
