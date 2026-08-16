# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT Network Definition builders for the distilled stereo model."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any


FEATURE_OUTPUT_NAMES = (
    "features_left_04",
    "features_left_08",
    "features_left_16",
    "features_left_32",
    "features_right_04",
    "stem_2x",
)
POST_INPUT_NAMES = FEATURE_OUTPUT_NAMES

_FEATURE_OUTPUT_SHAPES = {
    "features_left_04": (1, 224, 176, 176),
    "features_left_08": (1, 192, 88, 88),
    "features_left_16": (1, 320, 44, 44),
    "features_left_32": (1, 304, 22, 22),
    "features_right_04": (1, 224, 176, 176),
    "stem_2x": (1, 16, 352, 352),
}


@contextmanager
def _model_source_scope(model_root: Path):
    old_cwd = Path.cwd()
    source = str(model_root)
    sys.path.insert(0, source)
    os.chdir(model_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == source:
            sys.path.pop(0)


def _load_model(model_root: Path, *, max_disparity: int, valid_iters: int):
    import torch

    from .prepare_model import (
        configure_official_model_args,
        install_official_io_import_shims,
    )

    install_official_io_import_shims()
    checkpoint = model_root / "weights/23-36-37/model_best_bp2_serialize.pth"
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configure_official_model_args(
        model,
        max_disparity=max_disparity,
        valid_iters=valid_iters,
    )
    model.eval()
    return model


def _validate_precision(precision: str) -> bool:
    if precision not in {"fp16", "fp32"}:
        raise ValueError(
            "Fast Foundation Stereo supports precision='fp16' or precision='fp32'; "
            f"got {precision!r}"
        )
    return precision == "fp16"


def _create_network(*, verbose: bool) -> tuple[Any, Any, Any]:
    from tensorrt_model_connect import trt_compat

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    network = builder.create_network(flags)
    if network is None:
        raise RuntimeError("TensorRT failed to create the Fast Foundation Stereo network")
    return trt, builder, network


def _serialize_network(
    trt: Any,
    builder: Any,
    network: Any,
    *,
    fp16: bool,
    verbose: bool,
) -> bytes:
    del fp16, verbose
    config = builder.create_builder_config()
    workspace_gib = int(os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_WORKSPACE_GIB", "8"))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(
            os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_OPT_LEVEL", "5")
        )
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = int(
            os.environ.get("TRTMC_FAST_FOUNDATION_STEREO_AUX_STREAMS", "2")
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Fast Foundation Stereo engine")
    return bytes(plan)


def _mark_feature_outputs(graph: Any, network: Any, outputs: dict[str, Any], *, fp16: bool) -> None:
    if set(outputs) != set(FEATURE_OUTPUT_NAMES):
        raise RuntimeError(
            "native feature graph output mismatch: "
            f"expected {FEATURE_OUTPUT_NAMES}, got {tuple(outputs)}"
        )
    for name in FEATURE_OUTPUT_NAMES:
        tensor = outputs[name]
        expected_shape = _FEATURE_OUTPUT_SHAPES[name]
        if tuple(int(dim) for dim in tensor.shape) != expected_shape:
            raise RuntimeError(
                f"native feature tensor {name} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )
        target_dtype = (
            graph.trt.float32 if not fp16 or name == "features_left_32" else graph.trt.float16
        )
        tensor = graph.cast(tensor, target_dtype)
        tensor.name = name
        network.mark_output(tensor)


def build_feature_engine(
    model_dir: str,
    *,
    precision: str,
    max_disparity: int,
    valid_iters: int,
    verbose: bool = False,
) -> bytes:
    from .native_feature import add_feature_graph
    from .native_graph import NativeGraph

    fp16 = _validate_precision(precision)
    model_root = Path(model_dir).resolve()
    with _model_source_scope(model_root):
        model = _load_model(
            model_root,
            max_disparity=max_disparity,
            valid_iters=valid_iters,
        )
        trt, builder, network = _create_network(verbose=verbose)
        left = network.add_input("left", trt.float32, (1, 3, 704, 704))
        right = network.add_input("right", trt.float32, (1, 3, 704, 704))
        graph = NativeGraph(network, trt, fp16=fp16)
        outputs = add_feature_graph(graph, model, left, right, fp16=fp16)
        _mark_feature_outputs(graph, network, outputs, fp16=fp16)
        return _serialize_network(
            trt,
            builder,
            network,
            fp16=fp16,
            verbose=verbose,
        )


def build_post_engine(
    model_dir: str,
    *,
    precision: str,
    max_disparity: int,
    valid_iters: int,
    verbose: bool = False,
) -> bytes:
    from .native_graph import NativeGraph
    from .native_post import add_post_graph
    from .native_plugin_builder import load_native_plugin

    fp16 = _validate_precision(precision)
    model_root = Path(model_dir).resolve()
    with _model_source_scope(model_root):
        model = _load_model(
            model_root,
            max_disparity=max_disparity,
            valid_iters=valid_iters,
        )
        load_native_plugin(verbose=verbose)
        trt, builder, network = _create_network(verbose=verbose)
        inputs = {}
        for name in POST_INPUT_NAMES:
            dtype = trt.float32 if not fp16 or name == "features_left_32" else trt.float16
            inputs[name] = network.add_input(name, dtype, _FEATURE_OUTPUT_SHAPES[name])
        graph = NativeGraph(network, trt, fp16=fp16)
        disparity = add_post_graph(
            graph,
            model,
            inputs,
            max_disparity=max_disparity,
            valid_iters=valid_iters,
        )
        disparity = graph.cast(disparity, trt.float32)
        disparity.name = "disp"
        network.mark_output(disparity)
        return _serialize_network(
            trt,
            builder,
            network,
            fp16=fp16,
            verbose=verbose,
        )
