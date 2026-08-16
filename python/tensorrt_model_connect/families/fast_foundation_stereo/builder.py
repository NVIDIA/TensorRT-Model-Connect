# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX-to-TensorRT builders for Fast Foundation Stereo's split graph."""

from __future__ import annotations

import io
import os
import sys
from contextlib import contextmanager
from pathlib import Path


FEATURE_OUTPUT_NAMES = (
    "features_left_04",
    "features_left_08",
    "features_left_16",
    "features_left_32",
    "features_right_04",
    "stem_2x",
)


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
    return model.cuda().eval()


def _unwrap_compiled(function):
    return getattr(function, "_torchdynamo_orig_callable", function)


def _disable_compile_wrappers() -> None:
    import core.foundation_stereo as foundation_stereo
    import core.geometry as geometry
    import core.submodule as submodule

    submodule.build_concat_volume_optimized_pytorch = _unwrap_compiled(
        submodule.build_concat_volume_optimized_pytorch
    )
    foundation_stereo.build_concat_volume_optimized_pytorch = (
        submodule.build_concat_volume_optimized_pytorch
    )
    geometry.bilinear_sampler1d = _unwrap_compiled(geometry.bilinear_sampler1d)


def _feature_onnx(model, *, fp16: bool) -> bytes:
    import torch
    from core.foundation_stereo import TrtFeatureRunner

    class FeatureWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.runner = TrtFeatureRunner(model)

        def forward(self, left, right):
            with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
                return self.runner(left, right)

    image = torch.zeros((1, 3, 704, 704), device="cuda", dtype=torch.float32)
    output = io.BytesIO()
    torch.onnx.export(
        FeatureWrapper().cuda().eval(),
        (image, image),
        output,
        input_names=("left", "right"),
        output_names=FEATURE_OUTPUT_NAMES,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    return output.getvalue()


def _post_onnx(model, *, fp16: bool) -> bytes:
    import torch
    from core.foundation_stereo import TrtPostRunner

    _disable_compile_wrappers()

    class PostWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.runner = TrtPostRunner(model)

        def forward(
            self,
            features_left_04,
            features_left_08,
            features_left_16,
            features_left_32,
            features_right_04,
            stem_2x,
            gwc_volume,
        ):
            with torch.amp.autocast("cuda", enabled=fp16, dtype=torch.float16):
                disparity = self.runner(
                    features_left_04,
                    features_left_08,
                    features_left_16,
                    features_left_32,
                    features_right_04,
                    stem_2x,
                    gwc_volume,
                )
            return disparity.float()

    dtype = torch.float16 if fp16 else torch.float32
    inputs = (
        torch.zeros((1, 224, 176, 176), device="cuda", dtype=dtype),
        torch.zeros((1, 192, 88, 88), device="cuda", dtype=dtype),
        torch.zeros((1, 320, 44, 44), device="cuda", dtype=dtype),
        torch.zeros((1, 304, 22, 22), device="cuda", dtype=torch.float32),
        torch.zeros((1, 224, 176, 176), device="cuda", dtype=dtype),
        torch.zeros((1, 16, 352, 352), device="cuda", dtype=dtype),
        torch.zeros((1, 8, 48, 176, 176), device="cuda", dtype=dtype),
    )
    output = io.BytesIO()
    torch.onnx.export(
        PostWrapper().cuda().eval(),
        inputs,
        output,
        input_names=FEATURE_OUTPUT_NAMES + ("gwc_volume",),
        output_names=("disp",),
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    return output.getvalue()


def _build_engine(
    onnx_bytes: bytes,
    *,
    fp16: bool,
    optimization_level: int,
    auxiliary_streams: int,
    workspace_gib: int,
    verbose: bool,
) -> bytes:
    from tensorrt_model_connect import trt_compat

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = trt_compat.network_creation_flags(explicit_batch=True)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse Fast Foundation Stereo ONNX:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if fp16:
        _set_fp16_flag_if_supported(config, trt)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = optimization_level
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = auxiliary_streams
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build Fast Foundation Stereo engine")
    return bytes(plan)


def _set_fp16_flag_if_supported(config, trt) -> None:
    """Enable the legacy weakly typed FP16 policy when TensorRT exposes it."""
    builder_flag = getattr(trt, "BuilderFlag", None)
    fp16_flag = getattr(builder_flag, "FP16", None)
    if fp16_flag is not None:
        config.set_flag(fp16_flag)


def _validate_precision(precision: str) -> bool:
    if precision not in {"fp16", "fp32"}:
        raise ValueError(
            "Fast Foundation Stereo supports precision='fp16' or precision='fp32'; "
            f"got {precision!r}"
        )
    return precision == "fp16"


def build_feature_engine(
    model_dir: str,
    *,
    precision: str,
    max_disparity: int,
    valid_iters: int,
    verbose: bool = False,
) -> bytes:
    model_root = Path(model_dir).resolve()
    fp16 = _validate_precision(precision)
    with _model_source_scope(model_root):
        model = _load_model(model_root, max_disparity=max_disparity, valid_iters=valid_iters)
        onnx_bytes = _feature_onnx(model, fp16=fp16)
        return _build_engine(
            onnx_bytes,
            fp16=fp16,
            optimization_level=3,
            auxiliary_streams=2,
            workspace_gib=8,
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
    model_root = Path(model_dir).resolve()
    fp16 = _validate_precision(precision)
    with _model_source_scope(model_root):
        model = _load_model(model_root, max_disparity=max_disparity, valid_iters=valid_iters)
        onnx_bytes = _post_onnx(model, fp16=fp16)
        return _build_engine(
            onnx_bytes,
            fp16=fp16,
            optimization_level=3,
            auxiliary_streams=2,
            workspace_gib=8,
            verbose=verbose,
        )
