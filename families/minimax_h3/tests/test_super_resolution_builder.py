# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from families.minimax_h3 import trt_compat


if not trt_compat.is_available("tensorrt") and trt_compat.is_available("tensorrt_rtx"):
    trt_compat.configure_backend(rtx=True)
trt = pytest.importorskip("tensorrt")

from families.minimax_h3.super_resolution_builder import (  # noqa: E402
    BATCH_MAX,
    BATCH_MIN,
    BATCH_OPT,
    INPUT_NAME,
    OUTPUT_NAME,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    SUPER_RESOLUTION_PLAN_FILENAME,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    build_super_resolution_engine_from_weights,
    checkpoint_shapes,
    load_super_resolution_weights,
    super_resolution_metadata,
)


def _weights(fill: float = 0.0) -> dict[str, np.ndarray]:
    return {
        name: np.full(shape, fill, dtype=np.float32) for name, shape in checkpoint_shapes().items()
    }


def test_super_resolution_public_contract_is_exact_and_path_free() -> None:
    shapes = checkpoint_shapes()
    assert len(shapes) == 101
    assert shapes["body.0.weight"] == (64, 3, 3, 3)
    assert shapes["body.66.weight"] == (48, 64, 3, 3)
    assert SUPER_RESOLUTION_PLAN_FILENAME == "video_super_resolution.plan"

    metadata = super_resolution_metadata()
    assert metadata["architecture"] == "SRVGGNetCompact"
    assert metadata["source_shape"] == [480, 864]
    assert metadata["target_shape"] == [720, 1296]
    assert metadata["batch_profile"] == [1, 4, 8]
    assert metadata["input_name"] == "frames"
    assert metadata["output_name"] == "upscaled_frames"
    assert metadata["layout"] == "NHWC"
    assert metadata["io_dtype"] == "float32"
    assert metadata["learned_residual_strength"] == 0.25
    assert metadata["runtime_framework"] is None
    assert "\\" not in repr(metadata)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_super_resolution_denoise_strength_fails_closed(value) -> None:
    with pytest.raises(ValueError, match="denoise_strength"):
        super_resolution_metadata(denoise_strength=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), True, "0.25"])
def test_super_resolution_learned_residual_strength_fails_closed(value) -> None:
    with pytest.raises(ValueError, match="learned_residual_strength"):
        super_resolution_metadata(learned_residual_strength=value)


def test_super_resolution_strength_endpoints_are_public() -> None:
    assert (
        super_resolution_metadata(learned_residual_strength=0.0)["learned_residual_strength"] == 0.0
    )
    assert (
        super_resolution_metadata(learned_residual_strength=1.0)["learned_residual_strength"] == 1.0
    )


def test_checkpoint_loader_uses_official_dni_order(tmp_path, monkeypatch) -> None:
    import torch

    primary_path = tmp_path / "realesr-general-x4v3.pth"
    weak_path = tmp_path / "realesr-general-wdn-x4v3.pth"
    primary_path.write_bytes(b"primary")
    weak_path.write_bytes(b"weak")
    states = {
        primary_path: {"params": _weights(2.0)},
        weak_path: {"params": _weights(6.0)},
    }

    def fake_load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        return states[path]

    monkeypatch.setattr(torch, "load", fake_load)
    blended = load_super_resolution_weights(
        primary_path,
        weak_path,
        denoise_strength=0.25,
    )
    assert len(blended) == 101
    np.testing.assert_array_equal(blended["body.0.weight"], 5.0)
    np.testing.assert_array_equal(blended["body.65.weight"], 5.0)
    np.testing.assert_array_equal(blended["body.66.bias"], 5.0)

    primary = load_super_resolution_weights(primary_path, denoise_strength=1.0)
    np.testing.assert_array_equal(primary["body.0.weight"], 2.0)
    with pytest.raises(ValueError, match="weak-denoise checkpoint is required"):
        load_super_resolution_weights(primary_path)


@pytest.mark.gpu
@pytest.mark.trt
def test_native_super_resolution_plan_serializes_with_exact_abi(tmp_path) -> None:
    output_path = tmp_path / SUPER_RESOLUTION_PLAN_FILENAME
    record = build_super_resolution_engine_from_weights(
        _weights(),
        workspace_bytes=8 << 30,
        output_path=output_path,
    )
    assert record["bytes"] == output_path.stat().st_size

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(output_path.read_bytes())
    assert engine is not None
    assert engine.num_optimization_profiles == 1
    assert engine.num_io_tensors == 2
    assert tuple(engine.get_tensor_shape(INPUT_NAME)) == (
        -1,
        SOURCE_HEIGHT,
        SOURCE_WIDTH,
        3,
    )
    assert tuple(engine.get_tensor_shape(OUTPUT_NAME)) == (
        -1,
        TARGET_HEIGHT,
        TARGET_WIDTH,
        3,
    )
    assert engine.get_tensor_dtype(INPUT_NAME) == trt.float32
    assert engine.get_tensor_dtype(OUTPUT_NAME) == trt.float32
    assert tuple(tuple(shape) for shape in engine.get_tensor_profile_shape(INPUT_NAME, 0)) == (
        (BATCH_MIN, SOURCE_HEIGHT, SOURCE_WIDTH, 3),
        (BATCH_OPT, SOURCE_HEIGHT, SOURCE_WIDTH, 3),
        (BATCH_MAX, SOURCE_HEIGHT, SOURCE_WIDTH, 3),
    )
