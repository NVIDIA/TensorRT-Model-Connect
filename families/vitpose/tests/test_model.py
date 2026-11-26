# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ViTPose engine against `VitPoseForPoseEstimation`, heatmap for heatmap, on the GPU.

Both sides run in float32 with TF32 off on the same pixel values: the reference processor's crop of the E2E image
and a seeded random tensor. This isolates the graph (patch embedding, position table, encoder, head) from the native
crop and keypoint decoding, which tests/test_image_ops.py holds to the reference processor on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.vitpose import config as config_module
from families.vitpose import model as model_module
from families.vitpose import weights as weights_module

pytestmark = [pytest.mark.gpu, pytest.mark.trt]

TEST_ROOT = Path(__file__).resolve().parent
# Measured on one B300 (TensorRT 11.4, torch 2.15): the largest difference is 1.3e-6 on the E2E image's crop (heatmaps
# peak at 1.03) and 1.6e-6 on the random input. A tanh GELU moves the reference's heatmaps by 5.7e-4 or more, and
# position row 0 left out by 5.6e-4 or more, on both inputs.
_MAX_ABS_ERROR = 1e-4


def _manifest() -> dict:
    (path,) = sorted((TEST_ROOT / "manifests").glob("*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _model_dir(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
        )
    )


def _run_engine(plan: bytes, pixels, torch):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    assert engine is not None
    assert [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)] == [
        model_module.INPUT,
        model_module.OUTPUT,
    ]
    for name in (model_module.INPUT, model_module.OUTPUT):
        assert engine.get_tensor_dtype(name) == trt.float32, name
    assert tuple(engine.get_tensor_shape(model_module.INPUT)) == tuple(pixels.shape)
    output = torch.empty(
        tuple(engine.get_tensor_shape(model_module.OUTPUT)), dtype=torch.float32, device="cuda"
    )
    context = engine.create_execution_context()
    assert context is not None
    assert context.set_tensor_address(model_module.INPUT, pixels.data_ptr())
    assert context.set_tensor_address(model_module.OUTPUT, output.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return output


def test_engine_heatmaps_match_the_reference_model() -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, VitPoseForPoseEstimation

    assert torch.cuda.is_available(), "the engine parity test requires a CUDA GPU"
    manifest = _manifest()
    model_dir = _model_dir(manifest)
    cfg = config_module.resolve(model_dir, weights_module.read_header(model_dir))
    plan = model_module.build_engine(weights_module.load(model_dir, cfg), cfg)

    image = Image.open(TEST_ROOT / manifest["testcases"][0]["test_image"]).convert("RGB")
    crop = AutoProcessor.from_pretrained(model_dir)(
        image, boxes=[[[0.0, 0.0, float(image.width), float(image.height)]]], return_tensors="pt"
    )["pixel_values"]
    generator = torch.Generator().manual_seed(288)
    inputs = {
        "image crop": crop.to(torch.float32),
        "random": torch.randn(crop.shape, generator=generator, dtype=torch.float32) * 1.5,
    }
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = (
            VitPoseForPoseEstimation.from_pretrained(model_dir, dtype=torch.float32)
            .to("cuda")
            .eval()
        )
        for name, pixels in inputs.items():
            pixels = pixels.contiguous().to("cuda")
            native = _run_engine(plan, pixels, torch)
            with torch.no_grad():
                expected = model(pixel_values=pixels).heatmaps
            assert native.shape == expected.shape == (1, cfg.num_keypoints, *cfg.heatmap_size), name
            assert bool(torch.isfinite(native).all()), name
            error = float((native - expected).abs().max())
            assert error <= _MAX_ABS_ERROR, f"{name}: largest heatmap difference {error:.3g}"
            # Every peak, the keypoint the decoding starts from, is the reference's.
            native_peaks = native.flatten(2).argmax(dim=2).cpu().numpy()
            expected_peaks = expected.flatten(2).argmax(dim=2).cpu().numpy()
            if name == "image crop":
                assert np.array_equal(native_peaks, expected_peaks), name
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
