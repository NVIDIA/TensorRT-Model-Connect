# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint, request, BN-folding and TensorRT activation contracts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorrt as trt
import torch

from families.yolox import graph
from families.yolox.checkpoint import Checkpoint
from families.yolox.model import _fold, build
from families.yolox.support import describe
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata


def test_exact_checkpoint_identity():
    assert describe(ModelMetadata(config={}, model_index={}, files=("yolox_s.pth",))) is not None
    for name in ("yolox_m.pth", "yolov5n.pt", "yolox.pth"):
        assert describe(ModelMetadata(config={}, model_index={}, files=(name,))) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "trt_rtx"),
        ("task", "image_classification"),
        ("dynamic_kv_cache", True),
        ("image_height", 320),
        ("image_width", 320),
        ("video_num_frames", 2),
        ("max_batch_size", 2),
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("quantization", "fp8"),
        ("fp32_layers", (0,)),
        ("max_sequence_length", 2),
    ],
)
def test_unsupported_request_fails_before_checkpoint_access(field, value):
    request = BuildRequest(
        model_dir=Path("missing"),
        output_path=Path("unused.bundle"),
        family="yolox",
        task="object_detection",
        precision="fp16",
    )
    with pytest.raises((NotImplementedError, ValueError)):
        build(replace(request, **{field: value}), None)


def test_fold_matches_pytorch_including_small_variance():
    generator = torch.Generator().manual_seed(10)
    conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False).eval()
    norm = torch.nn.BatchNorm2d(4, eps=1e-3).eval()
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.1)
        norm.weight.copy_(torch.tensor([0.5, 2.0, -0.1, 1.0]))
        norm.bias.copy_(torch.tensor([1.0, -2.0, 0.3, 0.0]))
        norm.running_mean.copy_(torch.tensor([0.2, -0.4, 0.0, 1.0]))
        norm.running_var.copy_(torch.tensor([0.0001, 0.01, 1.0, 4.0]))
    state = {
        "block.conv.weight": conv.weight.detach(),
        **{f"block.bn.{name}": value for name, value in norm.state_dict().items()},
    }
    weight, bias = _fold(Checkpoint(state), "block", np.float32)
    pixels = torch.randn((1, 3, 7, 9), generator=generator)
    actual = torch.nn.functional.conv2d(
        pixels, torch.from_numpy(weight), torch.from_numpy(bias), padding=1
    )
    torch.testing.assert_close(actual, norm(conv(pixels)), atol=2e-5, rtol=2e-5)
    # Wrong PyTorch-default epsilon must fail this numerical oracle.
    norm.eps = 1e-5
    assert (actual - norm(conv(pixels))).abs().max() > 1.0


def test_checkpoint_rejects_missing_extra_and_nonfinite_tensors(tmp_path):
    torch.save({"model": {"head.weight": torch.ones(1)}}, tmp_path / "yolox_s.pth")
    checkpoint = Checkpoint.open(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        checkpoint.tensor("backbone.weight")
    with pytest.raises(ValueError, match="Unsupported"):
        checkpoint.assert_consumed()
    with pytest.raises(ValueError, match="non-finite"):
        Checkpoint({"weight": torch.tensor([float("nan")])})


@pytest.mark.trt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TensorRT activation test requires CUDA")
def test_half_silu_matches_the_official_pytorch_activation():
    values = torch.tensor(
        [8.0078125, -8.0078125, 1, -1, 0, 3, 7], device="cuda", dtype=torch.float16
    )
    expected = torch.nn.functional.silu(values)
    # The old two-operation FP16 expression rounds the positive probe to 8.0.
    assert not torch.equal(values * values.sigmoid(), expected)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    tensor = network.add_input("input", trt.float16, tuple(values.shape))
    output = graph.silu(network, tensor)
    output.name = "output"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    actual = torch.empty_like(values)
    assert context.set_tensor_address("input", values.data_ptr())
    assert context.set_tensor_address("output", actual.data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
