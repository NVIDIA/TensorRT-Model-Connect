# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vision tower follows the build precision: bf16 in a bf16 build, float32 in an fp32 build."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from .. import vision  # noqa: E402 - needs TensorRT

# A shrunken tower: the real patch grid (224 / 14 = 256 tokens), narrow layers.
WIDTH, HEADS, HEAD_DIM, MLP, DEPTH, PROJECTION = 32, 2, 16, 64, 2, 24


def _gpu_trt():
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("a TensorRT network needs a GPU")
    return trt


@pytest.fixture
def tiny_tower(monkeypatch):
    for name, value in (
        ("WIDTH", WIDTH),
        ("HEADS", HEADS),
        ("HEAD_DIM", HEAD_DIM),
        ("MLP", MLP),
        ("DEPTH", DEPTH),
    ):
        monkeypatch.setattr(vision, name, value)
    rng = np.random.default_rng(0)

    def tensor(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    root = vision._ROOT
    weights = {
        root + "embeddings.patch_embedding.weight": tensor(WIDTH, 3, vision.PATCH, vision.PATCH),
        root + "embeddings.patch_embedding.bias": tensor(WIDTH),
        root + "embeddings.position_embedding.weight": tensor(vision.TOKENS, WIDTH),
        root + "post_layernorm.weight": 1.0 + tensor(WIDTH),
        root + "post_layernorm.bias": tensor(WIDTH),
        "paligemma.model.multi_modal_projector.linear.weight": tensor(PROJECTION, WIDTH),
        "paligemma.model.multi_modal_projector.linear.bias": tensor(PROJECTION),
    }
    for index in range(DEPTH):
        p = f"{root}encoder.layers.{index}."
        for norm in ("layer_norm1", "layer_norm2"):
            weights[p + norm + ".weight"] = 1.0 + tensor(WIDTH)
            weights[p + norm + ".bias"] = tensor(WIDTH)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weights[p + f"self_attn.{proj}.weight"] = tensor(WIDTH, WIDTH)
            weights[p + f"self_attn.{proj}.bias"] = tensor(WIDTH)
        weights[p + "mlp.fc1.weight"], weights[p + "mlp.fc1.bias"] = tensor(MLP, WIDTH), tensor(MLP)
        weights[p + "mlp.fc2.weight"], weights[p + "mlp.fc2.bias"] = (
            tensor(WIDTH, MLP),
            tensor(WIDTH),
        )
    return weights


def _tower(trt, weights, dtype):
    from .. import model

    builder, network, config = model._builder("bf16" if dtype == trt.bfloat16 else "fp32", False)
    size = vision.IMAGE_SIZE
    pixels = network.add_input("pixel_values", trt.float32, (2, 3, size, size))
    out = vision.build_vision(network, pixels, weights, height=size, width=size, dtype=dtype)
    out.name = "image_tokens"
    network.mark_output(out)
    return builder, network, config, out


def _layers(network):
    return [network.get_layer(i) for i in range(network.num_layers)]


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_tower_layers_run_in_the_build_precision(tiny_tower, precision: str) -> None:
    trt = _gpu_trt()
    dtype = trt.bfloat16 if precision == "bf16" else trt.float32
    _, network, _, out = _tower(trt, tiny_tower, dtype)
    layers = _layers(network)
    matmuls = [layer for layer in layers if layer.type == trt.LayerType.MATRIX_MULTIPLY]
    # Four projections, two attention matmuls and two MLP matmuls per layer, plus the projector.
    assert len(matmuls) == 8 * DEPTH + 1
    assert all(layer.get_output(0).dtype == dtype for layer in matmuls)
    assert out.dtype == dtype
    # The pixel normalisation and the patch embedding stay float32 in both precisions.
    (conv,) = [layer for layer in layers if layer.type == trt.LayerType.CONVOLUTION]
    assert conv.get_output(0).dtype == trt.float32
    # So does the residual stream: two float32 additions of computed tensors per layer.
    producers = {
        layer.get_output(i).name: layer for layer in layers for i in range(layer.num_outputs)
    }

    def computed(tensor) -> bool:
        source = producers.get(tensor.name)
        return (
            source is not None
            and source.type != trt.LayerType.CONSTANT
            and not (
                source.type == trt.LayerType.CAST
                and producers[source.get_input(0).name].type == trt.LayerType.CONSTANT
            )
        )

    def is_sum(layer) -> bool:
        layer.__class__ = trt.IElementWiseLayer  # get_layer returns the base ILayer
        return layer.op == trt.ElementWiseOperation.SUM

    residual = [
        layer
        for layer in layers
        if layer.type == trt.LayerType.ELEMENTWISE
        and tuple(layer.get_output(0).shape) == (2, vision.TOKENS, WIDTH)
        and all(computed(layer.get_input(i)) for i in range(2))
        and layer.get_input(0).dtype == trt.float32
        and layer.get_output(0).dtype == trt.float32
        and is_sum(layer)
    ]
    assert len(residual) == 2 * DEPTH
    if precision == "fp32":
        # An fp32 build has no reduced-precision tensor anywhere in the tower.
        assert not any(layer.type == trt.LayerType.CAST for layer in layers)
    else:
        # Norm statistics and softmax are still taken in float32.
        softmax = [layer for layer in layers if layer.type == trt.LayerType.SOFTMAX]
        assert len(softmax) == DEPTH and all(
            layer.get_output(0).dtype == trt.float32 for layer in softmax
        )


def test_prefix_builds_the_tower_in_its_working_dtype(monkeypatch) -> None:
    from .. import prefix

    seen = []

    class _Stop(Exception):
        pass

    def fake_vision(*_args, **kwargs):
        seen.append(kwargs["dtype"])
        raise _Stop

    monkeypatch.setattr(vision, "build_vision", fake_vision)
    network = SimpleNamespace(add_input=lambda *_args: object())
    cfg = SimpleNamespace(
        camera_slots=3, tokenizer_max_length=200, camera_height=256, camera_width=256
    )
    for dtype in ("bf16-marker", "fp32-marker"):
        with pytest.raises(_Stop):
            prefix.build_prefix(network, {}, cfg, dtype=dtype)
    assert seen == ["bf16-marker", "fp32-marker"]


def test_bf16_tower_engine_tracks_the_fp32_tower(tiny_tower) -> None:
    trt = _gpu_trt()
    import torch

    from .. import model

    pixels = torch.rand(
        (2, 3, vision.IMAGE_SIZE, vision.IMAGE_SIZE), generator=torch.Generator().manual_seed(0)
    )
    outputs = {}
    for dtype, torch_dtype in ((trt.float32, torch.float32), (trt.bfloat16, torch.bfloat16)):
        builder, network, config, _ = _tower(trt, tiny_tower, dtype)
        plan = model._serialize(builder, network, config, "vision")
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        source = pixels.cuda()
        result = torch.empty((2, vision.TOKENS, PROJECTION), dtype=torch_dtype, device="cuda")
        context.set_tensor_address("pixel_values", source.data_ptr())
        context.set_tensor_address("image_tokens", result.data_ptr())
        stream = torch.cuda.Stream()
        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        outputs[dtype] = result.float().cpu()
    reference, reduced = outputs[trt.float32], outputs[trt.bfloat16]
    assert torch.isfinite(reduced).all()
    cosine = torch.nn.functional.cosine_similarity(reduced.flatten(), reference.flatten(), dim=0)
    assert cosine > 0.999
