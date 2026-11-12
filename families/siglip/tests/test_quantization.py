# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, config as config_module, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    calibrate_siglip,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, PATCH, GRID, VOCAB, WINDOW = 128, 256, 2, 4, 16, 4, 100, 8
TOWER_PREFIXES = ("vision_model", "text_model")


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor both towers read, at a tiny geometry."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def norm(prefix):
        return {f"{prefix}.weight": np.ones(D, np.float32), f"{prefix}.bias": normal(D)}

    def linear(prefix, out, inp):
        return {f"{prefix}.weight": normal(out, inp), f"{prefix}.bias": normal(out)}

    weights = {"logit_scale": np.zeros((1,), np.float32), "logit_bias": np.zeros((1,), np.float32)}
    for tower in TOWER_PREFIXES:
        for layer in range(LAYERS):
            prefix = f"{tower}.encoder.layers.{layer}"
            weights.update(norm(f"{prefix}.layer_norm1"))
            weights.update(norm(f"{prefix}.layer_norm2"))
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                weights.update(linear(f"{prefix}.self_attn.{name}", D, D))
            weights.update(linear(f"{prefix}.mlp.fc1", FFN, D))
            weights.update(linear(f"{prefix}.mlp.fc2", D, FFN))
    weights["text_model.embeddings.token_embedding.weight"] = normal(VOCAB, D)
    weights["text_model.embeddings.position_embedding.weight"] = normal(WINDOW, D)
    weights.update(norm("text_model.final_layer_norm"))
    weights.update(linear("text_model.head", D, D))
    weights["vision_model.embeddings.patch_embedding.weight"] = normal(D, 3, PATCH, PATCH)
    weights["vision_model.embeddings.patch_embedding.bias"] = normal(D)
    weights["vision_model.embeddings.position_embedding.weight"] = normal(GRID * GRID, D)
    weights.update(norm("vision_model.post_layernorm"))
    head = "vision_model.head"
    weights[f"{head}.probe"] = normal(1, 1, D)
    weights[f"{head}.attention.in_proj_weight"] = normal(3 * D, D)
    weights[f"{head}.attention.in_proj_bias"] = normal(3 * D)
    weights.update(linear(f"{head}.attention.out_proj", D, D))
    weights.update(norm(f"{head}.layernorm"))
    weights.update(linear(f"{head}.mlp.fc1", FFN, D))
    weights.update(linear(f"{head}.mlp.fc2", D, FFN))
    return weights


def _config(tmp_path: Path, weights: dict) -> dict:
    """Resolve the tiny geometry the way ``build`` does."""
    stated = {"num_attention_heads": HEADS, "hidden_act": "gelu_pytorch_tanh"}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["SiglipModel"],
                "text_config": dict(stated),
                "vision_config": dict(stated),
            }
        )
    )
    return config_module.resolve(tmp_path, weights)


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny weights, their resolved geometry, and activation scales for every projection."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    return weights, cfg


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is max|w| / 448 and the stored bytes round-trip to the weight."""
    weight = np.random.default_rng(1).normal(0.0, 0.1, (16, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= 0.07 * np.abs(weight).max()


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax has no scale."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), np.float32))


def test_every_projection_of_both_towers_is_quantized(tiny) -> None:
    """Six projections per layer in each tower get an FP8 weight of the checkpoint's shape."""
    weights, cfg = tiny
    ctx = calibrate_siglip(weights, cfg, "fp8")
    assert len(ctx.scales) == 2 * LAYERS * 6 == len(projections(cfg))
    entry = ctx.scales["vision_model.encoder.layers.1.mlp.fc1"]
    assert entry.weight.packed.shape == (FFN, D)
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("vision_model.head.mlp.fc1")
    assert not ctx.should_quantize("text_model.head")


def test_projection_without_scale_stays_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection with no activation scale is left in float32."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    del scales["text_model.encoder.layers.0.self_attn.k_proj"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    ctx = calibrate_siglip(weights, cfg, "fp8")
    assert not ctx.should_quantize("text_model.encoder.layers.0.self_attn.k_proj")
    assert ctx.matmul("text_model.encoder.layers.0.self_attn.k_proj") is None
    assert ctx.should_quantize("text_model.encoder.layers.0.self_attn.q_proj")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg = tiny
    weights["vision_model.encoder.layers.0.mlp.fc1.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_siglip(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_siglip(weights, cfg, "nvfp4")


def test_unknown_model_shape_is_reported(tmp_path: Path) -> None:
    """A shape with no committed scales fails with the available shapes."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        calibrate_siglip(weights, cfg, "fp8")


def test_scale_key_comes_from_the_resolved_geometry(tmp_path: Path) -> None:
    """The key names both towers, the patch, the image size and the prompt window."""
    cfg = _config(tmp_path, _tiny_weights())
    assert scale_key(cfg) == f"siglip-v{D}x{LAYERS}p{PATCH}i{PATCH * GRID}-t{D}x{LAYERS}w{WINDOW}"


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a positive finite scale for each projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"siglip-v(\d+)x(\d+)p(\d+)i(\d+)-t(\d+)x(\d+)w(\d+)", key)
        assert match, key
        vision_layers, text_layers = int(match.group(2)), int(match.group(6))
        assert len(scales) == (vision_layers + text_layers) * 6, key
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key
        for tower, layers in (("vision_model", vision_layers), ("text_model", text_layers)):
            for layer in range(layers):
                for stem in quantization_module._LAYER_PROJECTIONS:
                    assert f"{tower}.encoder.layers.{layer}.{stem}" in scales, (key, tower, layer)


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only encoder projections of the two towers are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("vision_model.encoder.layers.3.mlp.fc1") == "vision_model.encoder.layers.3.mlp.fc1"
    assert name("text_model.encoder.layers.0.self_attn.out_proj") is not None
    assert name("vision_model.head.mlp.fc1") is None
    assert name("vision_model.head.attention.out_proj") is None
    assert name("text_model.head") is None
    assert name("vision_model.embeddings.patch_embedding") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The fetch and calibrate commands reach their functions with the parsed arguments."""
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_images", lambda *args: calls.append(args))
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: calls.append(args))
    monkeypatch.setattr(
        "sys.argv", ["calibrate", "fetch", "--out", str(tmp_path / "images"), "--count", "10"]
    )
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "calibrate", "--model-dir", str(tmp_path), "--images", str(tmp_path)],
    )
    calibrate_fp8_scales.main()
    assert calls == [(tmp_path / "images", 10), (tmp_path, tmp_path, 200)]


class _Engine:
    """A deserialized plan that can be run on CUDA tensors."""

    def __init__(self, plan: bytes) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def run(self, torch, name: str, value, output: str):
        out = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda")
        self.context.set_tensor_address(name, value.data_ptr())
        self.context.set_tensor_address(output, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(-1)


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_engines_build_and_match_float32(tiny) -> None:
    """Both FP8 engines build, are smaller, and track the float32 engines."""
    torch = pytest.importorskip("torch")
    weights, cfg = tiny
    ctx = calibrate_siglip(weights, cfg, "fp8")
    image_size = cfg["vision"]["image_size"]
    pixels = torch.randn(1, 3, image_size, image_size, device="cuda")
    ids = torch.randint(0, VOCAB, (WINDOW,), dtype=torch.int32, device="cuda")

    plain_vision = model.build_vision_engine(weights, cfg)
    plain_text = model.build_text_engine(weights, cfg)
    quant_vision = model.build_vision_engine(weights, cfg, quant_ctx=ctx)
    quant_text = model.build_text_engine(weights, cfg, quant_ctx=ctx)
    assert len(quant_vision) < len(plain_vision)
    assert len(quant_text) < len(plain_text)

    cases = (
        ("pixel_values", pixels, "image_embeds", plain_vision, quant_vision),
        ("input_ids", ids, "text_embeds", plain_text, quant_text),
    )
    for name, value, output, plain, quant in cases:
        reference = _Engine(plain).run(torch, name, value, output)
        actual = _Engine(quant).run(torch, name, value, output)
        cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
        assert cosine > 0.9, (output, cosine)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names(tiny) -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    _, _ = tiny
    ctx = quantization_module.SiglipQuantContext(scales={"x": object()})
    _, _, config = model._builder(False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = model._builder(False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED


def test_build_requires_float32_for_fp8(tmp_path: Path) -> None:
    """fp8 keeps the family's float32 rule: the quantized graph still runs float32 elsewhere."""
    from tensorrt_model_connect import BuildRequest

    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b",
        family="siglip",
        task="text_image_to_relevance",
        precision="fp16",
        quantization="fp8",
    )
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model.build(request, None)
