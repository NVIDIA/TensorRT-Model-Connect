# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 FP8 quantization: weight scaling, projection coverage, calibration tool, engines."""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, convnext_builder, graph_ops, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    calibrate_dinov3,
    projection_weights,
    requires_fp32_work,
    quantize_fp8_weight,
    scale_key,
)

HIDDEN, FFN, LAYERS, HEADS, PATCH, IMAGE, REGISTERS = 64, 128, 2, 2, 16, 32, 2
VIT_RAW = {
    "model_type": "dinov3_vit",
    "image_size": IMAGE,
    "patch_size": PATCH,
    "hidden_size": HIDDEN,
    "intermediate_size": FFN,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": HEADS,
    "num_register_tokens": REGISTERS,
}
CONVNEXT_RAW = {
    "model_type": "dinov3_convnext",
    "hidden_sizes": [32, 64],
    "depths": [1, 2],
    "image_size": 64,
}


def _tiny_vit_weights(seed: int = 0) -> WeightDict:
    """Random fp16 weights in the builder's logical layout (linear weights ``[in, out]``)."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float16)

    weights = WeightDict(
        {
            "cls_token": normal(1, 1, HIDDEN),
            "register_tokens": normal(1, REGISTERS, HIDDEN),
            "patch.weight": normal(HIDDEN, 3, PATCH, PATCH),
            "patch.bias": normal(HIDDEN),
            "norm.weight": np.ones(HIDDEN, np.float16),
            "norm.bias": normal(HIDDEN),
        }
    )
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for name in ("norm1", "norm2"):
            weights[f"{prefix}.{name}.weight"] = np.ones(HIDDEN, np.float16)
            weights[f"{prefix}.{name}.bias"] = normal(HIDDEN)
        for name in ("layer_scale1", "layer_scale2"):
            weights[f"{prefix}.{name}.lambda1"] = np.full(HIDDEN, 0.5, np.float16)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.attention.{name}.weight"] = normal(HIDDEN, HIDDEN)
        weights[f"{prefix}.attention.q_proj.bias"] = normal(HIDDEN)
        weights[f"{prefix}.attention.v_proj.bias"] = normal(HIDDEN)
        weights[f"{prefix}.attention.o_proj.bias"] = normal(HIDDEN)
        weights[f"{prefix}.mlp.up_proj.weight"] = normal(HIDDEN, FFN)
        weights[f"{prefix}.mlp.up_proj.bias"] = normal(FFN)
        weights[f"{prefix}.mlp.down_proj.weight"] = normal(FFN, HIDDEN)
        weights[f"{prefix}.mlp.down_proj.bias"] = normal(HIDDEN)
    return weights


def _tiny_convnext_weights(seed: int = 0) -> WeightDict:
    """Random fp16 ConvNeXt weights in the builder's logical layout."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float16)

    weights = WeightDict()
    sizes, depths = CONVNEXT_RAW["hidden_sizes"], CONVNEXT_RAW["depths"]
    for stage, (channels, depth) in enumerate(zip(sizes, depths)):
        key = f"stage.{stage}"
        in_channels = 3 if stage == 0 else sizes[stage - 1]
        kernel = 4 if stage == 0 else 2
        norm_channels = channels if stage == 0 else in_channels
        weights[f"{key}.downsample.weight"] = normal(channels, in_channels, kernel, kernel)
        weights[f"{key}.downsample.bias"] = normal(channels)
        weights[f"{key}.downsample_norm.weight"] = np.ones(norm_channels, np.float16)
        weights[f"{key}.downsample_norm.bias"] = normal(norm_channels)
        for block in range(depth):
            base = f"{key}.block.{block}"
            weights[f"{base}.depthwise.weight"] = normal(channels, 1, 7, 7)
            weights[f"{base}.depthwise.bias"] = normal(channels)
            weights[f"{base}.norm.weight"] = np.ones(channels, np.float16)
            weights[f"{base}.norm.bias"] = normal(channels)
            weights[f"{base}.pointwise1.weight"] = normal(channels, 4 * channels)
            weights[f"{base}.pointwise1.bias"] = normal(4 * channels)
            weights[f"{base}.pointwise2.weight"] = normal(4 * channels, channels)
            weights[f"{base}.pointwise2.bias"] = normal(channels)
            weights[f"{base}.gamma"] = np.full(channels, 0.5, np.float16)
    weights["final_norm.weight"] = np.ones(sizes[-1], np.float16)
    weights["final_norm.bias"] = normal(sizes[-1])
    return weights


def _vit_cfg() -> dict:
    return model.resolve_vit_config(VIT_RAW)


def _convnext_cfg() -> dict:
    return convnext_builder.resolve_convnext_config(dict(CONVNEXT_RAW))


def _vit_names(cfg: dict) -> list[str]:
    stems = ("attention.qkv", "attention.o_proj", "mlp.up_proj", "mlp.down_proj")
    return [f"layer.{i}.{stem}" for i in range(cfg["num_hidden_layers"]) for stem in stems]


def _convnext_names(cfg: dict) -> list[str]:
    return [
        f"stage.{stage}.block.{block}.pointwise{index}"
        for stage, depth in enumerate(cfg["depths"])
        for block in range(depth)
        for index in (1, 2)
    ]


@pytest.fixture
def vit(monkeypatch):
    """Tiny ViT weights, geometry, and activation scales for every projection."""
    cfg, weights = _vit_cfg(), _tiny_vit_weights()
    scales = {name: 0.05 for name in _vit_names(cfg)}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *_: scales)
    return weights, cfg


@pytest.fixture
def convnext(monkeypatch):
    """Tiny ConvNeXt weights, geometry, and activation scales for every projection."""
    cfg, weights = _convnext_cfg(), _tiny_convnext_weights()
    scales = {name: 0.05 for name in _convnext_names(cfg)}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *_: scales)
    return weights, cfg


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is max|w| / 448 and the stored bytes round-trip to the weight."""
    weight = np.random.default_rng(1).normal(0.0, 0.1, (32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= 0.07 * np.abs(weight).max()


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax has no scale."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), np.float32))


def test_every_vit_projection_is_quantized(vit) -> None:
    """Four projections per layer get an FP8 weight; q, k and v share one fused weight."""
    weights, cfg = vit
    ctx = calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")
    assert set(ctx.scales) == set(_vit_names(cfg))
    assert ctx.scales["layer.1.attention.qkv"].weight.packed.shape == (HIDDEN, 3 * HIDDEN)
    assert ctx.scales["layer.0.mlp.down_proj"].weight.packed.shape == (FFN, HIDDEN)
    assert ctx.has_fp8_weights and not ctx.should_quantize("layer.0.attention.q_proj")


def test_fused_qkv_weight_scale_covers_all_three_weights(vit) -> None:
    """The fused q/k/v weight uses one scale from the largest of the three."""
    weights, cfg = vit
    weights["layer.0.attention.k_proj.weight"] = weights["layer.0.attention.k_proj.weight"] * 8
    ctx = calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")
    expected = float(np.abs(weights["layer.0.attention.k_proj.weight"].astype(np.float32)).max())
    assert ctx.scales["layer.0.attention.qkv"].weight.weight_scale == pytest.approx(expected / 448)


def test_gated_mlp_adds_gate_projection(vit) -> None:
    """A gated MLP also quantizes its gate projection."""
    weights, cfg = vit
    cfg = dict(cfg, use_gated_mlp=True)
    for layer in range(LAYERS):
        weights[f"layer.{layer}.mlp.gate_proj.weight"] = weights[
            f"layer.{layer}.mlp.up_proj.weight"
        ]
    assert "layer.1.mlp.gate_proj" in projection_weights("dinov3_vit", cfg, weights)


def test_every_convnext_projection_is_quantized(convnext) -> None:
    """Both pointwise projections of every block get an FP8 weight."""
    weights, cfg = convnext
    ctx = calibrate_dinov3("dinov3_convnext", cfg, weights, "fp8")
    assert set(ctx.scales) == set(_convnext_names(cfg)) and len(ctx.scales) == 6
    assert ctx.scales["stage.1.block.1.pointwise1"].weight.packed.shape == (64, 256)


def test_projection_without_scale_raises(vit, monkeypatch) -> None:
    """A projection with no activation scale is an error, never a silent unquantized layer."""
    weights, cfg = vit
    scales = {name: 0.05 for name in _vit_names(cfg) if name != "layer.0.mlp.up_proj"}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"1 projections without a scale.*layer\.0\.mlp\.up_proj"):
        calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")


def test_scale_without_projection_raises(vit, monkeypatch) -> None:
    """A table with an entry the model has no projection for belongs to another shape."""
    weights, cfg = vit
    scales = {name: 0.05 for name in _vit_names(cfg)} | {"layer.0.mlp.gate_proj": 0.05}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"1 scales without a projection.*gate_proj"):
        calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")


def test_gated_mlp_without_gate_scale_raises(vit) -> None:
    """The gate projection must have a scale as well: the table of a non-gated shape fails."""
    weights, cfg = vit
    cfg = dict(cfg, use_gated_mlp=True)
    for layer in range(LAYERS):
        weights[f"layer.{layer}.mlp.gate_proj.weight"] = weights[
            f"layer.{layer}.mlp.up_proj.weight"
        ]
    with pytest.raises(ValueError, match="gate_proj"):
        calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_activation_scale_raises(vit, monkeypatch, bad) -> None:
    """A scale must be finite and positive."""
    weights, cfg = vit
    scales = {name: 0.05 for name in _vit_names(cfg)} | {"layer.1.mlp.down_proj": bad}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")


def test_shapes_that_overflow_fp16_need_fp32_work(vit) -> None:
    """ViT-L and larger give NaN in fp16 (activations above 65504): FP8 needs fp32 work."""
    weights, cfg = vit
    assert not requires_fp32_work("dinov3_vit", cfg)
    assert not requires_fp32_work("dinov3_convnext", {"hidden_sizes": [192, 384, 768, 1536]})
    for hidden in (1024, 1280, 4096):
        assert requires_fp32_work("dinov3_vit", dict(cfg, hidden_size=hidden))
    with pytest.raises(ValueError, match="overflows fp16"):
        calibrate_dinov3(
            "dinov3_vit", dict(cfg, hidden_size=1024), weights, "fp8", precision="fp16"
        )
    assert calibrate_dinov3("dinov3_vit", cfg, weights, "fp8", precision="fp16").scales


def test_fp16_build_rejects_scales_below_the_fp16_normal_range(vit) -> None:
    """A Q/DQ scale that fp16 cannot hold with full precision is rejected for an fp16 engine."""
    weights, cfg = vit
    weights["layer.0.mlp.up_proj.weight"] = weights["layer.0.mlp.up_proj.weight"] * 1e-3
    with pytest.raises(ValueError, match="smallest normal fp16"):
        calibrate_dinov3("dinov3_vit", cfg, weights, "fp8", precision="fp16")
    assert calibrate_dinov3("dinov3_vit", cfg, weights, "fp8", precision="fp32").scales


def test_unsupported_format_is_rejected(vit) -> None:
    """Only fp8 is implemented."""
    weights, cfg = vit
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_dinov3("dinov3_vit", cfg, weights, "nvfp4")


def test_unknown_model_shape_is_reported() -> None:
    """A shape with no committed scales fails with the available shapes."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        calibrate_dinov3("dinov3_vit", _vit_cfg(), _tiny_vit_weights(), "fp8")


def test_scale_key_comes_from_the_resolved_geometry() -> None:
    """The key names the geometry and the input size of each architecture."""
    assert scale_key("dinov3_vit", _vit_cfg()) == (
        f"dinov3_vit-h{HIDDEN}-l{LAYERS}-i{FFN}-p{PATCH}-s{IMAGE}x{IMAGE}"
    )
    assert scale_key("dinov3_vit", dict(_vit_cfg(), use_gated_mlp=True)).endswith(
        f"-p{PATCH}-gated-s{IMAGE}x{IMAGE}"
    )
    assert scale_key("dinov3_vit", dict(_vit_cfg(), image_h=64, image_w=64)).endswith("-s64x64")
    assert scale_key("dinov3_convnext", _convnext_cfg()) == "dinov3_convnext-c32x64-d1x2-s64x64"
    with pytest.raises(ValueError, match="unsupported"):
        scale_key("dinov2", {})


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a positive finite scale for each projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key
        vit_match = re.fullmatch(
            r"dinov3_vit-h(\d+)-l(\d+)-i(\d+)-p(\d+)(-gated)?-s(\d+)x(\d+)", key
        )
        if vit_match:
            layers = int(vit_match.group(2))
            stems = ["attention.qkv", "attention.o_proj", "mlp.up_proj", "mlp.down_proj"]
            stems += ["mlp.gate_proj"] if vit_match.group(5) else []
            assert set(scales) == {f"layer.{i}.{stem}" for i in range(layers) for stem in stems}, (
                key
            )
            continue
        match = re.fullmatch(r"dinov3_convnext-c([\dx]+)-d([\dx]+)-s(\d+)x(\d+)", key)
        assert match, key
        depths = [int(v) for v in match.group(2).split("x")]
        assert set(scales) == {
            f"stage.{s}.block.{b}.pointwise{i}"
            for s, depth in enumerate(depths)
            for b in range(depth)
            for i in (1, 2)
        }, key


def test_shipped_model_shapes_have_scale_tables() -> None:
    """ViT-S/B/L/H+ at 224, ViT-B/L/H+ at 512 and ConvNeXt-T at 224 are calibrated."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert set(table) == {
        "dinov3_vit-h384-l12-i1536-p16-s224x224",
        "dinov3_vit-h768-l12-i3072-p16-s224x224",
        "dinov3_vit-h768-l12-i3072-p16-s512x512",
        "dinov3_vit-h1024-l24-i4096-p16-s224x224",
        "dinov3_vit-h1024-l24-i4096-p16-s512x512",
        "dinov3_vit-h1280-l32-i5120-p16-gated-s224x224",
        "dinov3_vit-h1280-l32-i5120-p16-gated-s512x512",
        "dinov3_convnext-c96x192x384x768-d3x3x9x3-s224x224",
    }


def test_calibration_tool_maps_module_names_to_builder_names() -> None:
    """Only the quantized projections are calibrated; q_proj stands for the fused qkv input."""
    name = calibrate_fp8_scales.projection_name
    assert name("layer.3.attention.q_proj") == "layer.3.attention.qkv"
    assert name("model.layer.3.attention.o_proj") == "layer.3.attention.o_proj"
    assert name("layer.0.mlp.down_proj") == "layer.0.mlp.down_proj"
    assert name("layer.7.mlp.gate_proj") == "layer.7.mlp.gate_proj"
    assert name("layer.0.attention.k_proj") is None
    assert name("layer.0.attention.v_proj") is None
    assert name("embeddings.patch_embeddings") is None
    assert name("stages.2.layers.5.pointwise_conv1") == "stage.2.block.5.pointwise1"
    assert name("stages.0.layers.0.depthwise_conv") is None


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
    assert calls == [(tmp_path / "images", 10), (tmp_path, tmp_path, 200, None)]


def test_build_rejects_unsupported_quantization(tmp_path: Path) -> None:
    """Only fp8 or no quantization is accepted."""
    from tensorrt_model_connect import BuildRequest

    (tmp_path / "config.json").write_text(json.dumps(VIT_RAW))
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b",
        family="dinov3",
        task="image_features",
        precision="fp16",
        quantization="nvfp4",
    )
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model.build(request, None)


class _Engine:
    """A deserialized plan that can be run on CUDA tensors."""

    def __init__(self, plan: bytes) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def run(self, torch, value, output: str = "last_hidden_state"):
        out = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda")
        self.context.set_tensor_address("pixel_values", value.data_ptr())
        for name in ("last_hidden_state", "pooler_output"):
            tensor = (
                out
                if name == output
                else torch.empty(tuple(self.engine.get_tensor_shape(name)), device="cuda")
            )
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(-1)


def _build_pair(kind: str, weights, cfg, precision: str = "fp16"):
    config = types.SimpleNamespace(raw=dict(VIT_RAW if kind == "vit" else CONVNEXT_RAW))
    model_type = "dinov3_vit" if kind == "vit" else "dinov3_convnext"
    if precision == "fp32":
        # The builders read weights in the work dtype, as the loader gives them.
        weights = WeightDict({key: np.asarray(value, np.float32) for key, value in weights.items()})
    ctx = calibrate_dinov3(model_type, cfg, weights, "fp8", precision=precision)
    if kind == "vit":
        plain = model.build_vit_engine(config, weights, precision=precision, verbose=False)
        quant = model.build_vit_engine(
            config, weights, precision=precision, verbose=False, quant_ctx=ctx
        )
    else:
        plain = convnext_builder.build_convnext_engine(config, weights, precision=precision)
        quant = convnext_builder.build_convnext_engine(
            config, weights, precision=precision, quant_ctx=ctx
        )
    return plain, quant


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("kind", ["vit", "convnext"])
def test_quantized_engines_build_and_match_unquantized(kind, precision, request) -> None:
    """The FP8 engine builds in the work precision, is smaller, finite, and tracks the base."""
    torch = pytest.importorskip("torch")
    weights, cfg = request.getfixturevalue(kind)
    plain, quant = _build_pair(kind, weights, cfg, precision)
    assert len(quant) < len(plain)
    size = IMAGE if kind == "vit" else CONVNEXT_RAW["image_size"]
    pixels = torch.randn(1, 3, size, size, device="cuda")
    reference = _Engine(plain).run(torch, pixels)
    actual = _Engine(quant).run(torch, pixels)
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
    assert cosine > 0.9, cosine


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("dtype", [trt.float16, trt.float32])
def test_fp8_matmul_runs_in_the_activation_type_without_casts(dtype, vit) -> None:
    """The Dequantize layers and the matmul use the work type; no Cast layer is added.

    With an fp16 matmul output the bias add and GELU can fuse into the GEMM epilogue.
    """
    weights, cfg = vit
    ctx = calibrate_dinov3("dinov3_vit", cfg, weights, "fp8")
    _, network, _ = graph_ops.new_network(False, ctx)
    activation = network.add_input("x", dtype, (1, 8, HIDDEN))
    output = ctx.matmul(network, activation, "layer.0.attention.o_proj")
    assert output.dtype == dtype
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in {layer.type for layer in layers}
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
    assert {layer.get_output(0).dtype for layer in dequantize} == {dtype}


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names() -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    ctx = quantization_module.Dinov3QuantContext(scales={"x": object()})
    _, _, config = graph_ops.new_network(False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = graph_ops.new_network(False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED
