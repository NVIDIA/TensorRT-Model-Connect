# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image FP8 self-quantization: weight scales, the scale table, the calibration tool, the build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest

from .. import calibrate_fp8_scales, model
from .. import quantization as quantization_module
from .. import z_image_dit_builder as dit
from ..checkpoint_mapper import WeightDict
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    BLOCK_PROJECTIONS,
    ZImageQuantContext,
    calibrate_z_image,
    configure_build,
    is_quantized_checkpoint,
    quantize_fp8_weight,
    quantized_names,
    scale_key,
)

DIM, HEADS, HEAD_DIM, FFN, ADALN = 128, 4, 32, 256, 32
LAYERS, REFINERS, PATCHES, TEXT = 2, 1, 16, 8
SHAPE = {"dim": DIM, "num_layers": LAYERS, "num_refiner_layers": REFINERS, "ffn_dim": FFN}


def _weights(seed: int = 0) -> WeightDict:
    """Random engine weights (``[in, out]`` projections) with every key the DiT engine reads."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    weights = WeightDict()
    blocks = (
        [(f"noise_refiner.{i}", True) for i in range(REFINERS)]
        + [(f"context_refiner.{i}", False) for i in range(REFINERS)]
        + [(f"main.{i}", True) for i in range(LAYERS)]
    )
    for prefix, adaln in blocks:
        for stem in ("to_q", "to_k", "to_v", "to_out"):
            weights[f"{prefix}.{stem}"] = mat(DIM, DIM)
        weights[f"{prefix}.norm_q"] = np.ones(HEAD_DIM, np.float32)
        weights[f"{prefix}.norm_k"] = np.ones(HEAD_DIM, np.float32)
        for norm in ("attn_norm1", "attn_norm2", "ffn_norm1", "ffn_norm2"):
            weights[f"{prefix}.{norm}"] = np.ones(DIM, np.float32)
        weights[f"{prefix}.ff_w1"] = mat(DIM, FFN)
        weights[f"{prefix}.ff_w2"] = mat(FFN, DIM)
        weights[f"{prefix}.ff_w3"] = mat(DIM, FFN)
        if adaln:
            weights[f"{prefix}.adaln.weight"] = mat(ADALN, 4 * DIM)
            weights[f"{prefix}.adaln.bias"] = np.zeros(4 * DIM, np.float32)
    weights["final_adaLN.weight"] = mat(ADALN, DIM)
    weights["final_adaLN.bias"] = np.zeros(DIM, np.float32)
    weights["final_linear.weight"] = mat(DIM, 8)
    weights["final_linear.bias"] = np.zeros(8, np.float32)
    return weights


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(LAYERS, REFINERS)}


@pytest.fixture
def fake_scales(monkeypatch):
    """Use scales for the tiny shape instead of the committed table."""
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    return scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is amax / 448 and the bytes decode to the weight within e4m3 precision."""
    weight = np.random.default_rng(1).normal(0.0, 0.3, (32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(decoded - weight).max() <= float(np.abs(weight).max()) / 8


def test_all_zero_weight_is_rejected() -> None:
    """A weight with no positive amax has no scale to derive."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_every_block_projection_is_quantized(fake_scales) -> None:
    """Gate and up of every block get a scale; attention, ff_w2, modulation and final layers do not."""
    ctx = calibrate_z_image(_weights(), quantization="fp8", **SHAPE)
    assert len(ctx.scales) == (2 * REFINERS + LAYERS) * len(BLOCK_PROJECTIONS) == 8
    assert ctx.should_quantize("main.1.ff_w1")
    assert ctx.should_quantize("context_refiner.0.ff_w3")
    for projection in ("to_q", "to_k", "to_v", "to_out", "ff_w2"):
        assert not ctx.should_quantize(f"main.1.{projection}")
    assert not ctx.should_quantize("main.0.adaln")
    assert not ctx.should_quantize("final_linear")
    assert ctx.has_fp8_weights


def test_projection_without_scale_stays_unquantized(monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    scales = _all_scales()
    del scales["main.0.ff_w1"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    ctx = calibrate_z_image(_weights(), quantization="fp8", **SHAPE)
    assert not ctx.should_quantize("main.0.ff_w1")
    assert ctx.should_quantize("main.1.ff_w1")


def test_already_quantized_checkpoint_is_rejected(fake_scales) -> None:
    """A checkpoint that already stores quantization scales is not quantized a second time."""
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_z_image(_weights(), quantization="fp8", quantized_checkpoint=True, **SHAPE)
    readers = SimpleNamespace(tensor_map={"layers.0.attention.to_q.weight_scale": object()})
    assert is_quantized_checkpoint(readers)
    assert not is_quantized_checkpoint(SimpleNamespace(tensor_map={"layers.0.a.weight": 1}))


def test_missing_weight_is_rejected(fake_scales) -> None:
    """A scale for a projection the checkpoint lacks fails loudly."""
    weights = _weights()
    del weights["main.1.ff_w1"]
    with pytest.raises(ValueError, match="main.1.ff_w1"):
        calibrate_z_image(weights, quantization="fp8", **SHAPE)


def test_unsupported_format_is_rejected() -> None:
    """Only fp8 is implemented."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_z_image(_weights(), quantization="nvfp4", **SHAPE)


def test_unknown_shape_has_a_clear_error() -> None:
    """A model shape with no committed scales names the shape and the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_z_image(_weights(), quantization="fp8", **{**SHAPE, "num_layers": 3})


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a finite positive scale for each quantized projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        parts = key.split("-")
        dim, layers, refiners, ffn = (int(part[1:]) for part in parts[1:])
        assert scale_key(dim, layers, refiners, ffn) == key
        names = set(quantized_names(layers, refiners))
        assert set(scales) <= names, key
        assert len(scales) >= len(names) - len(calibrate_fp8_scales.UNQUANTIZED), key
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build options touch."""

    def __init__(self, all_build_routes: str) -> None:
        self.all_build_routes = all_build_routes
        self.build_route = ""
        self.profiling_verbosity = None


@pytest.mark.parametrize(
    ("routes", "expected"),
    [('{"peep:match_dual_gemm": {}}', "-peep:match_dual_gemm=off"), ("", "")],
)
def test_dual_gemm_route_is_set_only_when_listed(routes: str, expected: str) -> None:
    """The FP8 build route is applied only when the TensorRT build lists the knob."""
    config = _FakeBuilderConfig(routes)
    configure_build(config)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None


def test_calibration_tool_maps_module_names_to_engine_names() -> None:
    """Diffusers module paths map to engine projection names; other linears are skipped."""
    weight_name = calibrate_fp8_scales.weight_name
    assert weight_name("layers.12.feed_forward.w1") == "main.12.ff_w1"
    assert weight_name("noise_refiner.1.feed_forward.w3") == "noise_refiner.1.ff_w3"
    assert weight_name("context_refiner.0.feed_forward.w3") == "context_refiner.0.ff_w3"
    for other in (
        "layers.12.attention.to_out.0",
        "layers.12.attention.to_k",
        "layers.12.feed_forward.w2",
        "layers.3.adaLN_modulation.0",
        "all_final_layer.2-1.linear",
        "all_x_embedder.2-1",
        "cap_embedder.1",
        "t_embedder.mlp.0",
        "layers.3.attention.to_q.base",
    ):
        assert weight_name(other) is None


def test_calibration_tool_shape_config_matches_the_scale_key(tmp_path: Path) -> None:
    """The tool derives the same shape key as the build from the transformer config."""
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"dim": 3840, "n_layers": 30, "n_refiner_layers": 2})
    )
    cfg = calibrate_fp8_scales.shape_config(tmp_path)
    key = scale_key(cfg.dim, cfg.num_layers, cfg.num_refiner_layers, cfg.ffn_dim)
    assert key == scale_key(
        model._ZImageModel._DIT_DIM,
        model._ZImageModel._DIT_NUM_LAYERS,
        model._ZImageModel._DIT_NUM_REFINER_LAYERS,
        model._ZImageModel._DIT_FFN_DIM,
    )


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    """The command line reaches ``calibrate`` with the parsed values."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(args=args))
    monkeypatch.setattr(
        "sys.argv",
        ["tool", "--model-dir", str(tmp_path), "--steps", "7", "--sizes", "512", "--prompts", "3"],
    )
    calibrate_fp8_scales.main()
    assert seen["args"] == (tmp_path, 7, [512], 3)


def _request(model_dir: Path, **overrides):
    values = {
        "model_dir": str(model_dir),
        "task": "image_generation",
        "precision": "fp16",
        "quantization": "fp8",
        "dynamic_kv_cache": False,
        "max_sequence_length": None,
        "video_num_frames": None,
        "context_parallel_size": 1,
        "tensor_parallel_size": 1,
        "fp32_layers": [],
        "mtp_seq_len": None,
        "image_height": None,
        "image_width": None,
        "max_batch_size": 1,
        "verbose": False,
        "backend": "trt",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_rejects_unknown_quantization(tmp_path: Path) -> None:
    """Quantization formats other than fp8 are refused."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        model.build(_request(tmp_path, quantization="nvfp4"), Mock())


@pytest.mark.parametrize(
    "overrides",
    [{"precision": "fp32"}, {"tensor_parallel_size": 2}, {"max_batch_size": 2}],
)
def test_build_rejects_unsupported_fp8_combinations(tmp_path: Path, overrides: dict) -> None:
    """FP8 needs fp16, one device and batch size one."""
    with pytest.raises(NotImplementedError, match="fp16"):
        model.build(_request(tmp_path, **overrides), Mock())


def test_dit_builder_rejects_fp8_with_fp32_precision() -> None:
    """The denoiser builder refuses an FP8 context outside fp16."""
    with pytest.raises(NotImplementedError, match="fp16"):
        dit.build_z_image_dit_engine(
            _weights(),
            dim=DIM,
            num_heads=HEADS,
            num_layers=LAYERS,
            num_refiner_layers=REFINERS,
            ffn_dim=FFN,
            num_patches=PATCHES,
            text_seq_len=TEXT,
            head_dim=HEAD_DIM,
            adaln_embed_dim=ADALN,
            precision="fp32",
            quant_ctx=ZImageQuantContext(scales={"main.0.to_q": 0.02}),
        )


class _RecordingNetwork:
    """Counts the FP8 layers a quantized projection adds, without building an engine."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def add(*args, **kwargs):
            self.calls.append(name)
            return SimpleNamespace(get_output=lambda _index: SimpleNamespace(dtype="out"))

        return add


def test_quantized_projection_uses_quantize_dequantize_matmul_cast() -> None:
    """A quantized projection adds weight constant, two Dequantize, one Quantize, a matmul and a cast."""
    ctx = ZImageQuantContext(scales={"main.0.to_q": 0.05})
    network = _RecordingNetwork()
    weight = np.random.default_rng(2).normal(0.0, 0.1, (32, 16)).astype(np.float32)
    ctx.matmul(network, SimpleNamespace(dtype="fp16"), weight, "main.0.to_q")
    assert network.calls.count("add_dequantize") == 2
    assert network.calls.count("add_quantize") == 1
    assert network.calls.count("add_matrix_multiply") == 1
    assert network.calls[-1] == "add_cast"
    assert len(ctx.keep_alive) == 1 and ctx.keep_alive[0].shape == (32, 16)


class _SpyContext(ZImageQuantContext):
    """Records which projections are quantized and with which input-scale factor."""

    def __init__(self, scales):
        super().__init__(scales=scales)
        self.seen: dict[str, float] = {}

    def matmul(self, network, x, w_io, name):
        self.seen[name] = float(np.abs(w_io).max())
        return super().matmul(network, x, w_io, name)


def _build(quant_ctx, fp32_layers=()) -> int:
    plan = dit.build_z_image_dit_engine(
        _weights(),
        dim=DIM,
        num_heads=HEADS,
        num_layers=LAYERS,
        num_refiner_layers=REFINERS,
        ffn_dim=FFN,
        num_patches=PATCHES,
        text_seq_len=TEXT,
        head_dim=HEAD_DIM,
        adaln_embed_dim=ADALN,
        precision="fp16",
        fp32_layers=fp32_layers,
        quant_ctx=quant_ctx,
    )
    return len(plan)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_and_is_smaller() -> None:
    """The tiny FP8 denoiser builds, keeps FP8 weights, and is smaller than the fp16 one."""
    ctx = _SpyContext(_all_scales())
    fp16_size = _build(None)
    fp8_size = _build(ctx)
    assert fp8_size < fp16_size
    assert set(ctx.seen) == set(_all_scales())
    # The SwiGLU up branch is pre-scaled by 1/64 in fp16 blocks; the FP8 weight carries it.
    weights = _weights()
    assert ctx.seen["main.0.ff_w1"] == pytest.approx(float(np.abs(weights["main.0.ff_w1"]).max()))
    assert ctx.seen["main.0.ff_w3"] == pytest.approx(
        float(np.abs(weights["main.0.ff_w3"]).max()) / 64.0
    )
    assert "main.0.to_out" not in ctx.seen


@pytest.mark.gpu
@pytest.mark.trt
def test_fp32_blocks_are_not_quantized() -> None:
    """A block selected for fp32 (selector 0 is the first noise refiner) keeps plain weights."""
    ctx = _SpyContext(_all_scales())
    _build(ctx, fp32_layers=(0,))
    assert not any(name.startswith("noise_refiner.0.") for name in ctx.seen)
    assert "main.1.ff_w1" in ctx.seen
