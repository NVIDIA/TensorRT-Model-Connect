# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image FP8 self-quantization: weight scales, the scale table, the calibration tool, the build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest

from .. import calibrate_fp8_scales, model
from .. import qwen_image_dit_builder as dit
from .. import quantization as quantization_module
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    BLOCK_PROJECTIONS,
    QwenImageQuantContext,
    calibrate_qwen_image,
    configure_build,
    quantize_fp8_weight,
    quantized_names,
    scale_key,
)

HIDDEN, HEADS, HEAD_DIM, FFN, TEXT_DIM, BLOCKS = 128, 2, 64, 256, 64, 2
N_IMG_GRID, N_TEXT = 2, 8


def _cfg(**overrides) -> dit.QwenImageDiTConfig:
    values = {
        "in_channels": 16,
        "out_channels": 4,
        "patch_size": 2,
        "hidden_size": HIDDEN,
        "num_joint_blocks": BLOCKS,
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "intermediate_size": FFN,
        "text_embed_dim": TEXT_DIM,
        "rope_axes_dim": [16, 24, 24],
        "timestep_embed_dim": 32,
    }
    values.update(overrides)
    return dit.QwenImageDiTConfig(**values)


def _weights(cfg: dit.QwenImageDiTConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """A random transformer state dict with every key the engine reads."""
    rng = np.random.default_rng(seed)
    h, p = cfg.hidden_size, cfg.patch_size

    def mat(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def zeros(*shape):
        return np.zeros(shape, dtype=np.float32)

    weights = {
        "img_in.weight": mat(h, cfg.in_channels),
        "img_in.bias": zeros(h),
        "txt_norm.weight": np.ones(cfg.text_embed_dim, dtype=np.float32),
        "txt_in.weight": mat(h, cfg.text_embed_dim),
        "txt_in.bias": zeros(h),
        "time_text_embed.timestep_embedder.linear_1.weight": mat(h, cfg.timestep_embed_dim),
        "time_text_embed.timestep_embedder.linear_1.bias": zeros(h),
        "time_text_embed.timestep_embedder.linear_2.weight": mat(h, h),
        "time_text_embed.timestep_embedder.linear_2.bias": zeros(h),
        "norm_out.linear.weight": mat(2 * h, h),
        "norm_out.linear.bias": zeros(2 * h),
        "proj_out.weight": mat(cfg.out_channels * p * p, h),
        "proj_out.bias": zeros(cfg.out_channels * p * p),
    }
    inner = cfg.intermediate_size
    for block in range(cfg.num_joint_blocks):
        prefix = f"transformer_blocks.{block}."
        for mod in ("img_mod.1", "txt_mod.1"):
            weights[f"{prefix}{mod}.weight"] = mat(6 * h, h)
            weights[f"{prefix}{mod}.bias"] = zeros(6 * h)
        for stem in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
        ):
            weights[f"{prefix}{stem}.weight"] = mat(h, h)
            weights[f"{prefix}{stem}.bias"] = zeros(h)
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            weights[f"{prefix}attn.{norm}.weight"] = np.ones(cfg.attention_head_dim, np.float32)
        for stream in ("img_mlp", "txt_mlp"):
            weights[f"{prefix}{stream}.net.0.proj.weight"] = mat(inner, h)
            weights[f"{prefix}{stream}.net.0.proj.bias"] = zeros(inner)
            weights[f"{prefix}{stream}.net.2.weight"] = mat(h, inner)
            weights[f"{prefix}{stream}.net.2.bias"] = zeros(h)
    return weights


def _all_scales(cfg, value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(cfg)}


@pytest.fixture
def fake_scales(monkeypatch):
    """Use scales for the tiny shape instead of the committed table."""
    scales = _all_scales(_cfg())
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda cfg: scales)
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
    """Eleven projections of every joint block get a scale; the image attention output does not."""
    cfg = _cfg()
    ctx = calibrate_qwen_image(cfg, _weights(cfg), "fp8")
    assert len(ctx.scales) == BLOCKS * len(BLOCK_PROJECTIONS) == 22
    assert ctx.should_quantize("transformer_blocks.1.attn.to_add_out")
    assert not ctx.should_quantize("transformer_blocks.0.attn.to_out.0")
    assert not ctx.should_quantize("transformer_blocks.0.img_mod.1")
    assert not ctx.should_quantize("proj_out")
    assert ctx.has_fp8_weights


def test_projection_without_scale_stays_unquantized(monkeypatch) -> None:
    """A projection with no activation scale is left in BF16."""
    cfg = _cfg()
    scales = _all_scales(cfg)
    del scales["transformer_blocks.0.img_mlp.net.2"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    ctx = calibrate_qwen_image(cfg, _weights(cfg), "fp8")
    assert not ctx.should_quantize("transformer_blocks.0.img_mlp.net.2")
    assert ctx.should_quantize("transformer_blocks.0.txt_mlp.net.2")


def test_already_quantized_checkpoint_is_rejected(fake_scales) -> None:
    """A checkpoint that already stores quantization scales is not quantized a second time."""
    cfg = _cfg()
    weights = _weights(cfg)
    weights["transformer_blocks.0.attn.to_q.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_qwen_image(cfg, weights, "fp8")


def test_missing_weight_is_rejected(fake_scales) -> None:
    """A scale for a projection the checkpoint lacks fails loudly."""
    cfg = _cfg()
    weights = _weights(cfg)
    del weights["transformer_blocks.1.attn.to_k.weight"]
    with pytest.raises(ValueError, match="to_k.weight"):
        calibrate_qwen_image(cfg, weights, "fp8")


def test_unsupported_format_is_rejected() -> None:
    """Only fp8 is implemented."""
    cfg = _cfg()
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_qwen_image(cfg, _weights(cfg), "nvfp4")


def test_unknown_shape_has_a_clear_error() -> None:
    """A model shape with no committed scales names the shape and the calibration tool."""
    cfg = _cfg(num_joint_blocks=3)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_qwen_image(cfg, _weights(cfg), "fp8")


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a finite positive scale for each quantized projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = key.split("-")
        cfg = SimpleNamespace(
            hidden_size=int(match[1][1:]),
            num_joint_blocks=int(match[2][1:]),
            intermediate_size=int(match[3][1:]),
        )
        assert scale_key(cfg) == key
        assert set(scales) <= set(quantized_names(cfg)), key
        assert len(scales) >= len(quantized_names(cfg)) - len(calibrate_fp8_scales.UNQUANTIZED), key
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
    """ModelOpt module paths map to engine projection names; other linears are skipped."""
    name = "transformer_blocks.12.attn.to_add_out"
    assert calibrate_fp8_scales.weight_name(name) == name
    assert calibrate_fp8_scales.weight_name("transformer_blocks.3.img_mlp.net.0.proj") is not None
    for other in (
        "transformer_blocks.3.attn.to_out.0",
        "transformer_blocks.3.img_mod.1",
        "proj_out",
        "img_in",
        "txt_in",
        "transformer_blocks.3.attn.to_q.base",
    ):
        assert calibrate_fp8_scales.weight_name(other) is None


def test_calibration_tool_shape_config_matches_the_scale_key(tmp_path: Path) -> None:
    """The tool derives the same shape key as the build from the transformer config."""
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"num_attention_heads": 24, "attention_head_dim": 128, "num_layers": 60})
    )
    assert scale_key(calibrate_fp8_scales.shape_config(tmp_path)) == "qwen_image-h3072-l60-i12288"


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    """The command line reaches ``calibrate`` with the parsed values."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(args=args))
    monkeypatch.setattr(
        "sys.argv", ["tool", "--model-dir", str(tmp_path), "--steps", "7", "--prompts", "3"]
    )
    calibrate_fp8_scales.main()
    assert seen["args"] == (tmp_path, 7, 1024, 1024, 3)


def _request(model_dir: Path, **overrides):
    values = {
        "model_dir": str(model_dir),
        "task": "image_generation",
        "precision": "bf16",
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


def test_build_rejects_fp8_for_image_edit(tmp_path: Path) -> None:
    """FP8 has no condition-image scales; Edit is refused."""
    with pytest.raises(NotImplementedError, match="image_generation"):
        model.build(
            _request(tmp_path, task="image_edit", image_height=512, image_width=512), Mock()
        )


def test_dit_builder_rejects_fp8_with_condition_images(fake_scales, tmp_path: Path) -> None:
    """The denoiser builder refuses condition-image grids with FP8."""
    cfg = _cfg()
    with pytest.raises(NotImplementedError, match="text-to-image"):
        dit.build_qwen_image_dit_engine(
            cfg,
            _weights(cfg),
            tmp_path / "dit.plan",
            h_lat=2,
            w_lat=2,
            n_text=N_TEXT,
            image_token_shapes=[(2, 2), (2, 2)],
            quantization="fp8",
        )


class _RecordingNetwork:
    """Counts the FP8 layers a quantized linear adds, without building an engine."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        def add(*args, **kwargs):
            self.calls.append(name)
            return SimpleNamespace(
                get_output=lambda _index: SimpleNamespace(dtype="out"),
            )

        return add


def test_quantized_linear_uses_quantize_dequantize_matmul_cast() -> None:
    """A quantized linear adds weight constant, two Dequantize, one Quantize, a matmul and a cast."""
    ctx = QwenImageQuantContext(scales={"transformer_blocks.0.attn.to_q": 0.05})
    network = _RecordingNetwork()
    weight = np.random.default_rng(2).normal(0.0, 0.1, (16, 32)).astype(np.float32)
    ctx.matmul_3d(network, SimpleNamespace(dtype="bf16"), weight, "transformer_blocks.0.attn.to_q")
    assert network.calls.count("add_dequantize") == 2
    assert network.calls.count("add_quantize") == 1
    assert network.calls.count("add_matrix_multiply") == 1
    assert network.calls[-1] == "add_cast"
    assert len(ctx.keep_alive) == 1 and ctx.keep_alive[0].shape == (16, 32)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_and_is_smaller(fake_scales, tmp_path: Path) -> None:
    """The tiny FP8 denoiser builds, keeps FP8 weights, and is smaller than the BF16 one."""
    cfg = _cfg()
    weights = _weights(cfg)
    plans = {}
    for label, quantization in (("bf16", None), ("fp8", "fp8")):
        path = tmp_path / f"{label}.plan"
        dit.build_qwen_image_dit_engine(
            cfg,
            weights,
            path,
            h_lat=N_IMG_GRID,
            w_lat=N_IMG_GRID,
            n_text=N_TEXT,
            quantization=quantization,
        )
        plans[label] = path.stat().st_size
    assert plans["fp8"] < plans["bf16"]
    assert dit._QUANT_CTX is None
