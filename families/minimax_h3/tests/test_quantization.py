# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 FP8 self-quantization: weight scales, the scale table, the calibration tool, the build."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest

from .. import calibrate_fp8_scales, model
from .. import quantization as quantization_module
from ..config import SOL_ENGINE_1344X768_124F
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    QUANTIZED,
    MiniMaxQuantContext,
    calibrate_minimax,
    quantize_fp8_weight,
    quantized_names,
    scale_key,
)

HIDDEN, LAYERS, FFN = 8, 2, 16
MODEL_CONFIG = {"hidden_size": HIDDEN, "num_layers": LAYERS, "ffn_dim": FFN}


def _weights() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    weights = {}
    for layer in range(LAYERS):
        p = f"transformer_blocks.{layer}"
        for name in ("q", "k", "v"):
            weights[f"{p}.attn.to_{name}.weight"] = rng.normal(0, 0.1, (8, HIDDEN)).astype(
                np.float32
            )
        weights[f"{p}.attn.to_out.0.weight"] = rng.normal(0, 0.1, (HIDDEN, 8)).astype(np.float32)
        weights[f"{p}.ff.net.0.proj.weight"] = rng.normal(0, 0.1, (2 * FFN, HIDDEN)).astype(
            np.float32
        )
        weights[f"{p}.ff.net.2.weight"] = rng.normal(0, 0.1, (HIDDEN, FFN)).astype(np.float32)
    return weights


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(LAYERS)}


@pytest.fixture
def fake_scales(monkeypatch):
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    return scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    weight = np.random.default_rng(1).normal(0.0, 0.3, (32, 16)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(decoded - weight).max() <= float(np.abs(weight).max()) / 8


def test_bf16_weight_is_accepted() -> None:
    weight = np.random.default_rng(1).normal(0.0, 0.3, (4, 4)).astype(ml_dtypes.bfloat16)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.shape == (4, 4) and scale > 0


def test_all_zero_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


def test_quantized_projections_get_a_scale(fake_scales) -> None:
    ctx = calibrate_minimax(_weights(), quantization="fp8", **MODEL_CONFIG)
    assert len(ctx.scales) == LAYERS * len(QUANTIZED)
    assert ctx.has_fp8_weights
    assert ctx.should_quantize("transformer_blocks.1.ff.net.0.proj")
    for other in (
        "transformer_blocks.1.attn.to_out.0",
        "transformer_blocks.0.attn.norm_q",
        "token_refiner.refiner_blocks.0.ff.net.2",
        "proj_in",
        "proj_out",
        "context_embedder",
    ):
        assert not ctx.should_quantize(other)


def test_projection_without_scale_is_rejected(monkeypatch) -> None:
    scales = _all_scales()
    del scales["transformer_blocks.0.ff.net.2"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    with pytest.raises(
        ValueError, match=r"transformer_blocks\.0\.ff\.net\.2.*calibrate_fp8_scales"
    ):
        calibrate_minimax(_weights(), quantization="fp8", **MODEL_CONFIG)


def test_already_quantized_checkpoint_is_rejected(fake_scales) -> None:
    weights = _weights()
    weights["transformer_blocks.0.ff.net.2.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_minimax(weights, quantization="fp8", **MODEL_CONFIG)


def test_missing_weight_is_rejected(fake_scales) -> None:
    weights = _weights()
    del weights["transformer_blocks.1.attn.to_q.weight"]
    with pytest.raises(ValueError, match=r"transformer_blocks\.1\.attn\.qkv"):
        calibrate_minimax(weights, quantization="fp8", **MODEL_CONFIG)


def test_unsupported_format_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_minimax(_weights(), quantization="nvfp4", **MODEL_CONFIG)


def test_unknown_model_config_has_a_clear_error() -> None:
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_minimax(_weights(), quantization="fp8", **{**MODEL_CONFIG, "num_layers": 3})


def test_committed_scales_cover_every_quantized_projection() -> None:
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    key = scale_key(
        SOL_ENGINE_1344X768_124F.hidden_size,
        SOL_ENGINE_1344X768_124F.num_layers,
        SOL_ENGINE_1344X768_124F.ffn_dim,
    )
    assert key == "minimax_h3-h5376-l50-i14336"
    for name, scales in table.items():
        layers = int(name.split("-")[2][1:])
        assert set(scales) >= set(quantized_names(layers)), name
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), name
    assert key in table


class _RecordingNetwork:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        def add(*args, **kwargs):
            self.calls.append((name, args))
            return SimpleNamespace(get_output=lambda _index: SimpleNamespace(dtype="out"))

        return add


def test_quantized_projection_is_quantize_dequantize_matmul_in_the_activation_type() -> None:
    ctx = MiniMaxQuantContext(scales={"transformer_blocks.0.ff.net.2": 0.05})
    network = _RecordingNetwork()
    weight = np.random.default_rng(2).normal(0.0, 0.1, (16, 32)).astype(np.float32)
    ctx.linear(network, SimpleNamespace(dtype="bf16"), weight, "transformer_blocks.0.ff.net.2")
    names = [name for name, _ in network.calls]
    assert names.count("add_dequantize") == 2 and names.count("add_quantize") == 1
    assert names.count("add_matrix_multiply") == 1 and "add_cast" not in names
    assert [args[-1] for name, args in network.calls if name == "add_dequantize"] == ["bf16"] * 2
    assert ctx.keep_alive[0].shape == (16, 32)


def test_calibration_tool_hooks_record_the_input_maximum() -> None:
    import torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.to_q = torch.nn.Linear(4, 4)
            self.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.ff = torch.nn.Module()
            self.ff.net = torch.nn.ModuleList(
                [torch.nn.Module(), torch.nn.Identity(), torch.nn.Linear(4, 4)]
            )
            self.ff.net[0].proj = torch.nn.Linear(4, 4)

    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.ModuleList([Block(), Block()])
    recorder = calibrate_fp8_scales._Recorder(transformer, 2)
    x = torch.tensor([[1.0, -3.0, 2.0, 0.5]])
    for block in transformer.transformer_blocks:
        block.attn.to_q(x)
        block.ff.net[0].proj(x * 2)
        block.ff.net[2](x * 3)
        block.attn.to_out[0](x)
    recorder.close()
    assert recorder.amax["transformer_blocks.1.attn.qkv"] == 3.0
    assert recorder.amax["transformer_blocks.0.ff.net.0.proj"] == 6.0
    assert recorder.amax["transformer_blocks.0.ff.net.2"] == 9.0
    assert set(QUANTIZED) <= set(calibrate_fp8_scales.HOOKED)


def test_calibration_tool_fails_when_a_module_is_missing() -> None:
    import torch

    with pytest.raises(RuntimeError, match="no module"):
        calibrate_fp8_scales._Recorder(torch.nn.Module(), 2)


def test_calibration_prompts_are_distinct_and_in_the_context_ir_format() -> None:
    prompts = calibrate_fp8_scales.CALIBRATION_PROMPTS
    assert len(prompts) == len(set(prompts)) == 8
    assert all(p.startswith("integrated_multimodal_description: [Shot 1]") for p in prompts)
    assert all("overall_soundscape:" in p and "non_diegetic_music:" in p for p in prompts)


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake(model_dir, **kwargs):
        seen.update(model_dir=model_dir, **kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", fake)
    monkeypatch.setattr(
        "sys.argv", ["tool", "--model-dir", str(tmp_path), "--steps", "7", "--num-prompts", "3"]
    )
    with pytest.raises(SystemExit):
        calibrate_fp8_scales.main()
    assert seen["steps"] == 7 and seen["num_prompts"] == 3 and seen["model_dir"] == tmp_path


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
    with pytest.raises(NotImplementedError, match="nvfp4"):
        model.build(_request(tmp_path, quantization="nvfp4"), Mock())


def test_fp8_with_first_block_cache_is_rejected() -> None:
    config = SimpleNamespace(raw={"denoiser_cache_mode": "first_block"})
    with pytest.raises(NotImplementedError, match="first_block_cache"):
        model._MiniMaxH3Model().build_components(
            "unused", config, {}, precision="bf16", quantization="fp8"
        )
    assert replace(SOL_ENGINE_1344X768_124F, first_block_cache=True).first_block_cache


def _tiny_profile():
    from ..config import MiniMaxH3Config

    return MiniMaxH3Config(
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        num_heads=4,
        head_dim=8,
        ffn_dim=16,
        video_in_channels=2,
        audio_in_channels=2,
        text_dim=8,
        timestep_input_dim=4,
        timestep_hidden_size=8,
        timestep_embed_dim=4,
        rope_freq_dim=1,
        video_rows=4,
        audio_rows=2,
        min_text_rows=1,
        opt_text_rows=2,
        text_rows=2,
        padded_sequence_length=8,
        max_timestep_count=2,
        context_parallel_size=1,
    )


class _SpyContext(MiniMaxQuantContext):
    seen: list[str] = []

    def linear(self, network, x, weight_oi, name):
        self.seen.append(name)
        return super().linear(network, x, weight_oi, name)


@pytest.mark.gpu
@pytest.mark.trt
def test_tiny_fp8_denoiser_quantizes_only_the_listed_projections(monkeypatch) -> None:
    """A tiny FP8 denoiser builds on bf16 activations; the refiner and to_out stay unquantized."""
    from ..dit_builder import build_dit_engine
    from .test_trt_builders import _weights as tiny_weights

    profile = _tiny_profile()
    scales = {name: 0.05 for name in quantized_names(1)}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    _SpyContext.seen = []
    original = quantization_module.MiniMaxQuantContext
    monkeypatch.setattr(quantization_module, "MiniMaxQuantContext", _SpyContext)
    plan = build_dit_engine(
        tiny_weights(profile), profile, workspace_bytes=1 << 30, quantization="fp8"
    )
    monkeypatch.setattr(quantization_module, "MiniMaxQuantContext", original)
    assert plan
    assert sorted(_SpyContext.seen) == sorted(quantized_names(1))
    assert build_dit_engine(tiny_weights(profile), profile, workspace_bytes=1 << 30)


@pytest.mark.gpu
@pytest.mark.trt
def test_tiny_fp8_denoiser_has_quantize_dequantize_layers(monkeypatch) -> None:
    """The quantized graph has one Quantize per quantized projection and two Dequantize."""
    import tensorrt as trt

    from .. import dit_builder
    from .test_trt_builders import _weights as tiny_weights

    profile = _tiny_profile()
    scales = {name: 0.05 for name in quantized_names(1)}
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    captured = {}
    original = dit_builder._serialize

    def spy(**kwargs):
        network = kwargs["network"]
        captured["layers"] = [network.get_layer(i).type for i in range(network.num_layers)]
        return original(**kwargs)

    monkeypatch.setattr(dit_builder, "_serialize", spy)
    dit_builder.build_dit_engine(
        tiny_weights(profile), profile, workspace_bytes=1 << 30, quantization="fp8"
    )
    kinds = captured["layers"]
    assert kinds.count(trt.LayerType.QUANTIZE) == len(QUANTIZED)
    assert kinds.count(trt.LayerType.DEQUANTIZE) == 2 * len(QUANTIZED)
