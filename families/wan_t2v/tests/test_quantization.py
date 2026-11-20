# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan T2V FP8 self-quantization: weight scales, the scale table, the calibration tool, the build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest
import tensorrt as trt
from safetensors.numpy import save_file

from .. import calibrate_fp8_scales, graph_ops, model, standard_dit_builder
from .. import quantization as quantization_module
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    BLOCK_PROJECTIONS,
    WanQuantContext,
    calibrate_wan_t2v,
    configure_build,
    quantize_fp8_weight,
    quantized_names,
    reject_quantized_checkpoint,
    scale_key,
)

DIM, HEADS, LAYERS, FFN, PATCHES, TEXT = 128, 4, 2, 256, 16, 8
OUT_DIM = 64


def _weights(seed: int = 0) -> dict[str, np.ndarray]:
    """A random DiT weight dict (engine layout, ``[in, out]``) with every key the engine reads."""
    rng = np.random.default_rng(seed)

    def mat(rows, cols):
        return rng.normal(0.0, 0.05, (rows, cols)).astype(np.float32)

    def vec(n, value=0.0):
        return np.full(n, value, dtype=np.float32)

    weights = {
        "scale_shift_table": rng.normal(0.0, 0.05, (1, 2, DIM)).astype(np.float32),
        "proj_out.weight": mat(DIM, OUT_DIM),
        "proj_out.bias": vec(OUT_DIM),
    }
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}"
        weights[f"{prefix}.scale_shift_table"] = rng.normal(0.0, 0.05, (1, 6, DIM)).astype(
            np.float32
        )
        for attn in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v", "to_out.0"):
                weights[f"{prefix}.{attn}.{proj}.weight"] = mat(DIM, DIM)
                weights[f"{prefix}.{attn}.{proj}.bias"] = vec(DIM)
            weights[f"{prefix}.{attn}.norm_q.weight"] = vec(DIM, 1.0)
            weights[f"{prefix}.{attn}.norm_k.weight"] = vec(DIM, 1.0)
        weights[f"{prefix}.norm2.weight"] = vec(DIM, 1.0)
        weights[f"{prefix}.norm2.bias"] = vec(DIM)
        weights[f"{prefix}.ffn.net.0.proj.weight"] = mat(DIM, FFN)
        weights[f"{prefix}.ffn.net.0.proj.bias"] = vec(FFN)
        weights[f"{prefix}.ffn.net.2.weight"] = mat(FFN, DIM)
        weights[f"{prefix}.ffn.net.2.bias"] = vec(DIM)
    return weights


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(LAYERS)}


def _calibrate(weights, quantization: str = "fp8") -> WanQuantContext:
    return calibrate_wan_t2v(weights, quantization, dim=DIM, num_layers=LAYERS, ffn_dim=FFN)


@pytest.fixture
def fake_scales(monkeypatch):
    """Use scales for the tiny shape instead of the committed table."""
    scales = _all_scales()
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *shape: scales)
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


def test_only_the_feed_forward_projections_are_quantized(fake_scales) -> None:
    """fc1 and fc2 of every block get a scale; attention, cross-attention and proj_out do not."""
    ctx = _calibrate(_weights())
    assert len(ctx.scales) == LAYERS * len(BLOCK_PROJECTIONS) == 4
    assert ctx.should_quantize("blocks.0.ffn.net.0.proj")
    assert ctx.should_quantize("blocks.1.ffn.net.2")
    for stem in ("to_q", "to_k", "to_v", "to_out.0"):
        assert not ctx.should_quantize(f"blocks.1.attn1.{stem}")
        assert not ctx.should_quantize(f"blocks.1.attn2.{stem}")
    assert not ctx.should_quantize("proj_out")
    assert not ctx.should_quantize("blocks.0.attn2.add_k_proj")
    assert ctx.has_fp8_weights


def test_projection_without_scale_stays_unquantized(monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    scales = _all_scales()
    del scales["blocks.0.ffn.net.2"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *shape: scales)
    ctx = _calibrate(_weights())
    assert not ctx.should_quantize("blocks.0.ffn.net.2")
    assert ctx.should_quantize("blocks.1.ffn.net.2")


def test_missing_weight_is_rejected(fake_scales) -> None:
    """A scale for a projection the checkpoint lacks fails loudly."""
    weights = _weights()
    del weights["blocks.1.ffn.net.0.proj.weight"]
    with pytest.raises(ValueError, match="ffn.net.0.proj.weight"):
        _calibrate(weights)


def test_invalid_activation_scale_is_rejected(monkeypatch) -> None:
    """Scales must be finite and positive."""
    scales = _all_scales()
    scales["blocks.0.ffn.net.0.proj"] = 0.0
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *shape: scales)
    with pytest.raises(ValueError, match="finite and positive"):
        _calibrate(_weights())


def test_unsupported_format_is_rejected() -> None:
    """Only fp8 is implemented."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        _calibrate(_weights(), "nvfp4")


def test_unknown_shape_has_a_clear_error() -> None:
    """A model shape with no committed scales names the shape and the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_wan_t2v(_weights(), "fp8", dim=DIM, num_layers=3, ffn_dim=FFN)


def test_already_quantized_checkpoint_is_rejected(tmp_path: Path) -> None:
    """A transformer that already stores quantization scales is not quantized a second time."""
    plain = tmp_path / "plain"
    plain.mkdir()
    save_file(
        {"blocks.0.attn1.to_q.weight": np.ones((4, 4), np.float32)},
        str(plain / "diffusion_pytorch_model.safetensors"),
    )
    reject_quantized_checkpoint(plain)

    scaled = tmp_path / "scaled"
    scaled.mkdir()
    save_file(
        {
            "blocks.0.attn1.to_q.weight": np.ones((4, 4), np.float32),
            "blocks.0.attn1.to_q.weight_scale": np.ones((1,), np.float32),
        },
        str(scaled / "diffusion_pytorch_model.safetensors"),
    )
    with pytest.raises(ValueError, match="already quantized"):
        reject_quantized_checkpoint(scaled)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model shape has a finite positive scale for each quantized projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"wan_t2v-h(\d+)-l(\d+)-i(\d+)", key)
        assert match, key
        dim, layers, ffn = (int(group) for group in match.groups())
        assert scale_key(dim, layers, ffn) == key
        assert set(scales) <= set(quantized_names(layers)), key
        assert len(scales) == len(quantized_names(layers)), key
        assert all(np.isfinite(v) and v > 0 for v in scales.values()), key


def test_fp8_build_keeps_layer_names() -> None:
    """FP8 builds ask for DETAILED profiling verbosity so that the inspector shows kernels."""
    config = SimpleNamespace(profiling_verbosity=None)
    configure_build(config)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


def test_calibration_tool_maps_module_names_to_engine_names() -> None:
    """Module paths map to engine projection names; other linears are skipped."""
    name = "blocks.12.ffn.net.2"
    assert calibrate_fp8_scales.weight_name(name) == name
    assert calibrate_fp8_scales.weight_name("blocks.3.ffn.net.0.proj") is not None
    # Experiments can record more projections.
    assert calibrate_fp8_scales.weight_name("blocks.3.attn1.to_q") is None
    assert (
        calibrate_fp8_scales.weight_name("blocks.3.attn1.to_q", ("attn1.to_q",))
        == "blocks.3.attn1.to_q"
    )
    for other in (
        "proj_out",
        "blocks.3.attn1.to_out.0",
        "blocks.3.attn2.to_out.0",
        "patch_embedding",
        "condition_embedder.time_embedder.linear_1",
        "blocks.3.attn1.to_q",
        "blocks.3.attn2.to_v",
        "blocks.3.ffn.net.2.base",
        "blocks.3.attn1.norm_q",
        "blocks.3.attn2.add_k_proj",
    ):
        assert calibrate_fp8_scales.weight_name(other) is None


def test_calibration_tool_shape_matches_the_scale_key(tmp_path: Path) -> None:
    """The tool derives the same shape key as the build from the transformer config."""
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": 12,
                "attention_head_dim": 128,
                "num_layers": 30,
                "ffn_dim": 8960,
            }
        )
    )
    assert scale_key(*calibrate_fp8_scales.shape(tmp_path)) == "wan_t2v-h1536-l30-i8960"
    assert (
        scale_key(
            model._WanT2VModel._DIT_DIM,
            model._WanT2VModel._DIT_NUM_LAYERS,
            model._WanT2VModel._DIT_FFN_DIM,
        )
        == "wan_t2v-h1536-l30-i8960"
    )


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    """The command line reaches ``calibrate`` with the parsed values."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(args=args))
    monkeypatch.setattr(
        "sys.argv",
        [
            "tool",
            "--model-dir",
            str(tmp_path),
            "--steps",
            "7",
            "--prompts",
            "3",
            "--size",
            "64x96x5",
        ],
    )
    calibrate_fp8_scales.main()
    assert seen["args"] == (tmp_path, 7, [(64, 96, 5)], 3)
    monkeypatch.setattr("sys.argv", ["tool", "--model-dir", str(tmp_path)])
    calibrate_fp8_scales.main()
    assert seen["args"][2] == [(480, 832, 81), (480, 832, 33), (480, 832, 17)]
    assert seen["args"][3] == 128


def test_calibration_prompts_are_128_distinct_texts() -> None:
    """The calibration set has 128 different prompts."""
    texts = calibrate_fp8_scales.prompts()
    assert len(texts) == len(set(texts)) == 128


def test_calibration_tool_rejects_a_malformed_size() -> None:
    """A size must look like HxWxF."""
    with pytest.raises(Exception, match="480x832x17"):
        calibrate_fp8_scales.parse_size("480x832")


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


def test_build_rejects_fp8_with_fp32_precision(tmp_path: Path) -> None:
    """FP8 builds need fp16 compute."""
    with pytest.raises(ValueError, match="fp16"):
        model.build(_request(tmp_path, precision="fp32"), Mock())


@pytest.mark.parametrize("parallel", [{"tensor_parallel_size": 4}, {"context_parallel_size": 4}])
def test_build_rejects_fp8_with_tensor_or_context_parallelism(tmp_path: Path, parallel) -> None:
    """The TP and CP builders have no FP8 path."""
    with pytest.raises(NotImplementedError, match="parallelism"):
        model.build(_request(tmp_path, **parallel), Mock())


class _RecordingNetwork:
    """Counts the layers a quantized linear adds, without building an engine."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.dequantize_types: list = []

    def __getattr__(self, name: str):
        def add(*args, **kwargs):
            self.calls.append(name)
            if name == "add_dequantize":
                self.dequantize_types.append(args[2])
            return SimpleNamespace(get_output=lambda _index: SimpleNamespace(dtype="out"))

        return add


def test_quantized_linear_is_quantize_dequantize_matmul_without_cast() -> None:
    """A quantized linear adds a weight constant, two Dequantize, one Quantize and a matmul; no cast."""
    ctx = WanQuantContext(scales={"blocks.0.ffn.net.2": 0.05})
    network = _RecordingNetwork()
    weight_in_out = np.random.default_rng(2).normal(0.0, 0.1, (32, 16)).astype(np.float32)
    ctx.maybe_quantized_matmul(
        network,
        SimpleNamespace(dtype=trt.float16),
        32,
        16,
        weight_in_out,
        "blocks.0.ffn.net.2",
    )
    assert network.calls.count("add_dequantize") == 2
    assert network.calls.count("add_quantize") == 1
    assert network.calls.count("add_matrix_multiply") == 1
    assert "add_cast" not in network.calls
    # Both Dequantize layers output fp16 (a float32 Dequantize adds a cast and bias kernel).
    assert network.dequantize_types == [trt.float16, trt.float16]
    # The FP8 constant keeps the checkpoint layout [out, in].
    assert len(ctx.keep_alive) == 1 and ctx.keep_alive[0].shape == (16, 32)


def test_quantized_linear_needs_fp16_activations() -> None:
    """The fp16 Dequantize path is built only for fp16 activations."""
    ctx = WanQuantContext(scales={"blocks.0.ffn.net.2": 0.05})
    with pytest.raises(ValueError, match="fp16"):
        ctx.maybe_quantized_matmul(
            _RecordingNetwork(),
            SimpleNamespace(dtype=trt.float32),
            4,
            4,
            np.ones((4, 4), np.float32),
            "blocks.0.ffn.net.2",
        )


def test_projection_without_scale_uses_the_plain_matmul(monkeypatch) -> None:
    """Without a scale the context adds the same constant matmul as an unquantized build."""
    calls = []
    monkeypatch.setattr(
        graph_ops, "add_matmul_rhs_constant", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    ctx = WanQuantContext(scales={"blocks.0.ffn.net.0.proj": 0.05})
    network = _RecordingNetwork()
    ctx.maybe_quantized_matmul(
        network, "x", 4, 8, np.zeros((4, 8), np.float32), "blocks.0.attn1.to_q"
    )
    assert len(calls) == 1 and network.calls == []


def _build_dit(quant_ctx, precision: str = "fp16") -> bytes:
    return standard_dit_builder.build_standard_dit_engine(
        _weights(),
        dim=DIM,
        num_heads=HEADS,
        num_layers=LAYERS,
        ffn_dim=FFN,
        context_dim=DIM,
        num_patches=PATCHES,
        text_seq_len=TEXT,
        precision=precision,
        quant_ctx=quant_ctx,
    )


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_and_is_smaller(fake_scales) -> None:
    """The tiny FP8 denoiser builds, keeps FP8 weights, and is smaller than the fp16 one."""
    plain = _build_dit(None)
    quantized = _build_dit(_calibrate(_weights()))
    assert len(quantized) < len(plain)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_requires_fp16(fake_scales) -> None:
    """The FP8 path is only built for fp16 compute."""
    with pytest.raises(ValueError, match="fp16"):
        _build_dit(_calibrate(_weights()), precision="fp32")
