# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion FP8 self-quantization: weight scales, scale table, calibration tool, build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

from .. import calibrate_fp8_scales, graph_ops, model
from .. import quantization as quantization_module
from ..config import ModelConfig
from ..quantization import (
    ACTIVATION_SCALES_FILENAME,
    PROJECTIONS,
    QUANTIZED,
    NemotronQuantContext,
    calibrate_nemotron,
    configure_build,
    quantize_fp8_weight,
    quantized_names,
    scale_key,
)

VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, MLP = 64, 128, 2, 4, 2, 256


def _rand(rng, *shape):
    return rng.normal(0.0, 0.05, shape).astype(np.float32)


def _write_checkpoint(path: Path) -> None:
    """A tiny random checkpoint with the tensor names of the HF model."""
    rng = np.random.default_rng(0)
    kv_hidden = KV_HEADS * (HIDDEN // HEADS)
    tensors = {"encoder.embed_tokens.weight": _rand(rng, VOCAB, HIDDEN)}
    for layer in range(LAYERS):
        p = f"encoder.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(HIDDEN, np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(HIDDEN, np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = _rand(rng, HIDDEN, HIDDEN)
        tensors[f"{p}.self_attn.k_proj.weight"] = _rand(rng, kv_hidden, HIDDEN)
        tensors[f"{p}.self_attn.v_proj.weight"] = _rand(rng, kv_hidden, HIDDEN)
        tensors[f"{p}.self_attn.o_proj.weight"] = _rand(rng, HIDDEN, HIDDEN)
        tensors[f"{p}.mlp.gate_proj.weight"] = _rand(rng, MLP, HIDDEN)
        tensors[f"{p}.mlp.up_proj.weight"] = _rand(rng, MLP, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = _rand(rng, HIDDEN, MLP)
    tensors["encoder.norm.weight"] = np.ones(HIDDEN, np.float32)
    tensors["diffusion_head.weight"] = _rand(rng, VOCAB, HIDDEN)
    config = {
        "model_type": "nemotron_labs_diffusion",
        "architectures": ["NemotronLabsDiffusionModel"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HIDDEN // HEADS,
        "mask_token_id": 100,
        "block_size": 32,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path):
    """Tiny model directory, its config and its engine weights (fp16)."""
    _write_checkpoint(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model._NemotronLabsDiffusionModel().load_weights(
        str(tmp_path), config, precision="fp16"
    )
    return tmp_path, config, weights


def _all_scales(value: float = 0.02) -> dict[str, float]:
    return {name: value for name in quantized_names(LAYERS)}


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


def test_quantized_projections_get_a_scale(tiny, fake_scales) -> None:
    """Every projection in ``QUANTIZED`` gets a scale; the output projection and head do not."""
    _, config, weights = tiny
    ctx = calibrate_nemotron(weights, config, graph_ops, "fp8")
    assert len(ctx.scales) == LAYERS * len(QUANTIZED)
    assert ctx.should_quantize("layer.1.w_k")
    assert ctx.should_quantize("layer.0.w_down")
    assert not ctx.should_quantize("layer.0.w_o")
    assert not ctx.should_quantize("w_out")
    assert ctx.has_fp8_weights


def test_projection_without_scale_stays_unquantized(tiny, monkeypatch) -> None:
    """A projection with no activation scale is left in fp16."""
    _, config, weights = tiny
    scales = _all_scales()
    del scales["layer.0.w_down"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda *args: scales)
    ctx = calibrate_nemotron(weights, config, graph_ops, "fp8")
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.1.w_down")


def test_already_quantized_checkpoint_is_rejected(tiny, fake_scales) -> None:
    """A checkpoint that already stores quantization scales is not quantized a second time."""
    _, config, weights = tiny
    weights["layer.0.w_q.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_nemotron(weights, config, graph_ops, "fp8")


def test_missing_weight_is_rejected(tiny, fake_scales) -> None:
    """A scale for a projection the checkpoint lacks fails loudly."""
    _, config, weights = tiny
    del weights["layer.1.w_v"]
    with pytest.raises(ValueError, match="layer.1.w_v"):
        calibrate_nemotron(weights, config, graph_ops, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    _, config, weights = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_nemotron(weights, config, graph_ops, "nvfp4")


def test_unknown_shape_has_a_clear_error(tiny) -> None:
    """A model shape with no committed scales names the shape and the calibration tool."""
    _, config, weights = tiny
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_nemotron(weights, config, graph_ops, "fp8")


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed model shape has a finite positive scale for each quantized projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        model_type, hidden, layers, inter = key.split("-")
        shape = SimpleNamespace(
            model_type=model_type,
            hidden_size=int(hidden[1:]),
            num_hidden_layers=int(layers[1:]),
            intermediate_size=int(inter[1:]),
        )
        assert scale_key(shape) == key
        assert set(scales) == set(quantized_names(shape.num_hidden_layers, PROJECTIONS)), key
        assert set(quantized_names(shape.num_hidden_layers, QUANTIZED)) <= set(scales), key
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
    """HF module paths of the projections map to engine names; other modules are skipped."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search
    for hf_stem, stem in calibrate_fp8_scales._WEIGHT_STEM.items():
        found = match(f"encoder.layers.12.{hf_stem}")
        assert found is not None
        assert (found.group(1), calibrate_fp8_scales._WEIGHT_STEM[found.group(2)]) == ("12", stem)
    assert set(calibrate_fp8_scales._WEIGHT_STEM.values()) == set(PROJECTIONS)
    for other in (
        "diffusion_head",
        "encoder.embed_tokens",
        "encoder.layers.3.mlp.act_fn",
        "encoder.layers.3.self_attn.q_proj.base",
    ):
        assert match(other) is None


def test_calibration_tool_cli_parses_its_options(monkeypatch, tmp_path: Path) -> None:
    """The command line reaches ``calibrate`` and ``fetch_prompts`` with the parsed values."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(cal=args))
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda *args: seen.update(fetch=args)
    )
    prompts = tmp_path / "p.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tool",
            "calibrate",
            "--model-dir",
            str(tmp_path),
            "--prompts",
            str(prompts),
            "--max-tokens",
            "64",
        ],
    )
    calibrate_fp8_scales.main()
    assert seen["cal"] == (tmp_path, prompts, 64, 24)
    monkeypatch.setattr("sys.argv", ["tool", "fetch", "--out", str(prompts), "--per-split", "3"])
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (prompts, 3)


def _request(model_dir: Path, **overrides):
    values = {
        "model_dir": str(model_dir),
        "task": "text_generation",
        "precision": "bf16",
        "quantization": "fp8",
        "dynamic_kv_cache": False,
        "max_sequence_length": 64,
        "video_num_frames": None,
        "context_parallel_size": 1,
        "tensor_parallel_size": 1,
        "fp32_layers": [],
        "mtp_seq_len": None,
        "max_batch_size": 1,
        "verbose": False,
        "backend": "trt",
        "image_height": None,
        "image_width": None,
        "audio_max_duration": None,
        "mtp_num_layers": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_rejects_unknown_quantization(tiny) -> None:
    """Quantization formats other than fp8 are refused."""
    with pytest.raises(NotImplementedError, match="nvfp4"):
        model.build(_request(tiny[0], quantization="nvfp4"), Mock())


def test_build_rejects_fp8_with_fp32_precision(tiny) -> None:
    """FP8 needs fp16 or bf16."""
    with pytest.raises(ValueError, match="fp16 or bf16"):
        model.build(_request(tiny[0], precision="fp32"), Mock())


def test_build_passes_the_context_to_both_engines(tiny, fake_scales, monkeypatch) -> None:
    """The FP8 context reaches the decoder engine and the extra (LoRA) engine."""
    seen = {}

    def fake_engine(self, config, weights, max_cache_length, **kwargs):
        seen.setdefault("engine", kwargs["quant_ctx"])
        return b"plan"

    def fake_extra(self, config, weights, max_cache_length, **kwargs):
        seen["extra"] = kwargs["quant_ctx"]
        return {}

    monkeypatch.setattr(model._NemotronLabsDiffusionModel, "build_engine", fake_engine)
    monkeypatch.setattr(model._NemotronLabsDiffusionModel, "build_extra_engines", fake_extra)
    model.build(_request(tiny[0]), Mock())
    assert isinstance(seen["engine"], NemotronQuantContext)
    assert seen["extra"] is seen["engine"]
    assert len(seen["engine"].scales) == LAYERS * len(QUANTIZED)


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
    ctx = NemotronQuantContext(scales={"layer.0.w_q": 0.05}, graph_ops=graph_ops)
    network = _RecordingNetwork()
    weight = np.random.default_rng(2).normal(0.0, 0.1, (32, 16)).astype(np.float16)
    lhs = SimpleNamespace(dtype="fp16", shape=(8, 32))
    ctx.maybe_quantized_matmul(network, lhs, 32, 16, weight, "layer.0.w_q")
    assert network.calls.count("add_dequantize") == 2
    assert network.calls.count("add_quantize") == 1
    assert network.calls.count("add_matrix_multiply") == 1
    assert network.calls[-1] == "add_cast"
    assert len(ctx.keep_alive) == 1 and ctx.keep_alive[0].shape == (32, 16)


def test_unlisted_projection_uses_the_regular_matmul(monkeypatch) -> None:
    """A projection with no scale goes through the unquantized constant matmul."""
    calls = []
    fake_ops = SimpleNamespace(add_matmul_rhs_constant=lambda *a, **k: calls.append(a) or "plain")
    ctx = NemotronQuantContext(scales={"layer.0.w_q": 0.05}, graph_ops=fake_ops)
    assert (
        ctx.maybe_quantized_matmul(Mock(), Mock(), 4, 4, np.zeros((4, 4)), "layer.0.w_o") == "plain"
    )
    assert len(calls) == 1 and not ctx.keep_alive


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_engine_builds_and_is_smaller(tiny, fake_scales) -> None:
    """The tiny FP8 decoder builds with FP8 weights and is smaller than the fp16 one."""
    _, config, weights = tiny
    ctx = calibrate_nemotron(weights, config, graph_ops, "fp8")
    nemotron = model._NemotronLabsDiffusionModel()
    fp16_plan = nemotron.build_engine(config, weights, 64, precision="fp16")
    fp8_plan = nemotron.build_engine(config, weights, 64, precision="fp16", quant_ctx=ctx)
    assert len(fp8_plan) < len(fp16_plan)
    assert len(ctx.keep_alive) == LAYERS * len(QUANTIZED)
