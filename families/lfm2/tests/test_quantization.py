# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2 FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import validate_dense_lfm2_config  # noqa: E402
from ..model import build_lfm2_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    Lfm2QuantContext,
    calibrate_lfm2_fp8,
    configure_fp8_build,
    load_activation_scales,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, HEADS, KV_HEADS = 512, 2048, 64, 8, 4
LAYER_TYPES = ["conv", "full_attention", "conv", "full_attention"]
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)


def _raw_config() -> dict:
    """Return a small dense LFM2 ``config.json`` for the tests."""
    return {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "block_dim": HIDDEN,
        "block_ff_dim": MLP,
        "intermediate_size": MLP,
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "layer_types": LAYER_TYPES,
        "conv_L_cache": 3,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 256,
        "conv_dim": HIDDEN,
        "conv_dim_out": HIDDEN,
    }


def _config():
    """Return the raw tiny config wrapped the way the family loader passes it."""
    return SimpleNamespace(raw=_raw_config())


def _resolved():
    """Return the validated dense LFM2 config of the tiny model."""
    return validate_dense_lfm2_config(_config())


def _weights() -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` layout of the tiny model."""
    cfg = _resolved()
    rng = np.random.RandomState(0)

    def rand(*shape: int) -> np.ndarray:
        return (rng.randn(*shape) * 0.05).astype(np.float32)

    weights = WeightDict()
    weights["embedding"] = rand(VOCAB, HIDDEN)
    for layer, layer_type in enumerate(cfg.layer_types):
        prefix = f"layer.{layer}"
        weights[f"{prefix}.operator_norm"] = np.ones(HIDDEN, dtype=np.float32)
        weights[f"{prefix}.ffn_norm"] = np.ones(HIDDEN, dtype=np.float32)
        if layer_type == "conv":
            weights[f"{prefix}.conv_in"] = rand(HIDDEN, 3 * HIDDEN)
            weights[f"{prefix}.conv_weight"] = rand(HIDDEN, 3)
            weights[f"{prefix}.conv_out"] = rand(HIDDEN, HIDDEN)
        else:
            weights[f"{prefix}.w_q"] = rand(HIDDEN, cfg.attention_size)
            weights[f"{prefix}.w_k"] = rand(HIDDEN, cfg.kv_attention_size)
            weights[f"{prefix}.w_v"] = rand(HIDDEN, cfg.kv_attention_size)
            weights[f"{prefix}.w_o"] = rand(cfg.attention_size, HIDDEN)
            weights[f"{prefix}.q_norm"] = np.ones(cfg.head_dim, dtype=np.float32)
            weights[f"{prefix}.k_norm"] = np.ones(cfg.head_dim, dtype=np.float32)
        weights[f"{prefix}.w1"] = rand(HIDDEN, MLP)
        weights[f"{prefix}.w3"] = rand(HIDDEN, MLP)
        weights[f"{prefix}.w2"] = rand(MLP, HIDDEN)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_lm_head"] = np.ascontiguousarray(weights["embedding"].T)
    weights["_layer_types"] = list(cfg.layer_types)
    weights["_num_attention_layers"] = cfg.num_attention_layers
    weights["_num_conv_layers"] = cfg.num_conv_layers
    weights["_hidden_size"] = HIDDEN
    weights["_intermediate_size"] = MLP
    weights["_head_dim"] = cfg.head_dim
    weights["_conv_l_cache"] = cfg.conv_l_cache
    return weights


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float]:
    """Return one activation scale per existing quantized projection of the tiny model."""
    weights = _weights()
    return {
        f"layer.{layer}.{stem}": 0.02
        for layer in range(len(LAYER_TYPES))
        for stem in _WEIGHT_STEMS.values()
        if stem not in skip and f"layer.{layer}.{stem}" in weights
    }


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.RandomState(0).randn(64, 128).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)

    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    weight = np.full((4, 4), 3.0, dtype=np.float32)
    packed, _ = quantize_fp8_weight(weight)
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_scale_key_identifies_model_shape_and_checkpoint() -> None:
    """The scale key names the model shape and tells checkpoints of the same shape apart."""
    base = scale_key(_resolved(), np.ones(HIDDEN, dtype=np.float32))
    assert base == f"lfm2-h{HIDDEN}-l{len(LAYER_TYPES)}-i{MLP}-n{float(HIDDEN):.4f}"
    assert base == scale_key(_resolved(), np.ones(HIDDEN, dtype=np.float32))
    assert base != scale_key(_resolved(), np.full(HIDDEN, 1.5, dtype=np.float32))


def test_missing_scale_entry_is_refused() -> None:
    """A checkpoint without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_resolved(), np.ones(HIDDEN, dtype=np.float32))


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Each committed shape has a positive scale for every projection its layers carry."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    # LFM2-350M, 700M, 1.2B, 2.6B and LFM2.5-1.2B-Instruct (same shape as LFM2-1.2B).
    shapes = sorted(key.rsplit("-n", 1)[0] for key in table)
    assert shapes == sorted(
        [
            "lfm2-h1024-l16-i4608",
            "lfm2-h1536-l16-i6912",
            "lfm2-h2048-l16-i8192",
            "lfm2-h2048-l16-i8192",
            "lfm2-h2048-l30-i10752",
        ]
    )
    for key, scales in table.items():
        assert re.fullmatch(r"lfm2-h\d+-l\d+-i\d+-n-?\d+\.\d{4}", key), key
        assert all(value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        # Every layer has the SwiGLU projections plus either conv or q/k/v projections.
        for layer in layers:
            names = {name.split(".", 2)[2] for name in scales if name.startswith(f"layer.{layer}.")}
            assert {"w1", "w2", "w3"} <= names, (key, layer)
            assert names - {"w1", "w2", "w3"} in ({"conv_in", "conv_out"}, {"w_q", "w_k", "w_v"})


def test_unsupported_projections_are_not_in_the_quantized_set() -> None:
    """The attention output projection and the depthwise conv stay unquantized."""
    assert "w_o" not in _WEIGHT_STEMS.values()
    assert "conv_weight" not in _WEIGHT_STEMS.values()


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_lfm2_fp8(_resolved(), weights)

    # conv layers: conv_in, conv_out + 3 MLP; attention layers: q, k, v + 3 MLP.
    assert len(ctx.scales) == 2 * 5 + 2 * 6
    entry = ctx.scales["layer.1.w2"]
    assert (entry.weight.out_features, entry.weight.in_features) == (HIDDEN, MLP)
    assert entry.input_scale == pytest.approx(0.02)
    conv_in = ctx.scales["layer.0.conv_in"]
    assert (conv_in.weight.out_features, conv_in.weight.in_features) == (3 * HIDDEN, HIDDEN)

    packed, scale = quantize_fp8_weight(np.asarray(weights["layer.1.w2"], dtype=np.float32).T)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.1.w_o")


def test_projection_without_a_scale_stays_unquantized(monkeypatch) -> None:
    """A projection without a scale is left unquantized."""
    monkeypatch.setattr(
        quantization,
        "load_activation_scales",
        lambda config, final_norm: _activation_scales(("w2",)),
    )
    ctx = calibrate_lfm2_fp8(_resolved(), _weights())
    assert not ctx.should_quantize("layer.0.w2")
    assert ctx.should_quantize("layer.0.w1")


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lfm2_fp8(_resolved(), weights)


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
    ctx = Lfm2QuantContext(scales={})
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_engine_builds_with_fp8_weights(monkeypatch, precision: str) -> None:
    """The quantized hybrid engine builds, keeps its I/O contract and stores FP8 weights."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_lfm2_fp8(_resolved(), _weights())

    plain = build_lfm2_engine(_config(), _weights(), 64, precision=precision)
    quantized = build_lfm2_engine(_config(), _weights(), 64, precision=precision, quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0", "present_conv_0", "conv_state_0"} <= names
    # FP8 halves the bytes of every quantized projection.
    assert len(quantized) < len(plain)


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a model shape without committed scales fails before any engine build."""
    from ..model import build

    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=Path("."),
        precision="fp16",
        max_sequence_length=64,
        tensor_parallel_size=1,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        backend="trt",
    )
    monkeypatch.setattr("families.lfm2.model._load_config", lambda model_dir: _config())
    monkeypatch.setattr("families.lfm2.model.load_lfm2_weights", lambda *a, **k: _weights())
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(request, writer=None)


def test_build_rejects_other_quantization_modes(monkeypatch) -> None:
    """Only fp8 is supported; other modes are refused loudly."""
    from ..model import build

    request = SimpleNamespace(
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_generation",
        model_dir=Path("."),
        precision="fp16",
        max_sequence_length=64,
        tensor_parallel_size=1,
        quantization="nvfp4",
        fp32_layers=None,
        mtp_seq_len=None,
    )
    monkeypatch.setattr("families.lfm2.model._load_config", lambda model_dir: _config())
    with pytest.raises(NotImplementedError, match="quantization='nvfp4'"):
        build(request, writer=None)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    pattern = calibrate_fp8_scales._PROJECTION_NAME
    stems = calibrate_fp8_scales._WEIGHT_STEM
    for module, weight in (
        ("model.layers.3.feed_forward.w2", "w2"),
        ("model.layers.3.conv.in_proj", "conv_in"),
        ("model.layers.3.conv.out_proj", "conv_out"),
        ("model.layers.3.self_attn.q_proj", "w_q"),
    ):
        match = pattern.search(module)
        assert match is not None and match.group(1) == "3"
        assert stems[match.group(2)] == weight
    assert pattern.search("model.layers.3.feed_forward.w2.base") is None
    assert pattern.search("model.layers.3.self_attn.out_proj") is None
    assert pattern.search("model.layers.3.conv.conv") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda out, n: seen.update(fetch=(out, n))
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, p, n, o: seen.update(calibrate=(d, p, n, o))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "p.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "p.jsonl", 64)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 512, None)


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"
