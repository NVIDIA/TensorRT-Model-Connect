# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM3 FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

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
from ..config import validate_dense_lfm3_config  # noqa: E402
from ..model import build_lfm3_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    Lfm3QuantContext,
    calibrate_lfm3_fp8,
    configure_fp8_build,
    layer_projections,
    load_activation_scales,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, HEADS, KV_HEADS, HEAD_DIM = 512, 2048, 64, 4, 2, 128
LAYER_TYPES = ["conv", "full_attention", "conv", "sliding_attention"]
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)


def _raw_config() -> dict:
    """Return a small dense LFM3 ``config.json`` for the tests."""
    return {
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": len(LAYER_TYPES),
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "layer_types": LAYER_TYPES,
        "conv_L_cache": 7,
        "norm_eps": 1e-5,
        "max_position_embeddings": 256,
        "sliding_window": 16,
        "eos_token_id": 2,
    }


def _config():
    """Return the raw tiny config wrapped the way the family loader passes it."""
    return SimpleNamespace(raw=_raw_config())


def _resolved():
    """Return the validated dense LFM3 config of the tiny model."""
    return validate_dense_lfm3_config(_config())


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
        for norm in ("operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
        if layer_type == "conv":
            weights[f"{prefix}.conv_in"] = rand(HIDDEN, 3 * HIDDEN)
            weights[f"{prefix}.conv_weight"] = rand(HIDDEN, cfg.conv_l_cache)
            weights[f"{prefix}.conv_out"] = rand(HIDDEN, HIDDEN)
        else:
            weights[f"{prefix}.w_q"] = rand(HIDDEN, cfg.attention_size)
            weights[f"{prefix}.w_k"] = rand(HIDDEN, cfg.kv_attention_size)
            weights[f"{prefix}.w_v"] = rand(HIDDEN, cfg.kv_attention_size)
            weights[f"{prefix}.w_o"] = rand(cfg.attention_size, HIDDEN)
            weights[f"{prefix}.w_gate"] = rand(HIDDEN, HEADS)
            weights[f"{prefix}.q_norm"] = np.ones(HEAD_DIM, dtype=np.float32)
            weights[f"{prefix}.k_norm"] = np.ones(HEAD_DIM, dtype=np.float32)
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


def _activation_scales(nulls: tuple[str, ...] = ()) -> dict[str, float | None]:
    """Return one entry per quantizable projection of the tiny model; ``nulls`` are None."""
    return {
        f"layer.{layer}.{stem}": (None if stem in nulls else 0.02)
        for layer, layer_type in enumerate(LAYER_TYPES)
        for stem in layer_projections(layer_type)
    }


def _use_scales(monkeypatch, table: dict) -> None:
    """Make the context read ``table`` instead of the committed scale file."""
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, final_norm: table)


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


def test_scale_key_identifies_model_config_and_checkpoint() -> None:
    """The key names the model config and tells checkpoints of the same config apart."""
    base = scale_key(_resolved(), np.ones(HIDDEN, dtype=np.float32))
    assert base == f"lfm3-h{HIDDEN}-l{len(LAYER_TYPES)}-i{MLP}-kv{KV_HEADS}-n{float(HIDDEN):.4f}"
    assert base == scale_key(_resolved(), np.ones(HIDDEN, dtype=np.float32))
    assert base != scale_key(_resolved(), np.full(HIDDEN, 1.5, dtype=np.float32))


def test_unknown_model_config_is_refused_and_names_the_tool() -> None:
    """A checkpoint without committed scales fails with an error naming the calibration tool."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        load_activation_scales(_resolved(), np.ones(HIDDEN, dtype=np.float32))


def test_committed_scales_cover_every_quantized_projection() -> None:
    """The committed entry has a positive scale or null for every projection of every layer."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    # LFM3-1.7B: hidden 2048, 24 layers (13 conv, 6 full and 5 sliding attention).
    assert sorted(key.rsplit("-n", 1)[0] for key in table) == ["lfm3-h2048-l24-i7680-kv4"]
    for key, scales in table.items():
        assert re.fullmatch(r"lfm3-h\d+-l\d+-i\d+-kv\d+-n-?\d+\.\d{4}", key), key
        assert all(value is None or value > 0 for value in scales.values()), key
        nulls = [name for name, value in scales.items() if value is None]
        assert len(nulls) <= 0.1 * len(scales), (key, nulls)
        layers = sorted({int(name.split(".")[1]) for name in scales})
        assert layers == list(range(24)), key
        for layer in layers:
            names = {name.split(".", 2)[2] for name in scales if name.startswith(f"layer.{layer}.")}
            assert names in (
                set(layer_projections("conv")),
                set(layer_projections("full_attention")),
            ), (key, layer)


def test_unsupported_projections_are_not_in_the_quantized_set() -> None:
    """The attention output and gate projections, the depthwise conv and lm_head stay 16-bit."""
    for stem in ("w_o", "w_gate", "conv_weight", "w_lm_head", "embedding"):
        assert stem not in _WEIGHT_STEMS.values()


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    _use_scales(monkeypatch, _activation_scales())
    ctx = calibrate_lfm3_fp8(_resolved(), weights)

    assert len(ctx.scales) == sum(len(layer_projections(t)) for t in LAYER_TYPES)
    entry = ctx.scales["layer.1.w2"]
    assert (entry.weight.out_features, entry.weight.in_features) == (HIDDEN, MLP)
    assert entry.input_scale == pytest.approx(0.02)
    conv_in = ctx.scales["layer.0.conv_in"]
    assert (conv_in.weight.out_features, conv_in.weight.in_features) == (3 * HIDDEN, HIDDEN)

    packed, scale = quantize_fp8_weight(np.asarray(weights["layer.1.w2"], dtype=np.float32).T)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.1.w_gate")
    assert not ctx.should_quantize("layer.1.w_o")


def test_null_entry_keeps_the_projection_unquantized(monkeypatch) -> None:
    """A projection stored as null (outlier) is left in 16 bit."""
    _use_scales(monkeypatch, _activation_scales(nulls=("w2",)))
    ctx = calibrate_lfm3_fp8(_resolved(), _weights())
    assert not ctx.should_quantize("layer.0.w2")
    assert ctx.should_quantize("layer.0.w1")


def test_missing_scale_entry_is_refused(monkeypatch) -> None:
    """A projection with no entry at all fails instead of silently staying 16 bit."""
    table = _activation_scales()
    del table["layer.3.w_k"]
    _use_scales(monkeypatch, table)
    with pytest.raises(ValueError, match="no entry for layer.3.w_k.*calibrate_fp8_scales.py"):
        calibrate_lfm3_fp8(_resolved(), _weights())


def test_entry_for_an_unquantized_projection_is_refused(monkeypatch) -> None:
    """A scale for a projection the recipe keeps in 16 bit is an error."""
    table = _activation_scales()
    table["layer.1.w_gate"] = 0.02
    _use_scales(monkeypatch, table)
    with pytest.raises(ValueError, match="unquantized projections"):
        calibrate_lfm3_fp8(_resolved(), _weights())


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_non_positive_scale_is_refused(monkeypatch, value: float) -> None:
    """A zero, negative or non-finite scale is refused."""
    table = _activation_scales()
    table["layer.0.w1"] = value
    _use_scales(monkeypatch, table)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_lfm3_fp8(_resolved(), _weights())


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    _use_scales(monkeypatch, _activation_scales())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lfm3_fp8(_resolved(), weights)


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build options touch."""

    def __init__(self, all_build_routes: str, build_route: str = "") -> None:
        self.all_build_routes = all_build_routes
        self.build_route = build_route
        self.profiling_verbosity = None


@pytest.mark.parametrize(
    ("routes", "before", "expected"),
    [
        ('{"peep:match_dual_gemm": {}}', "", "-peep:match_dual_gemm=off"),
        ('{"peep:match_dual_gemm": {}}', "-a=on", "-a=on -peep:match_dual_gemm=off"),
        ("", "", ""),
    ],
)
def test_dual_gemm_route_is_set_only_when_listed(routes: str, before: str, expected: str) -> None:
    """The FP8 build route is applied only when the TensorRT build lists the knob."""
    config = _FakeBuilderConfig(routes, before)
    configure_fp8_build(config, Lfm3QuantContext(scales={}))
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def _fp8_layer_count(plan: bytes) -> int:
    """Count engine layers whose description mentions an FP8 tensor."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    return sum(
        1
        for layer in info.get("Layers", [])
        if re.search("fp8|e4m3", json.dumps(layer), re.IGNORECASE)
    )


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_engine_builds_with_fp8_weights(monkeypatch, precision: str) -> None:
    """The quantized hybrid engine builds, keeps its I/O contract and stores FP8 weights."""
    _use_scales(monkeypatch, _activation_scales())
    ctx = calibrate_lfm3_fp8(_resolved(), _weights())

    plain = build_lfm3_engine(_config(), _weights(), 64, precision=precision)
    quantized = build_lfm3_engine(_config(), _weights(), 64, precision=precision, quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0", "present_conv_0", "conv_state_1"} <= names
    # FP8 halves the bytes of every quantized projection.
    assert len(quantized) < 0.8 * len(plain)
    assert _fp8_layer_count(quantized) > 0


def _request(quantization_mode: str):
    """Return a minimal build request for the family ``build`` entry point."""
    return SimpleNamespace(
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
        quantization=quantization_mode,
        fp32_layers=None,
        mtp_seq_len=None,
        backend="trt",
        verbose=False,
    )


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a model config without committed scales fails before any build."""
    from ..model import build

    monkeypatch.setattr("families.lfm3.model._load_config", lambda model_dir: _config())
    monkeypatch.setattr("families.lfm3.model.load_lfm3_weights", lambda *a, **k: _weights())
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(_request("fp8"), writer=None)


def test_build_rejects_other_quantization_modes(monkeypatch) -> None:
    """Only fp8 is supported; other modes are refused loudly."""
    from ..model import build

    monkeypatch.setattr("families.lfm3.model._load_config", lambda model_dir: _config())
    with pytest.raises(NotImplementedError, match="quantization='nvfp4'"):
        build(_request("nvfp4"), writer=None)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    pattern = calibrate_fp8_scales._PROJECTION_NAME
    stems = calibrate_fp8_scales._WEIGHT_STEM
    for module, weight in (
        ("model.layers.3.feed_forward.w2", "w2"),
        ("model.layers.3.conv.in_proj", "conv_in"),
        ("model.layers.3.conv.out_proj", "conv_out"),
        ("model.layers.2.self_attn.q_proj", "w_q"),
        ("model.layers.2.self_attn.v_proj", "w_v"),
    ):
        match = pattern.search(module)
        assert match is not None and match.group(1) == module.split(".")[2]
        assert stems[match.group(2)] == weight
    assert pattern.search("model.layers.3.feed_forward.w2.base") is None
    assert pattern.search("model.layers.2.self_attn.gate_proj") is None
    assert pattern.search("model.layers.2.self_attn.o_proj") is None
    assert pattern.search("model.layers.3.conv.conv") is None


def test_calibration_tool_drops_outliers() -> None:
    """A scale above 20x the median of its kind becomes null; others are kept."""
    scales = {f"layer.{i}.w2": 1.0 for i in range(5)}
    scales["layer.5.w2"] = 25.0
    scales.update({f"layer.{i}.w1": 0.1 for i in range(6)})
    entries = calibrate_fp8_scales.drop_outliers(scales)
    assert entries["layer.5.w2"] is None
    assert entries["layer.0.w2"] == 1.0
    assert all(entries[f"layer.{i}.w1"] == 0.1 for i in range(6))


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
    assert seen["fetch"] == (tmp_path / "p.jsonl", 128)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 1024, None)


def test_render_prompt_handles_text_and_messages() -> None:
    """Text rows are used as is; chat rows without a template are joined as plain text."""

    class _NoTemplate:
        chat_template = None

    assert calibrate_fp8_scales.render_prompt(_NoTemplate(), {"text": "plain"}) == "plain"
    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(),
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
    )
    assert text == "a\n\nb"
