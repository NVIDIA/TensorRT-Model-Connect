# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hunyuan FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph_ops, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..default_decoder import build_standard_decoder_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    HunyuanQuantContext,
    calibrate_hunyuan_fp8,
    configure_fp8_build,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, LAYERS, HEADS = 512, 2048, 32, 2, 4

_SHAPES = {
    "w_q": (HIDDEN, HIDDEN),
    "w_k": (HIDDEN, HIDDEN),
    "w_v": (HIDDEN, HIDDEN),
    "w_o": (HIDDEN, HIDDEN),
    "w_gate": (HIDDEN, MLP),
    "w_up": (HIDDEN, MLP),
    "w_down": (MLP, HIDDEN),
}
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)


def _config() -> ModelConfig:
    """Build a small Hunyuan model config for the tests."""
    return ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "hunyuan_v1_dense",
                "architectures": ["HunYuanDenseV1ForCausalLM"],
                "vocab_size": VOCAB,
                "hidden_size": HIDDEN,
                "intermediate_size": MLP,
                "num_hidden_layers": LAYERS,
                "num_attention_heads": HEADS,
                "num_key_value_heads": HEADS,
                "head_dim": HIDDEN // HEADS,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
            }
        )
    )


def _weights() -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` layout of the tiny model."""
    rng = np.random.RandomState(0)
    weights = WeightDict()
    weights["embedding"] = rng.randn(VOCAB, HIDDEN).astype(np.float16)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm", "q_norm", "k_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = rng.randn(*shape).astype(np.float16)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_out"] = rng.randn(HIDDEN, VOCAB).astype(np.float16)
    weights["_attention_size"] = HIDDEN
    weights["_kv_attention_size"] = HIDDEN
    weights["_mlp_size"] = MLP
    return weights


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float | None]:
    """Return one activation scale per projection; skipped stems are ``None`` (left in fp16)."""
    return {
        f"layer.{layer}.{stem}": (None if stem in skip else 0.02)
        for layer in range(LAYERS)
        for stem in _WEIGHT_STEMS.values()
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


def test_scale_key_identifies_model_config_and_weights() -> None:
    """The scale key names the model config and carries the final-norm weight sum."""
    ones = np.ones(HIDDEN, dtype=np.float32)
    assert (
        scale_key(_config(), ones) == f"hunyuan_v1_dense-h{HIDDEN}-l{LAYERS}-i{MLP}-n{HIDDEN}.0000"
    )
    # Two checkpoints with one config (Hy-MT2-7B and HY-MT1.5-7B) must not share scales.
    assert scale_key(_config(), ones) != scale_key(_config(), ones * 1.01)


def test_missing_scale_entry_is_refused() -> None:
    """A model config without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config(), np.ones(HIDDEN, dtype=np.float32))


def _committed_scales() -> dict:
    """Return the committed activation-scale table."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_scales_are_the_two_7b_checkpoints() -> None:
    """Scales ship for Hy-MT2-7B and HY-MT1.5-7B (one config, different final-norm sums)."""
    assert sorted(_committed_scales()) == [
        "hunyuan_v1_dense-h4096-l32-i14336-n10686.9551",
        "hunyuan_v1_dense-h4096-l32-i14336-n10918.4258",
    ]


def test_1_8b_config_has_no_scales_and_names_the_tool() -> None:
    """Hy-MT2-1.8B (hidden 2048, 32 layers) is not shipped; its FP8 build asks for calibration."""
    config = ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "hunyuan_v1_dense",
                "architectures": ["HunYuanDenseV1ForCausalLM"],
                "vocab_size": VOCAB,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "max_position_embeddings": 128,
            }
        )
    )
    final_norm = np.full(2048, 6025.0994 / 2048, dtype=np.float64)
    assert scale_key(config, final_norm) not in _committed_scales()
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        load_activation_scales(config, final_norm)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has a scale or an explicit null for every projection."""
    table = _committed_scales()
    assert table
    for key, scales in table.items():
        assert key.startswith("hunyuan_v1_dense-h"), key
        assert all(value is None or value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {f"layer.{layer}.{stem}" for layer in layers for stem in _WEIGHT_STEMS.values()}
        assert set(scales) == expected, key
        assert any(value is not None for value in scales.values()), key


def test_attention_output_projection_is_in_the_quantized_set() -> None:
    """w_o is quantized (it builds as FP8 after the native attention layer, see the pull request)."""
    assert "w_o" in _WEIGHT_STEMS.values()


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_hunyuan_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(_WEIGHT_STEMS)
    entry = ctx.scales["layer.1.w_down"]
    in_f, out_f = _SHAPES["w_down"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = np.asarray(weights["layer.1.w_down"], dtype=np.float32).T
    packed, scale = quantize_fp8_weight(source)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert ctx.should_quantize("layer.0.w_o")


def test_projection_with_a_null_scale_stays_unquantized(monkeypatch) -> None:
    """A projection stored as null (massive activation) is left in fp16 on purpose."""
    monkeypatch.setattr(
        quantization,
        "load_activation_scales",
        lambda config, final_norm: _activation_scales(("w_down",)),
    )
    ctx = calibrate_hunyuan_fp8(_config(), _weights(), graph_ops)
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.0.w_up")


def test_projection_missing_from_the_scale_file_is_an_error(monkeypatch) -> None:
    """A projection with no entry at all is an error, never a silent fp16 fallback."""
    scales = _activation_scales()
    del scales["layer.1.w_up"]
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, final_norm: scales)
    with pytest.raises(ValueError, match="layer.1.w_up has no activation scale"):
        calibrate_hunyuan_fp8(_config(), _weights(), graph_ops)


def test_scale_for_a_projection_that_is_not_quantized_is_an_error(monkeypatch) -> None:
    """A stored scale for a tensor that is not a quantized projection does not match this recipe."""
    scales = {**_activation_scales(), "layer.0.q_norm": 0.02}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, final_norm: scales)
    with pytest.raises(ValueError, match="not quantized"):
        calibrate_hunyuan_fp8(_config(), _weights(), graph_ops)


def test_projection_names_cover_seven_projections_per_layer() -> None:
    """The recipe quantizes q, k, v, o, gate, up and down of every layer."""
    names = projection_names(_config())
    assert len(names) == LAYERS * 7
    assert any(name.endswith("w_o") for name in names)


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_hunyuan_fp8(_config(), weights, graph_ops)


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
    ctx = HunyuanQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(monkeypatch) -> None:
    """The quantized engine builds and stores FP8 (not full-precision) weights."""
    config = _config()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, final_norm: _activation_scales()
    )
    ctx = calibrate_hunyuan_fp8(config, _weights(), graph_ops)

    plain = build_standard_decoder_engine(config, _weights(), 4, precision="fp16")
    quantized = build_standard_decoder_engine(
        config, _weights(), 4, precision="fp16", quant_ctx=ctx
    )

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 7 projections x 2 layers: FP8 halves the fp16 projection bytes.
    assert len(quantized) < len(plain)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEM[match.group(2)] == "w_down"
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj.base") is None
    out = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.o_proj")
    assert out is not None and calibrate_fp8_scales._WEIGHT_STEM[out.group(2)] == "w_o"
    assert (
        calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.query_layernorm")
        is None
    )


def test_calibration_tool_splits_outlier_scales() -> None:
    """A scale more than 20 times the median of its projection is left out."""
    scales = {f"layer.{i}.w_down": 0.01 for i in range(7)} | {"layer.7.w_down": 0.5}
    scales |= {f"layer.{i}.w_up": 0.01 for i in range(8)}
    kept, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.7.w_down"]
    assert "layer.7.w_down" not in kept and len(kept) == 15


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "fetch_prompts",
        lambda out, n, wiki, start: seen.update(fetch=(out, n, wiki, start)),
    )
    monkeypatch.setattr(
        calibrate_fp8_scales, "calibrate", lambda d, p, n: seen.update(calibrate=(d, p, n))
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "p.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "p.jsonl", 64, 192, 0)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 1024)


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"
