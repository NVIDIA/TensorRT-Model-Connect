# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect import BuildRequest  # noqa: E402

from .. import calibrate_fp8_scales, graph_ops, model, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..default_decoder import build_standard_decoder_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    InternLMQuantContext,
    calibrate_internlm_fp8,
    configure_fp8_build,
    load_activation_scales,
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
    """Build a small InternLM model config for the tests."""
    return ModelConfig.create_tiny(
        "internlm",
        hidden_size=HIDDEN,
        vocab_size=VOCAB,
        intermediate_size=MLP,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        head_dim=HIDDEN // HEADS,
    )


def _weights() -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` layout of the tiny model."""
    rng = np.random.RandomState(0)
    weights = WeightDict()
    weights["embedding"] = rng.randn(VOCAB, HIDDEN).astype(np.float16)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = rng.randn(*shape).astype(np.float16)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["w_out"] = rng.randn(HIDDEN, VOCAB).astype(np.float16)
    weights["_attention_size"] = HIDDEN
    weights["_kv_attention_size"] = HIDDEN
    weights["_mlp_size"] = MLP
    return weights


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float]:
    """Return one activation scale per quantized projection of the tiny model."""
    return {
        f"layer.{layer}.{stem}": 0.02
        for layer in range(LAYERS)
        for stem in _WEIGHT_STEMS.values()
        if stem not in skip
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


def test_scale_key_identifies_model() -> None:
    """The scale key names the model type and config."""
    assert scale_key(_config()) == f"internlm-h{HIDDEN}-l{LAYERS}-i{MLP}"


def test_missing_scale_entry_is_refused() -> None:
    """A model config without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config())


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed model config has a positive scale for every quantized projection."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert all(value is None or value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {f"layer.{layer}.{stem}" for layer in layers for stem in _WEIGHT_STEMS.values()}
        assert set(scales) == expected, key


def test_calibration_leaves_out_massive_activation_projections() -> None:
    """A scale far above the median of the same projection is split off, the others are kept."""
    scales = {f"layer.{layer}.w_down": 0.01 for layer in range(6)}
    scales["layer.2.w_down"] = 1.5
    scales.update({f"layer.{layer}.w_q": 0.03 for layer in range(6)})
    scales["layer.5.w_q"] = 0.3  # 10x the median is not an outlier
    kept, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.2.w_down"]
    assert "layer.2.w_down" not in kept and len(kept) == len(scales) - 1


def test_committed_scales_leave_out_only_massive_activation_down_projections() -> None:
    """The committed file marks exactly the massive-activation projections as null."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    for key, scales in json.loads(path.read_text(encoding="utf-8")).items():
        left_out = sorted(name for name, value in scales.items() if value is None)
        recomputed = calibrate_fp8_scales.split_outlier_scales(
            {name: value for name, value in scales.items() if value is not None}
        )[1]
        assert recomputed == [], key  # the kept scales hold no further outlier
        assert all(name.endswith(".w_down") for name in left_out), key


def test_attention_output_projection_is_not_in_the_quantized_set() -> None:
    """w_o stays unquantized (TensorRT 11.4 cannot compile FP8 Quantize after fused attention)."""
    assert "w_o" not in _WEIGHT_STEMS.values()


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_internlm_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(_WEIGHT_STEMS)
    entry = ctx.scales["layer.1.w_down"]
    in_f, out_f = _SHAPES["w_down"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = np.asarray(weights["layer.1.w_down"], dtype=np.float32).T
    packed, scale = quantize_fp8_weight(source)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.0.w_o")


def test_projection_without_a_scale_is_refused(monkeypatch) -> None:
    """A projection without a scale fails the build instead of staying silently unquantized."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _activation_scales(("w_down",))
    )
    with pytest.raises(ValueError, match=r"layer\.0\.w_down.*calibrate_fp8_scales\.py"):
        calibrate_internlm_fp8(_config(), _weights(), graph_ops)


def test_null_scale_leaves_a_projection_unquantized(monkeypatch) -> None:
    """A null entry (massive activation) is left in fp16 on purpose; the others are quantized."""
    scales = {**_activation_scales(), "layer.0.w_down": None}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: scales)
    ctx = calibrate_internlm_fp8(_config(), _weights(), graph_ops)
    assert not ctx.should_quantize("layer.0.w_down")
    assert ctx.should_quantize("layer.1.w_down") and ctx.should_quantize("layer.0.w_up")


def test_scale_for_an_unquantized_projection_is_refused(monkeypatch) -> None:
    """A stored scale for a projection this config does not quantize is an error."""
    scales = {**_activation_scales(), "layer.0.w_o": 0.02}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: scales)
    with pytest.raises(ValueError, match=r"layer\.0\.w_o"):
        calibrate_internlm_fp8(_config(), _weights(), graph_ops)


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_internlm_fp8(_config(), weights, graph_ops)


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
    ctx = InternLMQuantContext(scales={}, graph_ops=graph_ops)
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
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_internlm_fp8(config, _weights(), graph_ops)

    plain = build_standard_decoder_engine(config, _weights(), 4, precision="fp16")
    quantized = build_standard_decoder_engine(
        config, _weights(), 4, precision="fp16", quant_ctx=ctx
    )

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # 6 of 7 projections x 2 layers: FP8 halves the fp16 projection bytes.
    assert len(quantized) < len(plain)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEM[match.group(2)] == "w_down"
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.mlp.down_proj.base") is None
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.layers.3.self_attn.o_proj") is None


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda out, n: seen.update(fetch=(out, n))
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
    assert seen["fetch"] == (tmp_path / "p.jsonl", 64)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 512)


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"


@pytest.mark.gpu
@pytest.mark.trt
def test_dequantize_and_matmul_run_in_fp32() -> None:
    """Both Dequantize layers output fp32 and the result is cast back to the activation type."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (4, 64))
    weight = np.random.RandomState(0).randn(64, 64).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    entry = quantization._LayerScales(0.02, quantization._FP8Weight(packed, scale, 64, 64))
    output = quantization._wrap_fp8_matmul(
        network, activation, entry, graph_ops=graph_ops, keep_alive=[]
    )
    dequantize = [
        network.get_layer(i)
        for i in range(network.num_layers)
        if network.get_layer(i).type == trt.LayerType.DEQUANTIZE
    ]
    assert len(dequantize) == 2
    assert all(layer.get_output(0).dtype == trt.float32 for layer in dequantize)
    assert output.dtype == trt.float16


def _write_config(directory: Path, **extra) -> Path:
    """Write a tiny InternLM2 config.json and return its directory."""
    config = {
        "model_type": "internlm2",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "max_position_embeddings": 256,
        **extra,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    ("overrides", "config_extra", "error", "message"),
    [
        ({"quantization": "nvfp4"}, {}, NotImplementedError, "does not support quantization"),
        ({"precision": "fp32"}, {}, NotImplementedError, "requires precision fp16 or bf16"),
        ({"tensor_parallel_size": 2}, {}, NotImplementedError, "tensor parallelism"),
        ({}, {"quantization_config": {"quant_method": "fp8"}}, ValueError, "already quantized"),
    ],
)
def test_fp8_build_refusals(tmp_path: Path, overrides, config_extra, error, message) -> None:
    """Unsupported FP8 requests fail before any engine work, with a clear message."""
    model_dir = _write_config(tmp_path, **config_extra)
    request = BuildRequest(
        model_dir=model_dir,
        output_path=tmp_path / "x.bundle",
        family="internlm",
        task="text_generation",
        **{"precision": "fp16", "quantization": "fp8", **overrides},
    )
    with pytest.raises(error, match=message):
        model.build(request, None)  # type: ignore[arg-type]
