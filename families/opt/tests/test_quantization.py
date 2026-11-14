# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPT FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph_ops, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..model import _OPTModel, build  # noqa: E402
from ..quantization import (  # noqa: E402
    OptQuantContext,
    calibrate_opt_fp8,
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
    "w_fc1": (HIDDEN, MLP),
    "w_fc2": (MLP, HIDDEN),
}
_WEIGHT_STEMS = dict(quantization.PROJECTIONS)


def _config() -> ModelConfig:
    """Build a small OPT model config for the tests."""
    return ModelConfig.create_tiny(
        "opt",
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
            weights[f"{prefix}.{norm}_beta"] = np.zeros(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = rng.randn(*shape).astype(np.float16)
        for bias, width in (
            ("q_bias", HIDDEN),
            ("k_bias", HIDDEN),
            ("v_bias", HIDDEN),
            ("o_bias", HIDDEN),
            ("fc1_bias", MLP),
            ("fc2_bias", HIDDEN),
        ):
            weights[f"{prefix}.{bias}"] = np.zeros(width, dtype=np.float32)
    weights["position_embedding"] = rng.randn(128, HIDDEN).astype(np.float16)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["final_norm_beta"] = np.zeros(HIDDEN, dtype=np.float32)
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
    assert scale_key(_config()) == f"opt-h{HIDDEN}-l{LAYERS}-i{MLP}"


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
        assert all(value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {f"layer.{layer}.{stem}" for layer in layers for stem in _WEIGHT_STEMS.values()}
        assert set(scales) == expected, key


def test_attention_output_projection_is_not_in_the_quantized_set() -> None:
    """w_o stays unquantized (TensorRT 11.4 cannot compile FP8 Quantize after fused attention)."""
    assert "w_o" not in _WEIGHT_STEMS.values()


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_opt_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(_WEIGHT_STEMS)
    entry = ctx.scales["layer.1.w_fc2"]
    in_f, out_f = _SHAPES["w_fc2"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = np.asarray(weights["layer.1.w_fc2"], dtype=np.float32).T
    packed, scale = quantize_fp8_weight(source)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.0.w_o")


def test_projection_without_a_scale_is_refused(monkeypatch) -> None:
    """A projection without a scale fails the build and names the projection and the tool."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _activation_scales(("w_fc2",))
    )
    with pytest.raises(ValueError, match=r"layer\.0\.w_fc2.*calibrate_fp8_scales\.py"):
        calibrate_opt_fp8(_config(), _weights(), graph_ops)


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_opt_fp8(_config(), weights, graph_ops)


def test_checkpoint_with_a_quantization_config_is_refused(monkeypatch) -> None:
    """A checkpoint whose config.json carries a quantization_config is refused."""
    config = _config()
    config.raw["quantization_config"] = {"quant_method": "fp8"}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_opt_fp8(config, _weights(), graph_ops)


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attribute the build options touch."""

    def __init__(self) -> None:
        self.profiling_verbosity = None


def test_fp8_build_keeps_layer_names_and_unquantized_build_is_untouched() -> None:
    """A quantized build asks for DETAILED profiling; an unquantized build changes nothing."""
    ctx = OptQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig()
    configure_fp8_build(config, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    unquantized = _FakeBuilderConfig()
    configure_fp8_build(unquantized, None)
    assert unquantized.profiling_verbosity is None


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(monkeypatch) -> None:
    """The quantized engine builds and stores FP8 (not full-precision) weights."""
    config = _config()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_opt_fp8(config, _weights(), graph_ops)

    for role in ("decode", "prefill"):
        config.raw["_decoder_engine_role"] = role
        plain = _OPTModel().build_engine(config, _weights(), 4, precision="fp16")
        quantized = _OPTModel().build_engine(config, _weights(), 4, precision="fp16", quant_ctx=ctx)
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(quantized)
        assert engine is not None
        names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        assert {"token_id", "logits", "present_k_0"} <= names
        # the quantized plan lists FP8 layers, the plain plan none
        layers = engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        assert re.search("e4m3|fp8", layers, re.IGNORECASE)
        plain_engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plain)
        plain_layers = plain_engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON
        )
        assert not re.search("e4m3|fp8", plain_layers, re.IGNORECASE)
        # 5 of 6 projections x 2 layers: FP8 halves the fp16 projection bytes.
        assert len(quantized) < len(plain)


def _build_request(**updates):
    """Return a minimal build request for the build-time refusal tests."""
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
        verbose=False,
        backend="trt",
    )
    for key, value in updates.items():
        setattr(request, key, value)
    return request


def _patch_tiny_model(monkeypatch) -> None:
    """Make ``build`` read the tiny config and weights instead of a checkpoint."""
    monkeypatch.setattr(ModelConfig, "from_dir", classmethod(lambda cls, model_dir: _config()))
    monkeypatch.setattr(_OPTModel, "load_weights", lambda self, model_dir, config: _weights())


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a model config without committed scales fails before any engine build."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(_build_request(), writer=None)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"quantization": "nvfp4"}, "quantization='nvfp4'"),
        ({"precision": "fp32"}, "requires precision fp16"),
        ({"tensor_parallel_size": 2}, "does not support tensor parallelism"),
    ],
)
def test_build_refuses_unsupported_fp8_requests(monkeypatch, updates, message) -> None:
    """Other quantization modes, fp32 precision and tensor parallelism are refused loudly."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match=message):
        build(_build_request(**updates), writer=None)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    match = calibrate_fp8_scales._PROJECTION_NAME.search("model.decoder.layers.3.fc2")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEM[match.group(2)] == "w_fc2"
    assert calibrate_fp8_scales._PROJECTION_NAME.search("model.decoder.layers.3.fc2.base") is None
    assert (
        calibrate_fp8_scales._PROJECTION_NAME.search("model.decoder.layers.3.self_attn.out_proj")
        is None
    )


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
