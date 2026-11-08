# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StableLM FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

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
from ..model import _StableLMModel, build  # noqa: E402
from ..quantization import (  # noqa: E402
    StableLMQuantContext,
    calibrate_stablelm_fp8,
    configure_fp8_build,
    load_activation_scales,
    quantize_fp8_weight,
    reject_quantized_checkpoint,
    scale_key,
    weights_fingerprint,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, VOCAB, LAYERS, HEADS = 512, 2048, 64, 2, 8

_SHAPES = {
    "w_q": (HIDDEN, HIDDEN),
    "w_k": (HIDDEN, HIDDEN),
    "w_v": (HIDDEN, HIDDEN),
    "w_o": (HIDDEN, HIDDEN),
    "w_gate": (HIDDEN, MLP),
    "w_up": (HIDDEN, MLP),
    "w_down": (MLP, HIDDEN),
}
_QUANTIZED_STEMS = tuple(stem for _, stems in quantization.PROJECTIONS for stem in stems)


def _config() -> ModelConfig:
    """Build a small StableLM model config for the tests."""
    return ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "stablelm",
                "vocab_size": VOCAB,
                "hidden_size": HIDDEN,
                "intermediate_size": MLP,
                "num_hidden_layers": LAYERS,
                "num_attention_heads": HEADS,
                "partial_rotary_factor": 0.25,
                "layer_norm_eps": 1e-5,
                "max_position_embeddings": 128,
            }
        )
    )


def _weights(seed: int = 0) -> WeightDict:
    """Return random weights in the loaded ``[in, out]`` layout of the tiny model."""
    rng = np.random.RandomState(seed)
    weights = WeightDict()
    weights["embedding"] = rng.randn(VOCAB, HIDDEN).astype(np.float32)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
            weights[f"{prefix}.{norm}_beta"] = np.zeros(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = (rng.randn(*shape) * 0.05).astype(np.float32)
        for bias, size in (("q_bias", HIDDEN), ("k_bias", HIDDEN), ("v_bias", HIDDEN)):
            weights[f"{prefix}.{bias}"] = np.zeros(size, dtype=np.float32)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["final_norm_beta"] = np.zeros(HIDDEN, dtype=np.float32)
    weights["w_out"] = (rng.randn(HIDDEN, VOCAB) * 0.05).astype(np.float32)
    weights["_attention_size"] = HIDDEN
    weights["_kv_attention_size"] = HIDDEN
    weights["_mlp_size"] = MLP
    return weights


def _activation_scales(skip: tuple[str, ...] = (), null: tuple[str, ...] = ()) -> dict:
    """Return one activation scale per quantized projection of the tiny model."""
    scales = {}
    for layer in range(LAYERS):
        for stem in _QUANTIZED_STEMS:
            name = f"layer.{layer}.{stem}"
            if stem in skip:
                continue
            scales[name] = None if name in null else 0.02
    return scales


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
    """The scale key names the model config and tells checkpoints of one config apart."""
    weights = _weights()
    assert scale_key(_config(), weights) == (
        f"stablelm-h{HIDDEN}-l{LAYERS}-i{MLP}-{weights_fingerprint(weights)}"
    )
    assert weights_fingerprint(weights) == weights_fingerprint(_weights())
    assert weights_fingerprint(weights) != weights_fingerprint(_weights(seed=1))


def test_missing_scale_entry_is_refused() -> None:
    """A checkpoint without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config(), _weights())


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed checkpoint has a positive scale, or null, for every quantized projection."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert all(value is None or value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        assert layers == set(range(len(layers))), key
        expected = {f"layer.{layer}.{stem}" for layer in layers for stem in _QUANTIZED_STEMS}
        assert set(scales) == expected, key
        assert re.fullmatch(rf"stablelm-h\d+-l{len(layers)}-i\d+-[0-9a-f]{{12}}", key), key
        # q, k and v read one input, and so do gate and up: ModelOpt gives them one amax.
        for layer in layers:
            for group in (("w_q", "w_k", "w_v"), ("w_gate", "w_up")):
                shared = {scales[f"layer.{layer}.{stem}"] for stem in group}
                assert len(shared) == 1, (key, layer)


def test_attention_output_projection_is_not_in_the_quantized_set() -> None:
    """w_o stays unquantized (TensorRT 11.4 cannot compile FP8 Quantize after fused attention)."""
    assert "w_o" not in _QUANTIZED_STEMS


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, weights: _activation_scales()
    )
    ctx = calibrate_stablelm_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(_QUANTIZED_STEMS)
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
    """A projection without a scale fails the build and names the projection and the tool."""
    monkeypatch.setattr(
        quantization,
        "load_activation_scales",
        lambda config, weights: _activation_scales(("w_down",)),
    )
    with pytest.raises(ValueError, match=r"layer\.0\.w_down.*calibrate_fp8_scales\.py"):
        calibrate_stablelm_fp8(_config(), _weights(), graph_ops)


def test_null_scale_leaves_the_projection_unquantized(monkeypatch) -> None:
    """A null entry is an intentional exclusion: that projection stays fp16, the rest is FP8."""
    monkeypatch.setattr(
        quantization,
        "load_activation_scales",
        lambda config, weights: _activation_scales(null=("layer.1.w_down",)),
    )
    ctx = calibrate_stablelm_fp8(_config(), _weights(), graph_ops)
    assert not ctx.should_quantize("layer.1.w_down")
    assert ctx.should_quantize("layer.0.w_down") and ctx.should_quantize("layer.1.w_q")


def test_fp32_layers_are_left_unquantized(monkeypatch) -> None:
    """Layers named in skip_layers (the fp32 layers of the build) have no FP8 projection."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, weights: _activation_scales()
    )
    ctx = calibrate_stablelm_fp8(_config(), _weights(), graph_ops, skip_layers=(1,))
    assert ctx.quantizes_layer(0) and not ctx.quantizes_layer(1)
    assert len(ctx.scales) == len(_QUANTIZED_STEMS)
    assert all(name.startswith("layer.0.") for name in ctx.scales)


def test_scale_for_a_projection_that_is_not_quantized_is_refused(monkeypatch) -> None:
    """A stored scale for w_o (not a quantized projection) means a stale or wrong scale file."""
    scales = _activation_scales()
    scales["layer.0.w_o"] = 0.02
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, weights: scales)
    with pytest.raises(ValueError, match=r"layer\.0\.w_o.*calibrate_fp8_scales\.py"):
        calibrate_stablelm_fp8(_config(), _weights(), graph_ops)


def test_already_quantized_weights_are_refused(monkeypatch) -> None:
    """A weight dict that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, weights: _activation_scales()
    )
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_stablelm_fp8(_config(), weights, graph_ops)


def test_already_quantized_checkpoint_is_refused(tmp_path: Path) -> None:
    """A checkpoint with a quantization config or FP8 scale tensors is refused before loading."""
    from safetensors.numpy import save_file

    config = ModelConfig.from_json(
        json.dumps({"model_type": "stablelm", "quantization_config": {}})
    )
    assert not config.raw["quantization_config"]
    config.raw["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="already quantized"):
        reject_quantized_checkpoint(tmp_path, config)

    save_file(
        {"model.layers.0.mlp.down_proj.weight_scale": np.ones(1, dtype=np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="already quantized"):
        reject_quantized_checkpoint(tmp_path, _config())

    save_file(
        {"model.layers.0.mlp.down_proj.weight": np.ones(1, dtype=np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    reject_quantized_checkpoint(tmp_path, _config())


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
    ctx = StableLMQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def _io(plan: bytes) -> dict[str, tuple]:
    """Return the I/O tensor names and shapes of a plan."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    return {
        engine.get_tensor_name(i): tuple(engine.get_tensor_shape(engine.get_tensor_name(i)))
        for i in range(engine.num_io_tensors)
    }


def _fp8_layers(plan: bytes) -> int:
    """Count the engine layers that carry an FP8 operand (a kernel name or an FP8 tensor)."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    layers = json.loads(info)["Layers"]
    return sum(
        1 for layer in layers if any(tag in json.dumps(layer).lower() for tag in ("e4m3", "fp8"))
    )


@pytest.mark.gpu
@pytest.mark.trt
def test_projection_dequantizes_in_fp32_and_returns_the_activation_type() -> None:
    """Dequantize and matmul run in fp32 (fp16 gives non-finite decode); the output is fp16."""
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (1, HIDDEN))
    packed, scale = quantize_fp8_weight(np.ones((MLP, HIDDEN), dtype=np.float32))
    entry = quantization._LayerScales(0.02, quantization._FP8Weight(packed, scale, MLP, HIDDEN))
    keep_alive: list = []

    output = quantization._wrap_fp8_matmul(
        network, activation, entry, graph_ops=graph_ops, keep_alive=keep_alive
    )

    layers = [network.get_layer(i) for i in range(network.num_layers)]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
    assert all(layer.get_output(0).dtype == trt.float32 for layer in dequantize)
    assert output.dtype == trt.float16
    assert len(keep_alive) == 1


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(monkeypatch) -> None:
    """The quantized decode and prefill engines build, keep the I/O, and store FP8 weights."""
    config = _config()
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, weights: _activation_scales()
    )
    ctx = calibrate_stablelm_fp8(config, _weights(), graph_ops)

    for role in ("decode", "prefill"):
        config.raw["_decoder_engine_role"] = role
        plain = _StableLMModel().build_engine(config, _weights(), 4, precision="fp16")
        quantized = _StableLMModel().build_engine(
            config, _weights(), 4, precision="fp16", quant_ctx=ctx
        )
        assert _io(quantized) == _io(plain)
        # 6 of 7 projections x 2 layers: FP8 halves the fp16 projection bytes.
        assert len(quantized) < len(plain)
        assert _fp8_layers(quantized) > 0
        assert _fp8_layers(plain) == 0


@pytest.mark.gpu
@pytest.mark.trt
def test_fp32_layer_with_fp8_projections_is_refused_and_unquantized_one_builds(monkeypatch) -> None:
    """The decode builder refuses an fp32 layer that has FP8 projections and accepts one without."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config, weights: _activation_scales()
    )
    config = _config()
    config.raw["_decoder_engine_role"] = "decode"
    config.raw["_fp32_layers"] = (1,)
    ctx = calibrate_stablelm_fp8(config, _weights(), graph_ops)
    with pytest.raises(ValueError, match=r"fp32_layers \[1\] have FP8 projections"):
        _StableLMModel().build_engine(config, _weights(), 4, precision="fp16", quant_ctx=ctx)

    ctx = calibrate_stablelm_fp8(config, _weights(), graph_ops, skip_layers=(1,))
    plan = _StableLMModel().build_engine(config, _weights(), 4, precision="fp16", quant_ctx=ctx)
    assert _fp8_layers(plan) > 0


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
        fp32_layers=(),
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
    monkeypatch.setattr(_StableLMModel, "load_weights", lambda self, model_dir, config: _weights())
    monkeypatch.setattr(quantization, "reject_quantized_checkpoint", lambda model_dir, config: None)


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a checkpoint without committed scales fails before any engine build."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(_build_request(), writer=None)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"quantization": "nvfp4"}, "quantization='nvfp4'"),
        ({"precision": "fp32"}, "requires precision fp16"),
        ({"precision": "bf16"}, "requires precision fp16"),
        ({"tensor_parallel_size": 2}, "does not support tensor parallelism"),
    ],
)
def test_build_refuses_unsupported_fp8_requests(monkeypatch, updates, message) -> None:
    """Other quantization modes, other precisions and tensor parallelism are refused loudly."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match=message):
        build(_build_request(**updates), writer=None)


def test_build_refuses_a_prequantized_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """quantization=fp8 on a checkpoint that is already quantized fails before loading weights."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "stablelm",
                "hidden_size": HIDDEN,
                "num_attention_heads": HEADS,
                "num_hidden_layers": LAYERS,
                "max_position_embeddings": 128,
                "quantization_config": {"quant_method": "fp8"},
            }
        )
    )
    with pytest.raises(ValueError, match="already quantized"):
        build(_build_request(model_dir=tmp_path), writer=None)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    pattern = calibrate_fp8_scales._PROJECTION_NAME
    match = pattern.search("model.layers.3.self_attn.q_proj")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEMS[match.group(2)] == ("w_q",)
    match = pattern.search("model.layers.3.mlp.down_proj")
    assert match is not None
    assert calibrate_fp8_scales._WEIGHT_STEMS[match.group(2)] == ("w_down",)
    assert pattern.search("model.layers.3.mlp.gate_proj.base") is None
    assert pattern.search("model.layers.3.self_attn.o_proj") is None
    assert pattern.search("lm_head") is None


def test_outlier_projection_is_split_off() -> None:
    """A scale above 20x the median of the same projection is left unquantized."""
    scales = {f"layer.{i}.w_down": 0.01 for i in range(5)}
    scales["layer.5.w_down"] = 0.5
    scales.update({f"layer.{i}.w_q": 0.01 for i in range(6)})
    kept, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.5.w_down"]
    assert len(kept) == len(scales) - 1


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "fetch_prompts",
        lambda out, n, wiki: seen.update(fetch=(out, n, wiki)),
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
    assert seen["fetch"] == (tmp_path / "p.jsonl", 128, 192)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 1024)


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"
