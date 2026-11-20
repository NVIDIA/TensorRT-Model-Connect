# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3 FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph_blocks, graph_ops, model, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..parallel import ParallelConfig  # noqa: E402
from ..default_decoder import build_standard_decoder_engine  # noqa: E402
from ..quantization import (  # noqa: E402
    PhiQuantContext,
    calibrate_phi_fp8,
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
_WEIGHT_STEMS = [stem for _, stem in quantization.PROJECTIONS]


def _config() -> ModelConfig:
    """Build a small Phi-3 model config for the tests."""
    return ModelConfig.create_tiny(
        "phi3",
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
        for stem in _WEIGHT_STEMS
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
    assert scale_key(_config()) == f"phi3-h{HIDDEN}-l{LAYERS}-i{MLP}-kv{HEADS}-v{VOCAB}"


def test_missing_scale_entry_is_refused() -> None:
    """A model config without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config())


def test_same_sizes_with_another_vocabulary_has_no_scales() -> None:
    """A model with the sizes of Phi-3-medium but another vocabulary (Phi-4) is not calibrated."""
    config = ModelConfig.create_tiny(
        "phi3",
        hidden_size=5120,
        vocab_size=100352,
        intermediate_size=17920,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=10,
    )
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(config)


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed model config has an entry for every quantized projection of every layer."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert {"phi3-h3072-l32-i8192-kv32-v32064", "phi3-h5120-l40-i17920-kv10-v32064"} <= set(table)
    for key, scales in table.items():
        assert all(value is None or value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {f"layer.{layer}.{stem}" for layer in layers for stem in _WEIGHT_STEMS}
        assert set(scales) == expected, key
        assert len(layers) == int(key.split("-l")[1].split("-")[0]), key


def test_attention_output_projection_is_not_in_the_quantized_set() -> None:
    """w_o stays unquantized (TensorRT 11.4 cannot compile FP8 Quantize after fused attention)."""
    assert "w_o" not in _WEIGHT_STEMS
    assert sorted(_WEIGHT_STEMS) == sorted(["w_q", "w_k", "w_v", "w_gate", "w_up", "w_down"])


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_phi_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(_WEIGHT_STEMS)
    for stem in ("w_q", "w_k", "w_v", "w_gate", "w_up", "w_down"):
        assert ctx.should_quantize(f"layer.0.{stem}")
    entry = ctx.scales["layer.1.w_down"]
    in_f, out_f = _SHAPES["w_down"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    source = np.asarray(weights["layer.1.w_down"], dtype=np.float32).T
    packed, scale = quantize_fp8_weight(source)
    assert np.array_equal(entry.weight.packed, packed)
    assert entry.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layer.0.w_o")


def test_null_scale_leaves_a_projection_unquantized(monkeypatch) -> None:
    """A null entry (massive activation) stays fp16 on purpose; the others are still quantized."""
    scales = _activation_scales()
    scales["layer.1.w_down"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: scales)
    ctx = calibrate_phi_fp8(_config(), _weights(), graph_ops)
    assert not ctx.should_quantize("layer.1.w_down")
    assert ctx.should_quantize("layer.0.w_down")
    assert len(ctx.scales) == LAYERS * len(_WEIGHT_STEMS) - 1


def test_projection_without_a_scale_is_refused(monkeypatch) -> None:
    """A projection without a scale fails the build and names the projection and the tool."""
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda config: _activation_scales(("w_down",))
    )
    with pytest.raises(ValueError, match=r"layer\.0\.w_down.*calibrate_fp8_scales\.py"):
        calibrate_phi_fp8(_config(), _weights(), graph_ops)


def test_scale_for_an_unquantized_projection_is_refused(monkeypatch) -> None:
    """A scale for the attention output projection (not quantized) points at a stale scale file."""
    scales = _activation_scales()
    scales["layer.0.w_o"] = 0.02
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: scales)
    with pytest.raises(ValueError, match=r"layer\.0\.w_o.*calibrate_fp8_scales\.py"):
        calibrate_phi_fp8(_config(), _weights(), graph_ops)


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    weights = _weights()
    weights["layer.0.qkv_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_phi_fp8(_config(), weights, graph_ops)


def test_checkpoint_with_a_quantization_config_is_refused(monkeypatch) -> None:
    """A checkpoint whose config.json carries a quantization_config is refused."""
    config = _config()
    config.raw["quantization_config"] = {"quant_method": "fp8"}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_phi_fp8(config, _weights(), graph_ops)


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
    ctx = PhiQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def test_build_route_keeps_knobs_already_set() -> None:
    """The dual-GEMM knob is appended to a build route that already carries another knob."""
    config = _FakeBuilderConfig('{"peep:match_dual_gemm": {}}')
    config.build_route = "-other:knob=off"
    quantization.append_build_route(config, "peep:match_dual_gemm")
    quantization.append_build_route(config, "peep:match_dual_gemm")
    assert config.build_route == "-other:knob=off -peep:match_dual_gemm=off"


def test_unquantized_matmul_is_the_plain_constant_matmul(monkeypatch) -> None:
    """Without a context every projection is the unchanged fp16 constant matmul."""
    calls = []
    monkeypatch.setattr(
        graph_ops, "add_matmul_rhs_constant", lambda *args, **kwargs: calls.append(args) or "plain"
    )
    matmul = graph_blocks.make_matmul_fn("net", np.float16, None)
    assert matmul("x", 4, 8, "w", "layer.0.w_q") == "plain"
    assert len(calls) == 1


def test_quantized_matmul_routes_through_the_context() -> None:
    """With a context the projection goes through ``maybe_quantized_matmul`` with its name."""
    seen = {}

    class _Ctx:
        def maybe_quantized_matmul(self, network, lhs, lhs_w, rhs_w, weights, name, dtype):
            seen["name"] = name
            return "quantized"

    matmul = graph_blocks.make_matmul_fn("net", np.float16, _Ctx())
    assert matmul("x", 4, 8, "w", "layer.0.w_gate") == "quantized"
    assert seen["name"] == "layer.0.w_gate"


def _engine_layers(plan: bytes) -> list[dict]:
    """Return the per-layer information of a plan built with DETAILED profiling."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    return [layer for layer in info.get("Layers", []) if isinstance(layer, dict)]


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(monkeypatch) -> None:
    """The quantized engine builds with the same I/O and stores FP8 (not fp16) weights."""
    config = _config()
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    ctx = calibrate_phi_fp8(config, _weights(), graph_ops)

    for role in ("decode", "prefill"):
        config.raw["_decoder_engine_role"] = role
        plain = build_standard_decoder_engine(config, _weights(), 4, precision="fp16")
        quantized = build_standard_decoder_engine(
            config, _weights(), 4, precision="fp16", quant_ctx=ctx
        )
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        plain_engine = runtime.deserialize_cuda_engine(plain)
        engine = runtime.deserialize_cuda_engine(quantized)
        assert engine is not None
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        assert names == [
            plain_engine.get_tensor_name(i) for i in range(plain_engine.num_io_tensors)
        ]
        assert {"token_id", "logits", "present_k_0"} <= set(names)
        # 6 of 7 projections x 2 layers: FP8 halves the fp16 projection bytes.
        assert len(quantized) < len(plain)
        # Every quantized projection is an FP8 layer; the plain plan has none.
        fp8_layers = [layer for layer in _engine_layers(quantized) if "e4m3" in json.dumps(layer)]
        assert fp8_layers, role
        assert not [layer for layer in _engine_layers(plain) if "e4m3" in json.dumps(layer)]


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
    monkeypatch.setattr(model._PhiModel, "load_weights", lambda self, model_dir, config, **kwargs: _weights())


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a model config without committed scales fails before any engine build."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        model.build(_build_request(), writer=None)


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
        model.build(_build_request(**updates), writer=None)


def test_build_refuses_a_prequantized_checkpoint_config(monkeypatch) -> None:
    """A config.json with a quantization_config is refused before any engine is built."""
    config = _config()
    config.raw["quantization_config"] = {"quant_method": "fp8"}
    monkeypatch.setattr(ModelConfig, "from_dir", classmethod(lambda cls, model_dir: config))
    monkeypatch.setattr(model._PhiModel, "load_weights", lambda self, model_dir, config, **kwargs: _weights())
    with pytest.raises(ValueError, match="already quantized"):
        model.build(_build_request(), writer=None)


def test_tensor_parallel_build_refuses_a_quantization_context() -> None:
    """The tensor-parallel decoder path does not quantize."""
    ctx = PhiQuantContext(scales={}, graph_ops=graph_ops)
    with pytest.raises(ValueError, match="do not support quantization"):
        model._PhiModel().build_engine(
            _config(), _weights(), 4, quant_ctx=ctx, parallel_config=ParallelConfig(tp_size=2)
        )


def test_rope_scaling_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint with rope_scaling (the family does not implement it) is refused."""
    config = _config()
    config.raw["rope_scaling"] = {"type": "longrope"}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config: _activation_scales())
    with pytest.raises(NotImplementedError, match="rope_scaling"):
        calibrate_phi_fp8(config, _weights(), graph_ops)


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    search = calibrate_fp8_scales._PROJECTION_NAME.search
    for name, stems in (
        ("self_attn.qkv_proj", ("w_q", "w_k", "w_v")),
        ("mlp.gate_up_proj", ("w_gate", "w_up")),
        ("mlp.down_proj", ("w_down",)),
    ):
        match = search(f"model.layers.11.{name}")
        assert match is not None and match.group(1) == "11"
        assert calibrate_fp8_scales._WEIGHT_STEMS[match.group(2)] == stems
    assert search("model.layers.3.self_attn.o_proj") is None  # attention output stays unquantized
    assert search("model.layers.3.mlp.down_proj.base") is None
    assert search("lm_head") is None


def test_calibration_tool_leaves_massive_activations_out() -> None:
    """A projection whose scale is far above the median of its kind is left unquantized."""
    scales = {f"layer.{i}.w_down": 1.0 for i in range(5)}
    scales["layer.2.w_down"] = 50.0
    kept, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.2.w_down"]
    assert set(kept) == {f"layer.{i}.w_down" for i in (0, 1, 3, 4)}


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_prompts", lambda out, n, web: seen.update(fetch=(out, n, web))
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
    assert seen["fetch"] == (tmp_path / "p.jsonl", 96, 128)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 1024)
