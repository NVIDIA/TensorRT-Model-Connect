# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLOOM FP8 self-quantization: weight quantizer, scale lookup, context build, engine build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, graph_ops, quantization  # noqa: E402
from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..model import _BloomModel, build  # noqa: E402
from ..quantization import (  # noqa: E402
    BloomQuantContext,
    calibrate_bloom_fp8,
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
_BIASES = {
    "q_bias": HIDDEN,
    "k_bias": HIDDEN,
    "v_bias": HIDDEN,
    "o_bias": HIDDEN,
    "fc1_bias": MLP,
    "fc2_bias": HIDDEN,
}


def _config() -> ModelConfig:
    """Build a small BLOOM model config for the tests."""
    return ModelConfig.create_tiny(
        "bloom",
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
    weights["embedding"] = rng.randn(VOCAB, HIDDEN).astype(np.float32)
    weights["embedding_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["embedding_norm_beta"] = np.zeros(HIDDEN, dtype=np.float32)
    for layer in range(LAYERS):
        prefix = f"layer.{layer}"
        for norm in ("input_norm", "post_attn_norm"):
            weights[f"{prefix}.{norm}"] = np.ones(HIDDEN, dtype=np.float32)
            weights[f"{prefix}.{norm}_beta"] = np.zeros(HIDDEN, dtype=np.float32)
        for stem, shape in _SHAPES.items():
            weights[f"{prefix}.{stem}"] = (rng.randn(*shape) * 0.05).astype(np.float32)
        for stem, width in _BIASES.items():
            weights[f"{prefix}.{stem}"] = np.zeros(width, dtype=np.float32)
    weights["final_norm"] = np.ones(HIDDEN, dtype=np.float32)
    weights["final_norm_beta"] = np.zeros(HIDDEN, dtype=np.float32)
    weights["w_out"] = (rng.randn(HIDDEN, VOCAB) * 0.05).astype(np.float32)
    weights["_attention_size"] = HIDDEN
    weights["_kv_attention_size"] = HIDDEN
    weights["_mlp_size"] = MLP
    return weights


def _activation_scales(skip: tuple[str, ...] = ()) -> dict[str, float]:
    """Return one activation scale per quantized projection of the tiny model."""
    return {
        f"layer.{layer}.{stem}": 0.02
        for layer in range(LAYERS)
        for stem in quantization.WEIGHT_STEMS
        if stem not in skip
    }


def _patch_scales(monkeypatch, scales: dict) -> None:
    """Make the scale lookup return ``scales`` for the tiny model."""
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, final_norm: scales)


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


def test_scale_key_identifies_checkpoint() -> None:
    """The scale key names the model config and the final norm sum of the checkpoint."""
    norm = np.ones(HIDDEN, dtype=np.float32)
    assert scale_key(_config(), norm) == f"bloom-h{HIDDEN}-l{LAYERS}-i{MLP}-n{float(HIDDEN):.4f}"
    assert scale_key(_config(), norm * 1.5) != scale_key(_config(), norm)


def test_missing_scale_entry_is_refused() -> None:
    """A model config without committed scales fails with a clear error."""
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        load_activation_scales(_config(), np.ones(HIDDEN, dtype=np.float32))


def test_committed_scales_cover_every_quantized_projection() -> None:
    """Every committed config has an entry for every quantized projection: a positive scale or null."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert all(value is None or value > 0 for value in scales.values()), key
        layers = {int(name.split(".")[1]) for name in scales}
        expected = {
            f"layer.{layer}.{stem}" for layer in layers for stem in quantization.WEIGHT_STEMS
        }
        assert set(scales) == expected, key
        # query_key_value has one input: q, k and v share one scale
        for layer in layers:
            q_scales = {scales[f"layer.{layer}.{stem}"] for stem in ("w_q", "w_k", "w_v")}
            assert len(q_scales) == 1, (key, layer)


def test_committed_scales_are_the_calibrated_bloom_sizes() -> None:
    """The file holds bloom-560m, 1b1, 1b7, 3b and 7b1, one key each, and no outlier was dropped."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    sizes = {}
    for key, scales in table.items():
        head, layers, mlp, _norm = key.split("-")[1:]
        sizes[(int(head[1:]), int(layers[1:]), int(mlp[1:]))] = scales
        assert len(scales) == int(layers[1:]) * len(quantization.WEIGHT_STEMS), key
        assert int(mlp[1:]) == 4 * int(head[1:]), key
        assert all(value is not None for value in scales.values()), key
    assert set(sizes) == {
        (1024, 24, 4096),
        (1536, 24, 6144),
        (2048, 24, 8192),
        (2560, 30, 10240),
        (4096, 30, 16384),
    }
    assert len(table) == len(sizes)  # no two sizes share a key


def test_attention_output_projection_is_not_in_the_quantized_set() -> None:
    """w_o stays unquantized (TensorRT 11.4 cannot compile FP8 Quantize after fused attention)."""
    assert "w_o" not in quantization.WEIGHT_STEMS
    assert quantization.WEIGHT_STEMS == ("w_q", "w_k", "w_v", "w_fc1", "w_fc2")


def test_context_quantizes_the_loaded_weights(monkeypatch) -> None:
    """The context quantizes the transposed loaded weights and keeps the committed scales."""
    weights = _weights()
    _patch_scales(monkeypatch, _activation_scales())
    ctx = calibrate_bloom_fp8(_config(), weights, graph_ops)

    assert len(ctx.scales) == LAYERS * len(quantization.WEIGHT_STEMS)
    entry = ctx.scales["layer.1.w_fc2"]
    in_f, out_f = _SHAPES["w_fc2"]
    assert (entry.weight.out_features, entry.weight.in_features) == (out_f, in_f)
    assert entry.input_scale == pytest.approx(0.02)

    # q, k and v are separate halves of the split fused weight: each has its own weight scale
    for stem in ("w_q", "w_k", "w_v", "w_fc1", "w_fc2"):
        source = np.asarray(weights[f"layer.1.{stem}"], dtype=np.float32).T
        packed, scale = quantize_fp8_weight(source)
        got = ctx.scales[f"layer.1.{stem}"].weight
        assert np.array_equal(got.packed, packed), stem
        assert got.weight_scale == pytest.approx(scale), stem
    assert not ctx.should_quantize("layer.0.w_o")


def test_projection_without_a_scale_is_refused(monkeypatch) -> None:
    """A projection with no entry in the scale file fails the build and names the tool."""
    _patch_scales(monkeypatch, _activation_scales(("w_fc2",)))
    with pytest.raises(ValueError, match=r"layer\.0\.w_fc2.*calibrate_fp8_scales\.py"):
        calibrate_bloom_fp8(_config(), _weights(), graph_ops)


def test_null_scale_leaves_the_projection_unquantized(monkeypatch) -> None:
    """A null entry (outlier left out by the calibration tool) is skipped on purpose."""
    scales = _activation_scales()
    scales["layer.1.w_fc2"] = None
    _patch_scales(monkeypatch, scales)
    ctx = calibrate_bloom_fp8(_config(), _weights(), graph_ops)
    assert not ctx.should_quantize("layer.1.w_fc2")
    assert ctx.should_quantize("layer.0.w_fc2")
    assert len(ctx.scales) == LAYERS * len(quantization.WEIGHT_STEMS) - 1


def test_already_quantized_checkpoint_is_refused(monkeypatch) -> None:
    """A checkpoint that carries quantization metadata or scale tensors is refused."""
    _patch_scales(monkeypatch, _activation_scales())
    weights = _weights()
    weights["layer.0.q_proj.weight_scale"] = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_bloom_fp8(_config(), weights, graph_ops)

    config = _config()
    config.raw["quantization_config"] = {"quant_method": "compressed-tensors"}
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_bloom_fp8(config, _weights(), graph_ops)


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
    ctx = BloomQuantContext(scales={}, graph_ops=graph_ops)
    config = _FakeBuilderConfig(routes)
    configure_fp8_build(config, ctx)
    assert config.build_route == expected
    assert config.profiling_verbosity is not None
    unquantized = _FakeBuilderConfig(routes)
    configure_fp8_build(unquantized, None)
    assert unquantized.build_route == "" and unquantized.profiling_verbosity is None


def _engine_info(plan: bytes):
    """Deserialize a plan and return its engine and the layer list of the inspector."""
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    info = json.loads(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    )
    return engine, info.get("Layers", [])


def _io(engine) -> dict[str, tuple]:
    """Return name -> (dtype, shape) of every I/O tensor of an engine."""
    return {
        engine.get_tensor_name(i): (
            engine.get_tensor_dtype(engine.get_tensor_name(i)),
            tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
        )
        for i in range(engine.num_io_tensors)
    }


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_engine_builds_with_fp8_weights(monkeypatch, precision: str) -> None:
    """The quantized decode and prefill engines build, keep the I/O and run FP8 projections."""
    config = _config()
    _patch_scales(monkeypatch, _activation_scales())
    ctx = calibrate_bloom_fp8(config, _weights(), graph_ops)

    for role in ("decode", "prefill"):
        config.raw["_decoder_engine_role"] = role
        plain = _BloomModel().build_engine(config, _weights(), 4, precision=precision)
        quantized = _BloomModel().build_engine(
            config, _weights(), 4, precision=precision, quant_ctx=ctx
        )
        plain_engine, plain_layers = _engine_info(plain)
        engine, layers = _engine_info(quantized)
        assert _io(engine) == _io(plain_engine)
        # 5 of 6 projections x 2 layers: FP8 halves the 16-bit projection bytes.
        assert len(quantized) < len(plain)
        fp8_layers = [layer for layer in layers if "e4m3" in json.dumps(layer).lower()]
        assert fp8_layers, "no FP8 layer in the quantized plan"
        assert not any("e4m3" in json.dumps(layer).lower() for layer in plain_layers)


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
    monkeypatch.setattr(_BloomModel, "load_weights", lambda self, model_dir, config: _weights())


def test_build_requires_calibrated_scales_for_fp8(monkeypatch) -> None:
    """quantization=fp8 on a model config without committed scales fails before any engine build."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match="no calibrated FP8 activation scales"):
        build(_build_request(), writer=None)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"quantization": "nvfp4"}, "quantization='nvfp4'"),
        ({"precision": "fp32"}, "requires precision fp16 or bf16"),
        ({"tensor_parallel_size": 2}, "does not support tensor parallelism"),
    ],
)
def test_build_refuses_unsupported_fp8_requests(monkeypatch, updates, message) -> None:
    """Other quantization modes, fp32 precision and tensor parallelism are refused loudly."""
    _patch_tiny_model(monkeypatch)
    with pytest.raises(NotImplementedError, match=message):
        build(_build_request(**updates), writer=None)


def test_loader_splits_the_interleaved_query_key_value_weight(tmp_path: Path) -> None:
    """The per-head interleaved fused weight is split into q, k and v before quantization."""
    from safetensors.numpy import save_file

    hidden, heads, ffn, vocab = 16, 4, 64, 8
    head_dim = hidden // heads
    config = ModelConfig.create_tiny(
        "bloom",
        hidden_size=hidden,
        vocab_size=vocab,
        intermediate_size=ffn,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=heads,
    )
    # row r of head h: 1000 * h + 100 * kind + r, kind 0 = q, 1 = k, 2 = v
    fused = np.zeros((3 * hidden, hidden), dtype=np.float32)
    for head in range(heads):
        for kind in range(3):
            for r in range(head_dim):
                fused[head * 3 * head_dim + kind * head_dim + r] = 1000 * head + 100 * kind + r
    prefix = "transformer.h.0"
    tensors = {
        "transformer.word_embeddings.weight": np.zeros((vocab, hidden), np.float32),
        "transformer.word_embeddings_layernorm.weight": np.ones(hidden, np.float32),
        "transformer.word_embeddings_layernorm.bias": np.zeros(hidden, np.float32),
        "transformer.ln_f.weight": np.ones(hidden, np.float32),
        "transformer.ln_f.bias": np.zeros(hidden, np.float32),
        f"{prefix}.input_layernorm.weight": np.ones(hidden, np.float32),
        f"{prefix}.input_layernorm.bias": np.zeros(hidden, np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(hidden, np.float32),
        f"{prefix}.post_attention_layernorm.bias": np.zeros(hidden, np.float32),
        f"{prefix}.self_attention.query_key_value.weight": fused,
        f"{prefix}.self_attention.query_key_value.bias": np.zeros(3 * hidden, np.float32),
        f"{prefix}.self_attention.dense.weight": np.zeros((hidden, hidden), np.float32),
        f"{prefix}.self_attention.dense.bias": np.zeros(hidden, np.float32),
        f"{prefix}.mlp.dense_h_to_4h.weight": np.zeros((ffn, hidden), np.float32),
        f"{prefix}.mlp.dense_h_to_4h.bias": np.zeros(ffn, np.float32),
        f"{prefix}.mlp.dense_4h_to_h.weight": np.zeros((hidden, ffn), np.float32),
        f"{prefix}.mlp.dense_4h_to_h.bias": np.zeros(hidden, np.float32),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    weights = _BloomModel().load_weights(str(tmp_path), config)
    for kind, stem in enumerate(("w_q", "w_k", "w_v")):
        rows = np.asarray(weights[f"layer.0.{stem}"]).T[:, 0]  # [out] after transposing back
        expected = [1000 * h + 100 * kind + r for h in range(heads) for r in range(head_dim)]
        assert rows.tolist() == expected, stem


def test_calibration_tool_maps_module_names_to_weight_stems() -> None:
    """The calibration tool maps module names to internal weight names."""
    pattern = calibrate_fp8_scales._PROJECTION_NAME
    match = pattern.search("transformer.h.3.self_attention.query_key_value")
    assert match is not None and match.group(1) == "3"
    assert calibrate_fp8_scales._WEIGHT_STEMS[match.group(2)] == ("w_q", "w_k", "w_v")
    match = pattern.search("transformer.h.12.mlp.dense_4h_to_h")
    assert match is not None and match.group(1) == "12"
    assert calibrate_fp8_scales._WEIGHT_STEMS[match.group(2)] == ("w_fc2",)
    assert pattern.search("transformer.h.3.mlp.dense_4h_to_h.base") is None
    assert pattern.search("transformer.h.3.self_attention.dense") is None
    assert pattern.search("lm_head") is None


def test_calibration_tool_finds_outliers_per_kind() -> None:
    """A scale above 20 times the median of its kind is reported, others are not."""
    scales = {f"layer.{i}.w_fc2": 1.0 for i in range(5)}
    scales["layer.5.w_fc2"] = 50.0
    scales.update({f"layer.{i}.w_fc1": 1.0 for i in range(6)})
    found = calibrate_fp8_scales.find_outliers(scales, 20.0)
    assert found == {"layer.5.w_fc2": 50.0}
    assert calibrate_fp8_scales.find_outliers(scales, 0.0) == {}


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The calibration tool command line accepts fetch and calibrate."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "fetch_prompts",
        lambda out, n, m: seen.update(fetch=(out, n, m)),
    )
    monkeypatch.setattr(
        calibrate_fp8_scales,
        "calibrate",
        lambda d, p, n, r, rep: seen.update(calibrate=(d, p, n, r, rep)),
    )
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path / "p.jsonl")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", str(tmp_path), "--prompts", "p.jsonl"]
    )
    calibrate_fp8_scales.main()
    assert seen["fetch"] == (tmp_path / "p.jsonl", 48, 12)
    assert seen["calibrate"] == (tmp_path, Path("p.jsonl"), 512, 20.0, None)


def test_render_prompt_falls_back_to_plain_text() -> None:
    """A tokenizer without a chat template gets the messages joined as plain text."""

    class _NoTemplate:
        chat_template = None

    text = calibrate_fp8_scales.render_prompt(
        _NoTemplate(), [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    )
    assert text == "a\n\nb"
