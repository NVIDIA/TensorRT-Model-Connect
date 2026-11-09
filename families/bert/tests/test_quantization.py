# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, cli, model as model_module, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    PROJECTIONS,
    calibrate_bert,
    load_activation_scales,
    module_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, INTER, LAYERS, HEADS, VOCAB, SEQ = 256, 512, 3, 4, 64, 32


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_bert(path: Path, seed: int = 0) -> None:
    """Write a small BERT checkpoint (``bert.`` prefix) and its config."""
    rng = np.random.default_rng(seed)
    config = {
        "model_type": "bert",
        "architectures": ["BertForMaskedLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "max_position_embeddings": SEQ,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def norm(prefix):
        return {
            f"{prefix}.weight": np.ones(HIDDEN, np.float32),
            f"{prefix}.bias": np.zeros(HIDDEN, np.float32),
        }

    def linear(prefix, out_features, in_features):
        return {
            f"{prefix}.weight": _rand(rng, out_features, in_features),
            f"{prefix}.bias": _rand(rng, out_features),
        }

    tensors = {
        "bert.embeddings.word_embeddings.weight": _rand(rng, VOCAB, HIDDEN),
        "bert.embeddings.position_embeddings.weight": _rand(rng, SEQ, HIDDEN),
        "bert.embeddings.token_type_embeddings.weight": _rand(rng, 2, HIDDEN),
        **norm("bert.embeddings.LayerNorm"),
    }
    for layer in range(LAYERS):
        prefix = f"bert.encoder.layer.{layer}"
        for name in ("query", "key", "value"):
            tensors.update(linear(f"{prefix}.attention.self.{name}", HIDDEN, HIDDEN))
        tensors.update(linear(f"{prefix}.attention.output.dense", HIDDEN, HIDDEN))
        tensors.update(norm(f"{prefix}.attention.output.LayerNorm"))
        tensors.update(linear(f"{prefix}.intermediate.dense", INTER, HIDDEN))
        tensors.update(linear(f"{prefix}.output.dense", HIDDEN, INTER))
        tensors.update(norm(f"{prefix}.output.LayerNorm"))
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's config and weights as the build loads them."""
    _write_tiny_bert(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module.load_weights(str(tmp_path), config)
    return config, weights, tmp_path


def _all_scales(config, value: float = 0.02, null: tuple[str, ...] = ()):
    return {
        name: (None if name in null else value) for name in module_names(config.num_hidden_layers)
    }


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_weight(np.full((4, 4), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_module_names_cover_every_layer_but_not_attention_output() -> None:
    """Five projections per layer; the attention output projection stays 16-bit."""
    names = module_names(4)
    assert len(names) == 4 * len(PROJECTIONS) == 20
    assert {
        "layer.3.attention.self.query",
        "layer.3.attention.self.key",
        "layer.3.attention.self.value",
        "layer.3.intermediate.dense",
        "layer.3.output.dense",
    } <= set(names)
    assert not any("attention.output" in name for name in names)


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and weight sums, so other weights get another key."""
    config, weights, _ = tiny
    key = scale_key(config, weights)
    assert key.startswith(f"bert-h{HIDDEN}-l{LAYERS}-i{INTER}-n{HEADS}-v{VOCAB}-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_bert(other, seed=1)
    other_weights = model_module.load_weights(str(other), config)
    assert scale_key(config, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(config, weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed entry has a positive scale, or null, for every projection of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        layers = int(key.split("-l")[1].split("-")[0])
        assert set(scales) == set(module_names(layers)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each projection gets its own FP8 weight in [in, out] layout and weight scale."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    ctx = calibrate_bert(config, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == LAYERS * len(PROJECTIONS)
    for stem, logical in PROJECTIONS.items():
        entry = ctx.scales[f"layer.1.{stem}"]
        matrix = weights[f"layer.1.{logical}"]
        assert entry.input_scale == pytest.approx(0.02)
        assert (entry.weight.in_features, entry.weight.out_features) == matrix.shape
        assert entry.weight.weight_scale == pytest.approx(float(np.abs(matrix).max()) / 448.0)
        assert np.array_equal(entry.weight.packed, quantize_fp8_weight(matrix)[0])
    assert ctx.scales["layer.0.intermediate.dense"].weight.out_features == INTER
    assert ctx.scales["layer.0.output.dense"].weight.in_features == INTER
    assert not ctx.should_quantize("layer.0.attention.output.dense")  # 16-bit, see quantization.py


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that projection 16-bit on purpose."""
    config, weights, _ = tiny
    name = "layer.1.output.dense"
    _patch_scales(monkeypatch, _all_scales(config, null=(name,)))
    ctx = calibrate_bert(config, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("layer.0.output.dense")
    assert len(ctx.scales) == LAYERS * len(PROJECTIONS) - 1


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A projection missing from the file fails the build and names the calibration tool."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    del scales["layer.0.intermediate.dense"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(
        ValueError, match=r"layer\.0\.intermediate\.dense.*calibrate_fp8_scales\.py"
    ):
        calibrate_bert(config, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_projection_is_refused(tiny, monkeypatch) -> None:
    """A table with more layers than the model is a mismatch, not a silent partial build."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.9.output.dense"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layer\.9\.output\.dense"):
        calibrate_bert(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan")])
def test_non_positive_scale_is_refused(tiny, monkeypatch, value) -> None:
    """A stored scale must be finite and positive."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.2.attention.self.key"] = value
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_bert(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_bert(config, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_bert(config, weights, "nvfp4", precision="fp16")


class _Writer:
    """Collects what ``build`` writes."""

    def __init__(self):
        self.files = {}

    def set_header(self, **_):
        pass

    def add_bytes(self, name, data):
        self.files[name] = data

    def add_json(self, name, data):
        self.files[name] = data


def _request(model_dir: Path, **overrides):
    values = dict(
        model_dir=model_dir,
        task="encoding",
        precision="fp16",
        max_sequence_length=SEQ,
        quantization="fp8",
    )
    values.update(overrides)
    return cli.BuildRequest(**values)


@pytest.fixture
def no_tokenizer(monkeypatch):
    """The tiny checkpoint has no tokenizer files."""
    monkeypatch.setattr(model_module, "_detect_tokenizer_frame", lambda _: ([], []))
    monkeypatch.setattr(model_module, "_ensure_tokenizer_json", lambda _: None)


def test_request_accepts_fp8_and_none_only(tmp_path) -> None:
    """The typed request takes fp8 or None; the CLI spelling ``none`` maps to None."""
    assert cli.BuildRequest(tmp_path, quantization="fp8").quantization == "fp8"
    assert cli.BuildRequest(tmp_path).quantization is None
    with pytest.raises(ValueError, match="quantization"):
        cli.BuildRequest(tmp_path, quantization="nvfp4")


@pytest.mark.parametrize(
    "tensor, quant_config",
    [
        ("bert.encoder.layer.0.output.dense.weight_scale", None),
        (None, {"quant_method": "modelopt"}),
    ],
    ids=["scale-tensor", "quantization-config"],
)
def test_already_quantized_checkpoint_is_refused(tiny, no_tokenizer, tensor, quant_config) -> None:
    """A checkpoint that carries FP8 scales or a quantization config is refused before loading."""
    *_, model_dir = tiny
    if tensor is not None:
        from safetensors.numpy import load_file

        tensors = load_file(str(model_dir / "model.safetensors"))
        tensors[tensor] = np.ones(1, dtype=np.float32)
        save_file(tensors, str(model_dir / "model.safetensors"))
    if quant_config is not None:
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        raw["quantization_config"] = quant_config
        (model_dir / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_an_unknown_model_config(tiny, no_tokenizer) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_build_refuses_fp8_without_fp16(tiny, no_tokenizer, monkeypatch, precision) -> None:
    """FP8 on the fp32 or bf16 engine is rejected before the checkpoint is read."""
    *_, model_dir = tiny
    monkeypatch.setattr(
        model_module,
        "load_weights",
        lambda *_: pytest.fail("FP8 without fp16 must fail before loading weights"),
    )
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision=precision), _Writer())


def test_build_refuses_fp8_with_tensor_parallel(tiny, no_tokenizer) -> None:
    """Tensor-parallel plans stay unquantized."""
    *_, model_dir = tiny
    with pytest.raises(ValueError, match="tensor.parallel"):
        model_module.build(_request(model_dir, tensor_parallel_size=2), _Writer())


def test_build_refuses_fp8_with_fp32_layers(tiny, no_tokenizer) -> None:
    """Mixed fp32 layers would feed fp32 activations to the fp16 Q/DQ path."""
    *_, model_dir = tiny
    with pytest.raises(ValueError, match="fp32_layers"):
        model_module.build(_request(model_dir, fp32_layers=(1,)), _Writer())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attribute the build option touches."""

    profiling_verbosity = None


def test_fp8_build_keeps_layer_details() -> None:
    """A quantized build asks for detailed profiling."""
    builder_config = _FakeBuilderConfig()
    quantization.BertQuantContext(scales={}).configure_build(builder_config)
    assert builder_config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


# --- calibration tool ---------------------------------------------------------------------


def test_projection_name_matches_only_quantized_projections() -> None:
    """Only q/k/v and the two feed-forward projections are calibrated, under any model root."""
    name = calibrate_fp8_scales.projection_name
    for stem in PROJECTIONS:
        assert name(f"encoder.layer.11.{stem}") == f"layer.11.{stem}"
        assert name(f"bert.encoder.layer.3.{stem}") == f"layer.3.{stem}"
        assert name(f"model.bert.encoder.layer.0.{stem}") == f"layer.0.{stem}"
    assert name("encoder.layer.2.attention.output.dense") is None
    assert name("pooler.dense") is None
    assert name("cls.predictions.transform.dense") is None
    assert name("encoder.layer.0.output.LayerNorm") is None
    assert name("encoder.layer.0.output.dense.input_quantizer") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"layer.{layer}.output.dense": 0.01 for layer in range(6)}
    scales.update({f"layer.{layer}.intermediate.dense": 0.02 for layer in range(6)})
    scales["layer.3.output.dense"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.3.output.dense"]
    assert table["layer.3.output.dense"] is None
    assert table["layer.2.output.dense"] == 0.01
    assert table["layer.3.intermediate.dense"] == 0.02


def test_passages_are_disjoint_runs_of_the_requested_lengths() -> None:
    """Passages take consecutive tokens across lines and never reuse a token."""
    lines = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert calibrate_fp8_scales.passages(lines, [2, 4, 2]) == [[1, 2], [3, 4, 5, 6], [7, 8]]
    assert calibrate_fp8_scales.passages(lines, [5, 5]) == [[1, 2, 3, 4, 5]]


def test_calibration_lengths_mix_short_and_long() -> None:
    """Short passages have 16 to the longest body; the long ones come last."""
    lengths = calibrate_fp8_scales.calibration_lengths(100, 3, 510)
    assert len(lengths) == 103
    assert all(16 <= length <= 510 for length in lengths[:100])
    assert lengths[100:] == [510] * 3
    assert lengths == calibrate_fp8_scales.calibration_lengths(100, 3, 510)


def test_calibration_cli(monkeypatch) -> None:
    """The command line passes the model directory and passage counts."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(c=args))
    monkeypatch.setattr(
        "sys.argv", ["x", "--model-dir", "m", "--short-passages", "5", "--long-passages", "1"]
    )
    calibrate_fp8_scales.main()
    assert seen["c"] == (Path("m"), 5, 1)


# --- engine -------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.trt
def test_layer_shares_one_quantize_for_query_key_value(tiny, monkeypatch) -> None:
    """One layer has three Quantize layers (q/k/v shared, fc1, fc2), all Dequantize fp16."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.0.attention.self.key"] = 0.03  # the shared input takes the largest scale
    _patch_scales(monkeypatch, scales)
    ctx = calibrate_bert(config, weights, "fp8", precision="fp16")
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    hidden = network.add_input("x", trt.float16, (SEQ, HIDDEN))
    mask = network.add_input("m", trt.float16, (1, 1, SEQ))
    out = model_module._add_encoder_layer(
        network=network,
        hidden=hidden,
        weights=weights,
        prefix="layer.0",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_heads=HEADS,
        head_dim=HIDDEN // HEADS,
        seq_length=SEQ,
        attn_mask=mask,
        eps=1e-12,
        dtype=np.float16,
        quant_ctx=ctx,
    )
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    quantize = [layer for layer in layers if layer.type == trt.LayerType.QUANTIZE]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(quantize) == 3
    assert len(dequantize) == 3 + len(PROJECTIONS)
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert out.dtype == trt.float16


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, no_tokenizer, monkeypatch) -> None:
    """The quantized engine builds from the real build path, stores FP8 weights and is smaller."""
    config, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    plain, quantized = _Writer(), _Writer()
    model_module.build(_request(model_dir, quantization=None), plain)
    model_module.build(_request(model_dir), quantized)

    plain_plan, fp8_plan = plain.files["engine.plan"], quantized.files["engine.plan"]
    assert len(fp8_plan) < 0.8 * len(plain_plan)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(fp8_plan)
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "input_ids",
        "attention_mask",
        "hidden_states",
    }
