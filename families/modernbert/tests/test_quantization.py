# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, model as model_module, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    PROJECTIONS,
    calibrate_modernbert,
    load_activation_scales,
    module_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, INTER, LAYERS, HEADS, VOCAB = 256, 384, 3, 4, 64


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_modernbert(path: Path, seed: int = 0) -> None:
    """Write a small ModernBERT checkpoint (3 layers, one local-attention layer) and config."""
    rng = np.random.default_rng(seed)
    config = {
        "model_type": "modernbert",
        "architectures": ["ModernBertModel"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "max_position_embeddings": 64,
        "global_attn_every_n_layers": 2,
        "local_attention": 8,
        "norm_eps": 1e-5,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "embeddings.tok_embeddings.weight": _rand(rng, VOCAB, HIDDEN),
        "embeddings.norm.weight": np.ones(HIDDEN, np.float32),
        "final_norm.weight": np.ones(HIDDEN, np.float32),
    }
    for layer in range(LAYERS):
        prefix = f"layers.{layer}"
        if layer:
            tensors[f"{prefix}.attn_norm.weight"] = np.ones(HIDDEN, np.float32)
        tensors[f"{prefix}.attn.Wqkv.weight"] = _rand(rng, 3 * HIDDEN, HIDDEN)
        tensors[f"{prefix}.attn.Wo.weight"] = _rand(rng, HIDDEN, HIDDEN)
        tensors[f"{prefix}.mlp_norm.weight"] = np.ones(HIDDEN, np.float32)
        tensors[f"{prefix}.mlp.Wi.weight"] = _rand(rng, 2 * INTER, HIDDEN)
        tensors[f"{prefix}.mlp.Wo.weight"] = _rand(rng, HIDDEN, INTER)
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's config and weights as the build loads them."""
    _write_tiny_modernbert(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._ModernBertModel().load_weights(str(tmp_path), config)
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
    """Three modules per layer: Wqkv, Wi and the MLP Wo; attn.Wo stays 16-bit."""
    names = module_names(4)
    assert len(names) == 4 * len(PROJECTIONS)
    assert {"layers.3.attn.Wqkv", "layers.3.mlp.Wi", "layers.3.mlp.Wo"} <= set(names)
    assert not any(name.endswith("attn.Wo") for name in names)


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    config, weights, _ = tiny
    key = scale_key(config, weights)
    assert key.startswith(f"modernbert-h{HIDDEN}-l{LAYERS}-i{INTER}-n{HEADS}-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_modernbert(other, seed=1)
    other_weights = model_module._ModernBertModel().load_weights(str(other), config)
    assert scale_key(config, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(config, weights)


def test_committed_scales_cover_every_module() -> None:
    """Every committed entry has a positive scale, or null, for every module of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        layers = int(key.split("-l")[1].split("-")[0])
        assert set(scales) == set(module_names(layers)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key
        assert all(scales[name] is None for name in quantization.UNQUANTIZED_MODULES), key


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each part gets an FP8 weight in [in, out] layout; the parts of a module share one scale."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    ctx = calibrate_modernbert(config, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == LAYERS * len(PROJECTIONS)
    qkv = ctx.scales["layers.1.attn.Wqkv"]
    assert qkv.input_scale == pytest.approx(0.02)
    assert set(qkv.parts) == {"q", "k", "v"}
    amax = max(float(np.abs(weights[f"layer.1.w_{part}"]).max()) for part in "qkv")
    for part in "qkv":
        weight = qkv.parts[part]
        assert (weight.in_features, weight.out_features) == (HIDDEN, HIDDEN)
        assert weight.weight_scale == pytest.approx(amax / 448.0)
        packed, _ = quantize_fp8_weight(weights[f"layer.1.w_{part}"], amax / 448.0)
        assert np.array_equal(weight.packed, packed)
    wi = ctx.scales["layers.0.mlp.Wi"].parts
    assert set(wi) == {"input", "gate"}
    assert (wi["gate"].in_features, wi["gate"].out_features) == (HIDDEN, INTER)
    down = ctx.scales["layers.2.mlp.Wo"].parts["down"]
    assert (down.in_features, down.out_features) == (INTER, HIDDEN)
    assert not ctx.should_quantize("layers.0.attn.Wo")  # 16-bit, see quantization.py


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that module 16-bit on purpose."""
    config, weights, _ = tiny
    name = "layers.1.mlp.Wo"
    _patch_scales(monkeypatch, _all_scales(config, null=(name,)))
    ctx = calibrate_modernbert(config, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("layers.0.mlp.Wo")
    assert len(ctx.scales) == LAYERS * len(PROJECTIONS) - 1


def test_module_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A module missing from the file fails the build and names the calibration tool."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    del scales["layers.0.mlp.Wi"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layers\.0\.mlp\.Wi.*calibrate_fp8_scales\.py"):
        calibrate_modernbert(config, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_module_is_refused(tiny, monkeypatch) -> None:
    """A table with more layers than the model is a mismatch, not a silent partial build."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layers.9.mlp.Wi"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layers\.9\.mlp\.Wi"):
        calibrate_modernbert(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize(
    "tensor, quant_config",
    [("layers.0.mlp.Wi.weight_scale", None), (None, {"quant_method": "modelopt"})],
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


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_modernbert(config, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_modernbert(config, weights, "nvfp4", precision="fp16")


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
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="text_to_pooled_features",
        model_dir=str(model_dir),
        precision="fp16",
        max_sequence_length=32,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        tensor_parallel_size=1,
        verbose=False,
        backend="tensorrt",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def no_tokenizer(monkeypatch):
    """The tiny checkpoint has no tokenizer files."""
    monkeypatch.setattr(model_module, "_tokenizer_runtime_contract", lambda _: {})


def test_build_refuses_an_unknown_model_config(tiny, no_tokenizer) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_fp8_with_fp32_precision(tiny, no_tokenizer, monkeypatch) -> None:
    """FP8 on the fp32 engine is rejected before the checkpoint is read."""
    *_, model_dir = tiny
    monkeypatch.setattr(
        model_module._ModernBertModel,
        "load_weights",
        lambda *_: pytest.fail("FP8 with fp32 must fail before loading weights"),
    )
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision="fp32"), _Writer())


def test_build_refuses_fp8_with_tensor_parallel(tiny, no_tokenizer) -> None:
    """Tensor-parallel plans stay unquantized."""
    *_, model_dir = tiny
    with pytest.raises(ValueError, match="tensor-parallel"):
        model_module.build(_request(model_dir, tensor_parallel_size=2), _Writer())


def test_build_refuses_other_quantization(tiny, no_tokenizer) -> None:
    """A quantization other than fp8 or none is rejected."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model_module.build(_request(model_dir, quantization="nvfp4"), _Writer())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attribute the build option touches."""

    profiling_verbosity = None


def test_fp8_build_keeps_layer_details() -> None:
    """A quantized build asks for detailed profiling."""
    builder_config = _FakeBuilderConfig()
    quantization.ModernBertQuantContext(scales={}).configure_build(builder_config)
    assert builder_config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


# --- calibration tool ---------------------------------------------------------------------


def test_projection_name_matches_only_quantized_modules() -> None:
    """Only Wqkv, Wi and the MLP Wo are calibrated, with or without the ``model.`` root."""
    name = calibrate_fp8_scales.projection_name
    for stem in PROJECTIONS:
        assert name(f"layers.21.{stem}") == f"layers.21.{stem}"
        assert name(f"model.layers.3.{stem}") == f"layers.3.{stem}"
    assert name("layers.2.attn.Wo") is None
    assert name("head.dense") is None
    assert name("decoder") is None
    assert name("layers.0.mlp_norm") is None
    assert name("layers.0.mlp.Wi.input_quantizer") is None


def test_outlier_modules_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"layers.{layer}.mlp.Wo": 0.01 for layer in range(6)}
    scales.update({f"layers.{layer}.mlp.Wi": 0.02 for layer in range(6)})
    scales["layers.3.mlp.Wo"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layers.3.mlp.Wo"]
    assert table["layers.3.mlp.Wo"] is None
    assert table["layers.2.mlp.Wo"] == 0.01
    assert table["layers.3.mlp.Wi"] == 0.02


def test_recipe_keeps_first_layer_gelu_input_16bit() -> None:
    """The stored table marks the recipe's unquantized modules null, next to the outliers."""
    scales = {name: 0.02 for name in module_names(4)}
    scales["layers.2.mlp.Wo"] = 1.0
    table, excluded = calibrate_fp8_scales.apply_recipe(scales)
    assert excluded == ["layers.2.mlp.Wo"]
    assert table["layers.0.mlp.Wi"] is None and table["layers.2.mlp.Wo"] is None
    assert table["layers.1.mlp.Wi"] == 0.02


def test_passages_are_disjoint_runs_of_the_requested_lengths() -> None:
    """Passages take consecutive tokens across lines and never reuse a token."""
    lines = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert calibrate_fp8_scales.passages(lines, [2, 4, 2]) == [[1, 2], [3, 4, 5, 6], [7, 8]]
    assert calibrate_fp8_scales.passages(lines, [5, 5]) == [[1, 2, 3, 4, 5]]


def test_calibration_lengths_mix_short_and_long() -> None:
    """Short passages have 16 to 512 tokens; the long ones come last."""
    lengths = calibrate_fp8_scales.calibration_lengths(100, 3)
    assert len(lengths) == 103
    assert all(16 <= length <= 512 for length in lengths[:100])
    assert lengths[100:] == [4096] * 3
    assert lengths == calibrate_fp8_scales.calibration_lengths(100, 3)


def test_calibration_cli(monkeypatch) -> None:
    """The command line passes the model directory and passage counts."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: seen.update(c=args))
    monkeypatch.setattr(
        "sys.argv", ["x", "--model-dir", "m", "--short-tokens", "5", "--long-tokens", "1"]
    )
    calibrate_fp8_scales.main()
    assert seen["c"] == (Path("m"), 5, 1)


# --- engine -------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.trt
def test_dequantize_layers_output_fp16() -> None:
    """Both Dequantize layers of a projection output fp16, and no Cast is added."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (8, 64))
    packed, scale = quantize_fp8_weight(
        np.random.default_rng(0).standard_normal((64, 32)).astype("f")
    )
    weight = quantization._FP8Weight(packed, scale, 64, 32)
    ctx = quantization.ModernBertQuantContext(
        scales={"m": quantization._ModuleScales(0.1, {"p": weight})}
    )
    out = ctx.matmul(network, ctx.quantize_input(network, activation, "m"), "m", "p")
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in [layer.type for layer in layers]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
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
