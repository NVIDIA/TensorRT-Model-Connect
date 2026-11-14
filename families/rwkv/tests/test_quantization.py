# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RWKV FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")
if not hasattr(trt, "Builder"):  # another test module put a stand-in module in sys.modules
    pytest.skip("needs TensorRT", allow_module_level=True)

from .. import calibrate_fp8_scales, model as model_module, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    PROJECTIONS,
    calibrate_rwkv,
    load_activation_scales,
    module_names,
    quantize_fp8_weight,
    quantized_layers,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, INTER, LAYERS, VOCAB = 256, 1024, 3, 64


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_rwkv(path: Path, seed: int = 0, extra: dict | None = None) -> None:
    """Write a small RWKV-4 checkpoint (``rwkv.`` prefix, ``pytorch_model.bin``) and its config."""
    rng = np.random.default_rng(seed)
    config = {
        "model_type": "rwkv",
        "architectures": ["RwkvForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "attention_hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "context_length": 64,
        "layer_norm_epsilon": 1e-5,
        "rescale_every": 0,
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def norm(prefix):
        return {
            f"{prefix}.weight": np.ones(HIDDEN, np.float32),
            f"{prefix}.bias": np.zeros(HIDDEN, np.float32),
        }

    def mix():
        return rng.uniform(0.0, 1.0, (1, 1, HIDDEN)).astype(np.float32)

    tensors = {"rwkv.embeddings.weight": _rand(rng, VOCAB, HIDDEN)}
    tensors.update(norm("rwkv.blocks.0.pre_ln"))
    for layer in range(LAYERS):
        prefix = f"rwkv.blocks.{layer}"
        tensors.update(norm(f"{prefix}.ln1"))
        tensors.update(norm(f"{prefix}.ln2"))
        tensors[f"{prefix}.attention.time_decay"] = _rand(rng, HIDDEN)
        tensors[f"{prefix}.attention.time_first"] = _rand(rng, HIDDEN)
        for name in ("key", "value", "receptance"):
            tensors[f"{prefix}.attention.time_mix_{name}"] = mix()
        for name in ("key", "value", "receptance", "output"):
            tensors[f"{prefix}.attention.{name}.weight"] = _rand(rng, HIDDEN, HIDDEN)
        tensors[f"{prefix}.feed_forward.time_mix_key"] = mix()
        tensors[f"{prefix}.feed_forward.time_mix_receptance"] = mix()
        tensors[f"{prefix}.feed_forward.key.weight"] = _rand(rng, INTER, HIDDEN)
        tensors[f"{prefix}.feed_forward.receptance.weight"] = _rand(rng, HIDDEN, HIDDEN)
        tensors[f"{prefix}.feed_forward.value.weight"] = _rand(rng, HIDDEN, INTER)
    tensors.update(norm("rwkv.ln_out"))
    tensors["head.weight"] = _rand(rng, VOCAB, HIDDEN)
    tensors.update(extra or {})
    torch.save(
        {name: torch.from_numpy(value) for name, value in tensors.items()},
        path / "pytorch_model.bin",
    )


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's config and weights as the build loads them."""
    _write_tiny_rwkv(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._RwkvModel().load_weights(str(tmp_path), config)
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


def test_module_names_cover_every_layer_but_the_last() -> None:
    """Seven projections per layer; every projection of the last layer stays 16-bit."""
    names = module_names(4)
    assert len(names) == 3 * len(PROJECTIONS) == 21
    assert {f"layer.2.{stem}" for stem in PROJECTIONS} <= set(names)
    assert not any(name.startswith("layer.3.") for name in names)
    assert list(quantized_layers(4)) == [0, 1, 2]


def test_single_layer_model_is_refused() -> None:
    """A one-layer model has no layer left to quantize."""
    with pytest.raises(ValueError, match="at least 2 layers"):
        module_names(1)


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and weight sums, so other weights get another key."""
    config, weights, _ = tiny
    key = scale_key(config, weights)
    assert key.startswith(f"rwkv-h{HIDDEN}-l{LAYERS}-i{INTER}-v{VOCAB}-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_rwkv(other, seed=1)
    other_weights = model_module._RwkvModel().load_weights(str(other), config)
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
    ctx = calibrate_rwkv(config, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == (LAYERS - 1) * len(PROJECTIONS)
    for stem, logical in PROJECTIONS.items():
        entry = ctx.scales[f"layer.1.{stem}"]
        matrix = weights[f"layer.1.{logical}"]
        assert entry.input_scale == pytest.approx(0.02)
        assert (entry.weight.in_features, entry.weight.out_features) == matrix.shape
        assert entry.weight.weight_scale == pytest.approx(float(np.abs(matrix).max()) / 448.0)
        assert np.array_equal(entry.weight.packed, quantize_fp8_weight(matrix)[0])
    assert ctx.scales["layer.0.feed_forward.key"].weight.out_features == INTER
    assert ctx.scales["layer.0.feed_forward.value"].weight.in_features == INTER
    assert not any(ctx.should_quantize(f"layer.{LAYERS - 1}.{stem}") for stem in PROJECTIONS)


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that projection 16-bit on purpose."""
    config, weights, _ = tiny
    name = "layer.1.feed_forward.value"
    _patch_scales(monkeypatch, _all_scales(config, null=(name,)))
    ctx = calibrate_rwkv(config, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("layer.0.feed_forward.value")
    assert len(ctx.scales) == (LAYERS - 1) * len(PROJECTIONS) - 1


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A projection missing from the file fails the build and names the calibration tool."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    del scales["layer.0.attention.output"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layer\.0\.attention\.output.*calibrate_fp8_scales\.py"):
        calibrate_rwkv(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize(
    "name", ["layer.9.feed_forward.value", f"layer.{LAYERS - 1}.attention.key"]
)
def test_scale_for_an_unquantized_projection_is_refused(tiny, monkeypatch, name) -> None:
    """A table for another model config (more layers, or the last layer) is a mismatch."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales[name] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=name.replace(".", r"\.")):
        calibrate_rwkv(config, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan")])
def test_non_positive_scale_is_refused(tiny, monkeypatch, value) -> None:
    """A stored scale must be finite and positive."""
    config, weights, _ = tiny
    scales = _all_scales(config)
    scales["layer.1.attention.key"] = value
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_rwkv(config, weights, "fp8", precision="fp16")


def test_fp8_needs_fp16_precision(tiny, monkeypatch) -> None:
    """FP8 is built on the fp16 engine only."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_rwkv(config, weights, "fp8", precision="fp32")


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    config, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_rwkv(config, weights, "nvfp4", precision="fp16")


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
        task="text_generation",
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


@pytest.mark.parametrize("sign", ["scale-tensor", "quantization-config", "hf-quant-config"])
def test_already_quantized_checkpoint_is_refused(tmp_path, monkeypatch, sign) -> None:
    """A checkpoint that carries FP8 scales or a quantization config is refused before loading."""
    extra = None
    if sign == "scale-tensor":
        extra = {"rwkv.blocks.0.feed_forward.key.weight_scale": np.ones(1, np.float32)}
    _write_tiny_rwkv(tmp_path, extra=extra)
    if sign == "quantization-config":
        raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        raw["quantization_config"] = {"quant_method": "modelopt"}
        (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    if sign == "hf-quant-config":
        (tmp_path / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        model_module._RwkvModel,
        "load_weights",
        lambda *_: pytest.fail("a quantized checkpoint must be refused before loading weights"),
    )
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(tmp_path), _Writer())


def test_build_refuses_an_unknown_model_config(tiny) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_fp8_without_fp16(tiny, monkeypatch) -> None:
    """FP8 on the fp32 engine is rejected before the checkpoint is read."""
    *_, model_dir = tiny
    monkeypatch.setattr(
        model_module._RwkvModel,
        "load_weights",
        lambda *_: pytest.fail("FP8 without fp16 must fail before loading weights"),
    )
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision="fp32"), _Writer())


def test_build_refuses_fp8_with_tensor_parallel(tiny) -> None:
    """Tensor-parallel plans stay unquantized."""
    *_, model_dir = tiny
    with pytest.raises(ValueError, match="tensor-parallel"):
        model_module.build(_request(model_dir, tensor_parallel_size=2), _Writer())


def test_build_refuses_other_quantization(tiny) -> None:
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
    quantization.RwkvQuantContext(scales={}).configure_build(builder_config)
    assert builder_config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


# --- calibration tool ---------------------------------------------------------------------


def test_projection_name_matches_only_the_projections() -> None:
    """Only the seven block projections are calibrated, under the rwkv root or without it."""
    name = calibrate_fp8_scales.projection_name
    for stem in PROJECTIONS:
        assert name(f"rwkv.blocks.11.{stem}") == f"layer.11.{stem}"
        assert name(f"blocks.3.{stem}") == f"layer.3.{stem}"
        assert name(f"backbone.blocks.0.{stem}") is None
    assert name("head") is None
    assert name("rwkv.blocks.0.ln1") is None
    assert name("rwkv.blocks.0.attention") is None
    assert name("rwkv.blocks.0.feed_forward.key.input_quantizer") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"layer.{layer}.feed_forward.value": 0.1 for layer in range(6)}
    scales.update({f"layer.{layer}.attention.output": 0.02 for layer in range(6)})
    scales["layer.3.feed_forward.value"] = 5.0
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer.3.feed_forward.value"]
    assert table["layer.3.feed_forward.value"] is None
    assert table["layer.2.feed_forward.value"] == 0.1
    assert table["layer.3.attention.output"] == 0.02


def test_passages_are_disjoint_runs_of_the_requested_lengths() -> None:
    """Passages take consecutive tokens across lines and never reuse a token."""
    lines = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert calibrate_fp8_scales.passages(lines, [2, 4, 2]) == [[1, 2], [3, 4, 5, 6], [7, 8]]
    assert calibrate_fp8_scales.passages(lines, [5, 5]) == [[1, 2, 3, 4, 5]]


def test_calibration_lengths_mix_short_and_long() -> None:
    """Short passages have 16 to the context length; the long ones come last."""
    lengths = calibrate_fp8_scales.calibration_lengths(100, 3, 1024)
    assert len(lengths) == 103
    assert all(16 <= length <= 1024 for length in lengths[:100])
    assert lengths[100:] == [1024] * 3
    assert lengths == calibrate_fp8_scales.calibration_lengths(100, 3, 1024)


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


def _layer_network(weights, ctx, layer: int):
    """One RWKV layer in an fp16 network; returns the network."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    def inp(name, dtype):
        return network.add_input(name, dtype, (1, HIDDEN))

    one = network.add_constant((1, 1), trt.Weights(np.ones(1, np.float16))).get_output(0)
    eps = network.add_constant((1, 1), trt.Weights(np.full(1, 1e-5, np.float16))).get_output(0)
    model_module._add_rwkv_layer(
        network=network,
        hidden=inp("h", trt.float16),
        attn_state_in=inp("a", trt.float16),
        ff_state_in=inp("f", trt.float16),
        num_state_in=inp("n", trt.float32),
        den_state_in=inp("d", trt.float32),
        max_state_in=inp("m", trt.float32),
        eps_tensor=eps,
        one_const=one,
        weights=weights,
        prefix=f"layer.{layer}",
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        dtype=np.float16,
        quant_ctx=ctx,
    )
    return network


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("layer, expected", [(0, len(PROJECTIONS)), (LAYERS - 1, 0)])
def test_each_projection_has_its_own_quantize(tiny, monkeypatch, layer, expected) -> None:
    """A quantized layer has one Quantize per projection; the last layer has none."""
    config, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(config))
    ctx = calibrate_rwkv(config, weights, "fp8", precision="fp16")
    network = _layer_network(weights, ctx, layer)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    quantize = [item for item in layers if item.type == trt.LayerType.QUANTIZE]
    dequantize = [item for item in layers if item.type == trt.LayerType.DEQUANTIZE]
    assert len(quantize) == expected
    assert len(dequantize) == 2 * expected
    assert all(item.get_output(0).dtype == trt.float16 for item in dequantize)


@pytest.mark.gpu
@pytest.mark.trt
def test_unquantized_layer_is_unchanged(tiny) -> None:
    """Without a context the layer is the same network as before (no Q/DQ, same layer count)."""
    config, weights, _ = tiny
    network = _layer_network(weights, None, 0)
    types = [network.get_layer(i).type for i in range(network.num_layers)]
    assert trt.LayerType.QUANTIZE not in types and trt.LayerType.DEQUANTIZE not in types
    assert types.count(trt.LayerType.MATRIX_MULTIPLY) == len(PROJECTIONS)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
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
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "max_state_0", f"present_max_{LAYERS - 1}"} <= names
    assert quantized.files["runtime.json"]["precision"] == "fp16"
