# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

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
    LAYER_PROJECTIONS,
    calibrate_patchtst,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

D_MODEL, FFN, LAYERS, HEADS, PATCH, CONTEXT, CHANNELS, HORIZON = 64, 128, 2, 4, 8, 48, 2, 4
NUM_PATCHES = CONTEXT // PATCH


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_patchtst(path: Path, seed: int = 0, extra_config: dict | None = None) -> None:
    """Write a small PatchTST forecast checkpoint (HF layout) and its config."""
    rng = np.random.default_rng(seed)
    config = {
        "architectures": ["PatchTSTForPrediction"],
        "model_type": "patchtst",
        "d_model": D_MODEL,
        "ffn_dim": FFN,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "context_length": CONTEXT,
        "patch_length": PATCH,
        "patch_stride": PATCH,
        "num_input_channels": CHANNELS,
        "prediction_length": HORIZON,
        "use_cls_token": True,
        "norm_type": "batchnorm",
        "loss": "mse",
        "scaling": "std",
        "activation_function": "gelu",
        "pre_norm": True,
        "share_embedding": True,
        "share_projection": True,
    }
    config.update(extra_config or {})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "model.encoder.embedder.input_embedding.weight": _rand(rng, D_MODEL, PATCH),
        "model.encoder.embedder.input_embedding.bias": _rand(rng, D_MODEL),
        "model.encoder.positional_encoder.position_enc": _rand(rng, NUM_PATCHES + 1, D_MODEL),
        "model.encoder.positional_encoder.cls_token": _rand(rng, 1, 1, 1, D_MODEL),
        "head.projection.weight": _rand(rng, HORIZON, D_MODEL),
        "head.projection.bias": _rand(rng, HORIZON),
    }
    for layer in range(LAYERS):
        prefix = f"model.encoder.layers.{layer}"
        for norm in ("norm_sublayer1", "norm_sublayer3"):
            tensors[f"{prefix}.{norm}.batchnorm.weight"] = np.ones(D_MODEL, np.float32)
            tensors[f"{prefix}.{norm}.batchnorm.bias"] = np.zeros(D_MODEL, np.float32)
            tensors[f"{prefix}.{norm}.batchnorm.running_mean"] = np.zeros(D_MODEL, np.float32)
            tensors[f"{prefix}.{norm}.batchnorm.running_var"] = np.ones(D_MODEL, np.float32)
        for part in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{prefix}.self_attn.{part}.weight"] = _rand(rng, D_MODEL, D_MODEL)
            tensors[f"{prefix}.self_attn.{part}.bias"] = _rand(rng, D_MODEL)
        tensors[f"{prefix}.ff.0.weight"] = _rand(rng, FFN, D_MODEL)
        tensors[f"{prefix}.ff.0.bias"] = _rand(rng, FFN)
        tensors[f"{prefix}.ff.3.weight"] = _rand(rng, D_MODEL, FFN)
        tensors[f"{prefix}.ff.3.bias"] = _rand(rng, D_MODEL)
    save_file(tensors, str(path / "model.safetensors"))


def _load(path: Path):
    config = ModelConfig.from_dir(path)
    weights = model_module._load_all_tensors(path, precision="fp32", depth=LAYERS)
    return config.raw, weights


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's raw config and its fp32 weights as the build loads them."""
    _write_tiny_patchtst(tmp_path)
    raw, weights = _load(tmp_path)
    return raw, weights, tmp_path


def _all_scales(raw: dict, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in projection_names(raw)}


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2), dtype=np.float32))


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    raw, weights, _ = tiny
    key = scale_key(raw, weights)
    assert key.startswith(
        f"patchtst-d{D_MODEL}-l{LAYERS}-f{FFN}-n{HEADS}-c{CONTEXT}-p{PATCH}-s{PATCH}-ch{CHANNELS}-w"
    )
    other = tmp_path_factory.mktemp("other")
    _write_tiny_patchtst(other, seed=1)
    assert scale_key(*_load(other)) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    raw, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(raw, weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed entry has a positive scale, or null, for every projection of its layers."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        layers = int(key.split("-l")[1].split("-")[0])
        assert list(scales) == projection_names({"num_hidden_layers": layers}), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_only_the_forecast_checkpoint_has_committed_scales() -> None:
    """FP8 is supported for the 7-channel granite forecaster only; the 6-channel regression
    checkpoint has no entry, so an FP8 build of it raises the unknown-model-config error."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    keys = list(json.loads(path.read_text(encoding="utf-8")))
    assert len(keys) == 1
    assert "-ch7-" in keys[0]


def test_quantized_modules_leave_the_attention_output_in_fp16() -> None:
    """Per layer q/k/v (one shared input) and the feed-forward; no attention output projection."""
    assert LAYER_PROJECTIONS == ("self_attn.qkv", "ff.0", "ff.3")
    assert quantization.weight_prefixes("layers.2.self_attn.qkv") == [
        "model.encoder.layers.2.self_attn.q_proj",
        "model.encoder.layers.2.self_attn.k_proj",
        "model.encoder.layers.2.self_attn.v_proj",
    ]
    assert quantization.weight_prefixes("layers.0.ff.3") == ["model.encoder.layers.0.ff.3"]


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each module gets FP8 weights in ``[in, out]`` layout and its bias; qkv gives three."""
    raw, weights, _ = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(raw))
    ctx = calibrate_patchtst(raw, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == LAYERS * len(LAYER_PROJECTIONS)
    qkv = ctx.scales["layers.1.self_attn.qkv"]
    assert qkv.input_scale == pytest.approx(0.02)
    assert len(qkv.weights) == 3
    assert all((w.in_features, w.out_features) == (D_MODEL, D_MODEL) for w in qkv.weights)
    ff0 = ctx.scales["layers.0.ff.0"].weights[0]
    assert (ff0.in_features, ff0.out_features) == (D_MODEL, FFN)
    ff3 = ctx.scales["layers.0.ff.3"].weights[0]
    assert (ff3.in_features, ff3.out_features) == (FFN, D_MODEL)
    assert ff3.bias.dtype == np.float16 and ff3.bias.shape == (D_MODEL,)

    k_weight = np.asarray(weights["model.encoder.layers.1.self_attn.k_proj.weight"], np.float32)
    packed, scale = quantize_fp8_weight(k_weight.T)
    assert np.array_equal(qkv.weights[1].packed, packed)
    assert qkv.weights[1].weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("layers.0.self_attn.out")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that projection in fp16 on purpose."""
    raw, weights, _ = tiny
    name = "layers.1.ff.3"
    monkeypatch.setattr(
        quantization, "load_activation_scales", lambda *_: _all_scales(raw, null=(name,))
    )
    ctx = calibrate_patchtst(raw, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("layers.0.ff.3")


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A projection missing from the file fails the build and names the calibration tool."""
    raw, weights, _ = tiny
    scales = _all_scales(raw)
    del scales["layers.0.ff.0"]
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"layers\.0\.ff\.0.*calibrate"):
        calibrate_patchtst(raw, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_projection_is_refused(tiny, monkeypatch) -> None:
    """A table with more layers than the model is a mismatch, not a silent partial build."""
    raw, weights, _ = tiny
    scales = _all_scales(raw)
    scales["layers.9.ff.0"] = 0.02
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"layers\.9"):
        calibrate_patchtst(raw, weights, "fp8", precision="fp16")


def test_already_quantized_checkpoint_is_detected(tiny) -> None:
    """Quantization metadata or scale tensors mark a checkpoint as already quantized."""
    raw, weights, model_dir = tiny
    assert not quantization.is_quantized_checkpoint(raw, model_dir, weights)
    assert quantization.is_quantized_checkpoint(
        {**raw, "quantization_config": {"quant_method": "modelopt"}}, model_dir, weights
    )
    scaled = {**weights, "model.encoder.layers.0.ff.0.weight_scale": np.ones(1, np.float32)}
    assert quantization.is_quantized_checkpoint(raw, model_dir, scaled)
    (model_dir / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    assert quantization.is_quantized_checkpoint(raw, model_dir, weights)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    raw, weights, _ = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(raw))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_patchtst(raw, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    raw, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_patchtst(raw, weights, "nvfp4", precision="fp16")


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
        task="time_series_forecast",
        model_dir=str(model_dir),
        precision="fp16",
        max_sequence_length=None,
        quantization="fp8",
        fp32_layers=(),
        mtp_seq_len=None,
        tensor_parallel_size=1,
        verbose=False,
        backend="trt",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_refuses_an_unknown_model_config(tiny) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_fp8_on_fp32(tiny) -> None:
    """FP8 needs the fp16 build."""
    *_, model_dir = tiny
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision="fp32"), _Writer())


def test_build_refuses_an_already_quantized_checkpoint(tmp_path: Path) -> None:
    """A checkpoint with quantization metadata is refused before any weight is quantized."""
    _write_tiny_patchtst(tmp_path, extra_config={"quantization_config": {"quant_method": "fp8"}})
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(tmp_path), _Writer())


def test_build_refuses_other_quantization(tiny) -> None:
    """A quantization other than fp8 or none is rejected."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model_module.build(_request(model_dir, quantization="nvfp4"), _Writer())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attributes the build option touches."""

    def __init__(self, routes: str = ""):
        self.profiling_verbosity = None
        self.build_route = None
        self.all_build_routes = routes


def test_fp8_build_options_apply_only_to_quantized_builds() -> None:
    """A quantized build asks for detailed profiling and turns the dual-GEMM route off when listed."""
    quantized = _FakeBuilderConfig("peep:match_dual_gemm,other")
    quantization.configure_fp8_build(quantized, object())
    assert quantized.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    assert quantized.build_route == "-peep:match_dual_gemm=off"
    unlisted = _FakeBuilderConfig()
    quantization.configure_fp8_build(unlisted, object())
    assert unlisted.build_route is None
    plain = _FakeBuilderConfig("peep:match_dual_gemm")
    quantization.configure_fp8_build(plain, None)
    assert plain.profiling_verbosity is None and plain.build_route is None


# --- calibration tool ---------------------------------------------------------------------


def test_projection_name_maps_qkv_to_one_entry() -> None:
    """q, k and v share one entry; norms, embedding and head are skipped."""
    name = calibrate_fp8_scales.projection_name
    for part in ("q", "k", "v"):
        assert name(f"model.encoder.layers.2.self_attn.{part}_proj") == "layers.2.self_attn.qkv"
    assert name("model.encoder.layers.0.self_attn.out_proj") == "layers.0.self_attn.out_proj"
    assert name("model.encoder.layers.1.ff.0") == "layers.1.ff.0"
    assert name("model.encoder.layers.1.ff.3") == "layers.1.ff.3"
    assert name("model.encoder.layers.1.ff.1") is None
    assert name("model.encoder.layers.1.norm_sublayer1.batchnorm") is None
    assert name("model.encoder.embedder.input_embedding") is None
    assert name("head.projection") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"layers.{layer}.ff.3": 0.01 for layer in range(6)}
    scales.update({f"layers.{layer}.ff.0": 0.5 for layer in range(6)})
    scales["layers.3.ff.3"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layers.3.ff.3"]
    assert table["layers.3.ff.3"] is None
    assert table["layers.2.ff.3"] == 0.01
    assert table["layers.3.ff.0"] == 0.5


def test_calibration_windows_come_from_the_training_split(tmp_path: Path) -> None:
    """Only the first 12 months are read, and windows are cut evenly from every file."""
    path = tmp_path / "ETTh1.csv"
    columns = ("date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
    lines = [",".join(columns)] + [
        ",".join([f"t{row}"] + [str(row + 0.1 * column) for column in range(7)])
        for row in range(9000)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = calibrate_fp8_scales.read_train_rows(path, 6)
    assert rows.shape == (8640, 6)
    assert rows[-1, 0] == 8639
    windows = calibrate_fp8_scales.cut_windows([rows, rows[:100]], 5, 16, np.random.default_rng(0))
    assert windows.shape == (5, 16, 6)
    assert windows[:, :, 0].max() < 8640


def test_calibration_cli_passes_its_arguments(monkeypatch) -> None:
    """The command line dispatches to calibrate with its arguments."""
    seen = {}
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *a: seen.update(c=a))
    calibrate_fp8_scales.main(["--model-dir", "m", "--ett-dir", "e", "--count", "8"])
    assert seen["c"] == (Path("m"), Path("e"), 8)


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
    ctx = quantization.PatchTSTQuantContext(
        scales={
            "m": quantization._ModuleScales(
                0.1, (quantization._FP8Weight(packed, scale, 64, 32, None),)
            )
        }
    )
    out = ctx.linear(network, ctx.quantize_input(network, activation, "m"), "m")
    ctx.linear(network, ctx.quantize_input(network, activation, "m"), "m")
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in [layer.type for layer in layers]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    # One weight Dequantize shared by both uses, one activation Dequantize per use.
    assert len(dequantize) == 3
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert out.dtype == trt.float16


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
    """fp32, fp16 and fp16 + FP8 build from the real build path; FP8 is the smallest plan."""
    raw, _, model_dir = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(raw))
    plans = {}
    for name, overrides in (
        ("fp32", dict(precision="fp32", quantization=None)),
        ("fp16", dict(quantization=None)),
        ("fp8", {}),
    ):
        writer = _Writer()
        model_module.build(_request(model_dir, **overrides), writer)
        plans[name] = writer.files["engine.plan"]
        assert writer.files["runtime.json"]["context_length"] == CONTEXT
    assert len(plans["fp8"]) < len(plans["fp16"])
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plans["fp8"])
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "past_values",
        "past_observed_mask",
        "prediction_outputs",
    }


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_weights_are_shared_by_every_channel_row(tmp_path: Path, monkeypatch) -> None:
    """FP8 weight constants and the fp16 out_proj / head constants do not grow with channels."""
    counts = {}
    for channels in (1, 5):
        path = tmp_path / f"ch{channels}"
        path.mkdir()
        _write_tiny_patchtst(path, extra_config={"num_input_channels": channels})
        raw, weights = _load(path)
        monkeypatch.setattr(
            quantization, "load_activation_scales", lambda *_, r=raw: _all_scales(r)
        )
        quant_ctx = calibrate_patchtst(raw, weights, "fp8", precision="fp16")
        config = ModelConfig.from_dir(path)
        config.raw["_fp32_layers"] = []
        fp16 = model_module._load_all_tensors(path, precision="fp16", depth=LAYERS)
        fp16["_task_type"] = model_module._normalize_task_type(config.raw)
        seen = {}

        def capture(builder, network, **_):
            constants = [
                network.get_layer(index)
                for index in range(network.num_layers)
                if network.get_layer(index).type == trt.LayerType.CONSTANT
            ]
            seen["fp8"] = sum(
                1 for layer in constants if layer.get_output(0).dtype == trt.DataType.FP8
            )
            seen["out_and_head"] = sum(
                1
                for layer in constants
                if tuple(layer.get_output(0).shape) in {(D_MODEL, D_MODEL), (D_MODEL, HORIZON)}
                and layer.get_output(0).dtype == trt.float16
            )
            return b""

        monkeypatch.setattr(model_module, "build_serialized_network", capture)
        model_module._build_patchtst_network(
            config, fp16, precision="fp16", verbose=False, quant_ctx=quant_ctx
        )
        counts[channels] = (seen["fp8"], seen["out_and_head"])
    # q, k, v, ff.0, ff.3 in FP8 per layer; out_proj per layer and one head in fp16.
    assert counts[1] == counts[5] == (5 * LAYERS, LAYERS + 1)
