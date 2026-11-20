# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_densenet FP8 self-quantization: weight quantizer, scale lookup, context, tool, engine."""

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
    calibrate_timm_densenet,
    conv_names,
    load_activation_scales,
    quantize_fp8_conv_weight,
    scale_key,
)

# Channel counts are multiples of 16 so the FP8 convolutions have kernels to pick from.
STEM, GROWTH, MID, BLOCKS, CLASSES, IMAGE = 32, 16, 32, (2, 2), 10, 64
SIZE = (IMAGE, IMAGE)


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


def _bn(rng, tensors: dict, prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = (1.0 + 0.1 * rng.standard_normal(channels)).astype(np.float32)
    tensors[f"{prefix}.bias"] = _rand(rng, channels)
    tensors[f"{prefix}.running_mean"] = _rand(rng, channels)
    tensors[f"{prefix}.running_var"] = (1.0 + np.abs(_rand(rng, channels))).astype(np.float32)


def _write_tiny_densenet(path: Path, seed: int = 0, config_extra: dict | None = None) -> None:
    """Write a small timm DenseNet checkpoint (64 px input, two dense blocks) and config."""
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "densenet121",
        "num_classes": CLASSES,
        "pretrained_cfg": {"input_size": [3, IMAGE, IMAGE]},
        **(config_extra or {}),
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {"features.conv0.weight": _rand(rng, STEM, 3, 7, 7)}
    _bn(rng, tensors, "features.norm0", STEM)
    channels = STEM
    for block, count in enumerate(BLOCKS, start=1):
        for layer in range(1, count + 1):
            prefix = f"features.denseblock{block}.denselayer{layer}"
            _bn(rng, tensors, f"{prefix}.norm1", channels)
            tensors[f"{prefix}.conv1.weight"] = _rand(rng, MID, channels, 1, 1)
            _bn(rng, tensors, f"{prefix}.norm2", MID)
            tensors[f"{prefix}.conv2.weight"] = _rand(rng, GROWTH, MID, 3, 3)
            channels += GROWTH
        if block < len(BLOCKS):
            _bn(rng, tensors, f"features.transition{block}.norm", channels)
            tensors[f"features.transition{block}.conv.weight"] = _rand(
                rng, channels // 2, channels, 1, 1
            )
            channels //= 2
    _bn(rng, tensors, "features.norm5", channels)
    tensors["classifier.weight"] = _rand(rng, CLASSES, channels)
    tensors["classifier.bias"] = _rand(rng, CLASSES)
    save_file(tensors, str(path / "model.safetensors"))


def _load(path: Path):
    config = ModelConfig.from_dir(path)
    weights = model_module._TimmDensenetModel().load_weights(
        str(path), config, precision="fp16", quantization="fp8"
    )
    return config.raw["_timm_densenet_config"]["block_layers"], weights


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's layout and weights as the FP8 build loads them."""
    _write_tiny_densenet(tmp_path)
    block_layers, weights = _load(tmp_path)
    return block_layers, weights, tmp_path


def _all_scales(block_layers, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in conv_names(block_layers)}


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_conv_weight_scales_per_output_channel_and_bytes() -> None:
    """Each output channel gets amax / 448 and the bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((16, 32, 3, 3)).astype(np.float32)
    weight[3] *= 10.0
    packed, scale = quantize_fp8_conv_weight(weight)
    amax = np.abs(weight.reshape(16, -1)).max(axis=1)
    np.testing.assert_allclose(scale, amax / 448.0, rtol=1e-6)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale[:, None, None, None]
    error = np.abs(restored - weight).reshape(16, -1).max(axis=1)
    assert np.all(error <= amax * 0.07)  # e4m3 half-step of each channel


def test_quantize_conv_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_conv_weight(np.full((4, 4, 1, 1), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_conv_weight_floors_tiny_channel_scales() -> None:
    """A zero channel gets the smallest normal fp16 scale and quantizes to zeros."""
    weight = np.ones((2, 4, 1, 1), dtype=np.float32)
    weight[1] = 0.0
    packed, scale = quantize_fp8_conv_weight(weight)
    assert scale[1] == pytest.approx(2.0**-14)
    assert not packed[1].any()


def test_quantize_conv_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_conv_weight(np.zeros((2, 2, 1, 1), dtype=np.float32))


def test_conv_names_cover_dense_layers_and_transitions(tiny) -> None:
    """Two convolutions per dense layer and one per transition; never the stem."""
    block_layers, _, _ = tiny
    names = conv_names(block_layers)
    assert len(names) == 2 * sum(block_layers) + len(block_layers) - 1
    assert "features.denseblock2.denselayer2.conv2" in names
    assert "features.transition1.conv" in names
    assert "features.transition2.conv" not in names
    assert "features.conv0" not in names


def test_fp8_loader_keeps_quantized_weights_float32(tiny) -> None:
    """The quantizer reads the original float32 values; other weights use the work precision."""
    _, weights, _ = tiny
    assert weights["features.denseblock1.denselayer1.conv1.weight"].dtype == np.float32
    assert weights["features.transition1.conv.weight"].dtype == np.float32
    assert weights["features.conv0.weight"].dtype == np.float16
    assert weights["classifier.weight"].dtype == np.float16


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    block_layers, weights, _ = tiny
    key = scale_key(block_layers, SIZE, weights)
    assert key.startswith(f"densenet-c{STEM}-g{GROWTH}-b2x2-s64x64-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_densenet(other, seed=1)
    _, other_weights = _load(other)
    assert scale_key(block_layers, SIZE, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    block_layers, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(block_layers, SIZE, weights)


def test_committed_scales_cover_every_convolution() -> None:
    """Every committed entry has a positive scale, or null, for every convolution of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        block_layers = [int(value) for value in key.split("-b")[1].split("-")[0].split("x")]
        assert set(scales) == set(conv_names(block_layers)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each convolution gets its FP8 weight and per-channel scales."""
    block_layers, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(block_layers))
    ctx = calibrate_timm_densenet(block_layers, SIZE, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == len(conv_names(block_layers))
    name = "features.denseblock2.denselayer1.conv2"
    entry = ctx.scales[name]
    assert entry.input_scale == pytest.approx(0.02)
    packed, scale = quantize_fp8_conv_weight(weights[f"{name}.weight"])
    assert np.array_equal(entry.weight.packed, packed)
    np.testing.assert_array_equal(entry.weight.weight_scale, scale)
    assert entry.weight.packed.shape == (GROWTH, MID, 3, 3)
    assert not ctx.should_quantize("features.conv0")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that convolution in fp16 on purpose."""
    block_layers, weights, _ = tiny
    name = "features.denseblock1.denselayer2.conv1"
    _patch_scales(monkeypatch, _all_scales(block_layers, null=(name,)))
    ctx = calibrate_timm_densenet(block_layers, SIZE, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("features.denseblock1.denselayer1.conv1")
    assert len(ctx.scales) == len(conv_names(block_layers)) - 1


def test_convolution_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A convolution missing from the file fails the build and names the calibration tool."""
    block_layers, weights, _ = tiny
    scales = _all_scales(block_layers)
    del scales["features.transition1.conv"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"transition1\.conv.*calibrate_fp8_scales\.py"):
        calibrate_timm_densenet(block_layers, SIZE, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_convolution_is_refused(tiny, monkeypatch) -> None:
    """A table with more layers than the model is a mismatch, not a silent partial build."""
    block_layers, weights, _ = tiny
    scales = _all_scales(block_layers)
    scales["features.denseblock1.denselayer9.conv1"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"denselayer9\.conv1"):
        calibrate_timm_densenet(block_layers, SIZE, weights, "fp8", precision="fp16")


def test_already_quantized_weights_are_refused(tiny, monkeypatch) -> None:
    """Weights that carry FP8 scales are refused instead of quantized twice."""
    block_layers, weights, _ = tiny
    weights["features.transition1.conv.weight_scale"] = np.ones(1, dtype=np.float32)
    _patch_scales(monkeypatch, _all_scales(block_layers))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_timm_densenet(block_layers, SIZE, weights, "fp8", precision="fp16")


@pytest.mark.parametrize(
    "marker", ["hf_quant_config", "quantization_config", "weight_scale_tensor"]
)
def test_loader_refuses_an_already_quantized_checkpoint(tmp_path: Path, marker) -> None:
    """The FP8 loader checks the checkpoint itself: metadata files, config and tensors."""
    extra = (
        {"quantization_config": {"quant_method": "modelopt"}}
        if marker == "quantization_config"
        else None
    )
    _write_tiny_densenet(tmp_path, config_extra=extra)
    if marker == "hf_quant_config":
        (tmp_path / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    if marker == "weight_scale_tensor":
        from safetensors.numpy import load_file

        tensors = load_file(str(tmp_path / "model.safetensors"))
        tensors["features.transition1.conv.weight_scale"] = np.ones(1, dtype=np.float32)
        save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        _load(tmp_path)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    block_layers, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(block_layers))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_timm_densenet(block_layers, SIZE, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    block_layers, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_densenet(block_layers, SIZE, weights, "nvfp4", precision="fp16")


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
        tensor_parallel_size=1,
        context_parallel_size=1,
        task="image_to_class_scores",
        model_dir=str(model_dir),
        precision="fp16",
        max_sequence_length=1,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        verbose=False,
        backend="tensorrt",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_refuses_an_unknown_model_config(tiny) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_fp8_with_fp32_precision(tiny, monkeypatch) -> None:
    """FP8 on the fp32 engine is rejected before any graph is built."""
    block_layers, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(block_layers))
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision="fp32"), _Writer())


def test_build_refuses_other_quantization(tiny) -> None:
    """A quantization other than fp8 or none is rejected."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model_module.build(_request(model_dir, quantization="nvfp4"), _Writer())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attribute the build option touches."""

    profiling_verbosity = None


def test_fp8_build_keeps_layer_details_only_for_quantized_builds() -> None:
    """A quantized build asks for detailed profiling; an unquantized build is left alone."""
    quantized = _FakeBuilderConfig()
    quantization.configure_fp8_build(quantized, object())
    assert quantized.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    plain = _FakeBuilderConfig()
    quantization.configure_fp8_build(plain, None)
    assert plain.profiling_verbosity is None


# --- calibration tool ---------------------------------------------------------------------


def test_conv_name_matches_only_quantized_convolutions() -> None:
    """Only the dense-layer and transition convolutions are calibrated."""
    name = calibrate_fp8_scales.conv_name
    for conv in ("conv1", "conv2"):
        module = f"features.denseblock3.denselayer17.{conv}"
        assert name(module) == module
    assert name("features.transition2.conv") == "features.transition2.conv"
    assert name("features.conv0") is None
    assert name("features.denseblock1.denselayer1.norm1") is None
    assert name("features.denseblock1.denselayer1.conv1.input_quantizer") is None
    assert name("classifier") is None


def test_outlier_convolutions_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"features.denseblock1.denselayer{i}.conv1": 0.01 for i in range(1, 7)}
    scales.update({f"features.denseblock1.denselayer{i}.conv2": 0.5 for i in range(1, 7)})
    scales["features.denseblock1.denselayer3.conv1"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["features.denseblock1.denselayer3.conv1"]
    assert table["features.denseblock1.denselayer3.conv1"] is None
    assert table["features.denseblock1.denselayer2.conv1"] == 0.01
    assert table["features.denseblock1.denselayer3.conv2"] == 0.5


def test_outlier_kind_ignores_block_and_layer_index() -> None:
    """The median of a kind runs over all blocks, and the transitions form one kind."""
    scales = {f"features.denseblock{b}.denselayer1.conv2": 0.1 for b in range(1, 5)}
    scales.update({f"features.transition{t}.conv": 0.2 for t in (1, 2, 3)})
    scales["features.denseblock4.denselayer1.conv2"] = 5.0
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["features.denseblock4.denselayer1.conv2"]
    assert table["features.transition2.conv"] == 0.2


def test_split_indices_spread_calibration_and_hold_out_the_rest() -> None:
    """Calibration and evaluation images are disjoint and both span the whole set."""
    calibration, held_out = calibrate_fp8_scales.split_indices(2000, 500)
    assert len(calibration) == 500 and calibration[-1] > 1900
    assert not set(calibration) & set(held_out)
    assert len(calibration) + len(held_out) == 2000


def test_calibration_cli_has_fetch_and_calibrate(monkeypatch, tmp_path: Path) -> None:
    """The command line dispatches to fetch and to calibrate with its arguments."""
    seen = {}
    monkeypatch.setattr(
        calibrate_fp8_scales, "fetch_images", lambda out, n: seen.update(f=(out, n))
    )
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda m, i, n: seen.update(c=(m, i, n)))
    monkeypatch.setattr("sys.argv", ["x", "fetch", "--out", str(tmp_path), "--count", "10"])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv", ["x", "calibrate", "--model-dir", "m", "--images", "i", "--calib-images", "5"]
    )
    calibrate_fp8_scales.main()
    assert seen["f"] == (tmp_path, 10)
    assert seen["c"] == (Path("m"), Path("i"), 5)


# --- engine -------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.trt
def test_dequantize_layers_output_fp16_with_per_channel_weight_scales() -> None:
    """Both Dequantize layers output fp16, the weight one per output channel, and no Cast."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (1, 32, 8, 8))
    packed, scale = quantize_fp8_conv_weight(
        np.random.default_rng(0).standard_normal((16, 32, 3, 3)).astype("f")
    )
    ctx = quantization.DensenetQuantContext(
        scales={"m": quantization._LayerScales(0.1, quantization._FP8Conv(packed, scale))}
    )
    out = ctx.conv2d(network, activation, "m", (3, 3), (1, 1))
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in [layer.type for layer in layers]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    weight_dequantize = [
        layer
        for layer in dequantize
        if len(layer.get_input(1).shape) == 1 and layer.get_input(1).shape[0] == 16
    ]
    assert len(weight_dequantize) == 1
    weight_dequantize[0].__class__ = trt.IDequantizeLayer
    assert weight_dequantize[0].axis == 0
    assert tuple(out.shape) == (1, 16, 8, 8)
    assert out.dtype == trt.float16


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_convolutions(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path and runs e4m3 convolutions."""
    block_layers, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(block_layers))
    quantized = _Writer()
    model_module.build(_request(model_dir), quantized)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        quantized.files["engine.plan"]
    )
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "pixel_values",
        "logits",
    }
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    fp8_inputs = [
        layer
        for layer in layers["Layers"]
        if any(tensor.get("Datatype") == "FP8" for tensor in layer.get("Inputs", []))
    ]
    assert fp8_inputs
