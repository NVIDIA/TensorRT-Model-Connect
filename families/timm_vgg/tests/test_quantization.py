# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_vgg FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, model as model_module, quantization  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    HEAD_PROJECTIONS,
    calibrate_timm_vgg,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

# Sequential indices of a small VGG: two convolutions, pool, two convolutions, pool.
CONV_INDICES, WIDTH, HEAD, CLASSES, IMAGE = (0, 2, 5, 7), 64, 128, 16, 32
SIZE = (IMAGE, IMAGE)


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_vgg(path: Path, seed: int = 0) -> None:
    """Write a small timm VGG checkpoint (32 px input, two pools, 8x8 head) and config."""
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "vgg16",
        "num_classes": CLASSES,
        "pretrained_cfg": {
            "input_size": [3, IMAGE, IMAGE],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "crop_mode": "center",
            "interpolation": "bilinear",
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {}
    channels = 3
    for index in CONV_INDICES:
        tensors[f"features.{index}.weight"] = _rand(rng, WIDTH, channels, 3, 3)
        tensors[f"features.{index}.bias"] = _rand(rng, WIDTH)
        channels = WIDTH
    tensors["pre_logits.fc1.weight"] = _rand(rng, HEAD, WIDTH, 8, 8)
    tensors["pre_logits.fc1.bias"] = _rand(rng, HEAD)
    tensors["pre_logits.fc2.weight"] = _rand(rng, HEAD, HEAD, 1, 1)
    tensors["pre_logits.fc2.bias"] = _rand(rng, HEAD)
    tensors["head.fc.weight"] = _rand(rng, CLASSES, HEAD)
    tensors["head.fc.bias"] = _rand(rng, CLASSES)
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's convolution indices and fp16 weights as the build loads them."""
    _write_tiny_vgg(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._TimmVggModel().load_weights(str(tmp_path), config, precision="fp16")
    return list(config.raw["_timm_vgg_config"]["conv_indices"]), weights, tmp_path


def _all_scales(indices, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in projection_names(indices)}


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((64, 32, 3, 3)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_weight(np.full((4, 4, 3, 3), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2, 3, 3), dtype=np.float32))


def test_projection_names_skip_the_input_convolution(tiny) -> None:
    """Every feature convolution after the first, and both convolutional head layers."""
    indices, _, _ = tiny
    names = projection_names(indices)
    assert names == ["features.2", "features.5", "features.7", *HEAD_PROJECTIONS]
    assert "features.0" not in names and "head.fc" not in names


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    indices, weights, _ = tiny
    key = scale_key(indices, SIZE, weights)
    assert key.startswith(f"vgg-c64x64x64x64-h{HEAD}-s32x32-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_vgg(other, seed=1)
    config = ModelConfig.from_dir(other)
    other_weights = model_module._TimmVggModel().load_weights(str(other), config, precision="fp16")
    assert scale_key(indices, SIZE, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    indices, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(indices, SIZE, weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed entry has a positive scale, or null, for every convolution of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert {name for name in scales if not name.startswith("features.")} == set(
            HEAD_PROJECTIONS
        ), key
        widths = key.split("-c")[1].split("-")[0].split("x")
        assert sum(name.startswith("features.") for name in scales) == len(widths) - 1, key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each convolution gets an FP8 weight in [out, in, kh, kw] layout with its own scale."""
    indices, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(indices))
    ctx = calibrate_timm_vgg(indices, SIZE, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == len(indices) - 1 + len(HEAD_PROJECTIONS)
    conv = ctx.scales["features.5"]
    assert conv.input_scale == pytest.approx(0.02)
    assert conv.weight.packed.shape == (WIDTH, WIDTH, 3, 3)
    assert ctx.scales["pre_logits.fc1"].weight.packed.shape == (HEAD, WIDTH, 8, 8)

    packed, scale = quantize_fp8_weight(np.asarray(weights["features.5.weight"], np.float32))
    assert np.array_equal(conv.weight.packed, packed)
    assert conv.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("features.0")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that convolution in fp16 on purpose."""
    indices, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(indices, null=("features.7",)))
    ctx = calibrate_timm_vgg(indices, SIZE, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize("features.7")
    assert ctx.should_quantize("features.5")


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A convolution missing from the file fails the build and names the calibration tool."""
    indices, weights, _ = tiny
    scales = _all_scales(indices)
    del scales["pre_logits.fc2"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"pre_logits\.fc2.*calibrate_fp8_scales\.py"):
        calibrate_timm_vgg(indices, SIZE, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_projection_is_refused(tiny, monkeypatch) -> None:
    """A table for another layout (or the input convolution) is a mismatch, not a partial build."""
    indices, weights, _ = tiny
    scales = _all_scales(indices)
    scales["features.0"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"features\.0"):
        calibrate_timm_vgg(indices, SIZE, weights, "fp8", precision="fp16")


def test_already_quantized_checkpoint_is_refused(tiny, monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    indices, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(indices))
    tensors = load_file(str(model_dir / "model.safetensors"))
    tensors["features.2.weight_scale"] = np.ones(1, dtype=np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


def test_quantization_config_is_refused(tmp_path: Path) -> None:
    """A config that declares a quantization is refused before any weight is read."""
    _write_tiny_vgg(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"] = {"quant_method": "modelopt"}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        quantization.refuse_quantized_checkpoint(tmp_path, [])


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    indices, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(indices))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_timm_vgg(indices, SIZE, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    indices, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_vgg(indices, SIZE, weights, "nvfp4", precision="fp16")


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
        task="classification",
        model_dir=str(model_dir),
        precision="fp16",
        max_sequence_length=1,
        quantization="fp8",
        fp32_layers=None,
        mtp_seq_len=None,
        tensor_parallel_size=1,
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
    indices, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(indices))
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


def test_projection_name_matches_only_convolutions() -> None:
    """Feature convolutions and the two head convolutions are calibrated, nothing else."""
    name = calibrate_fp8_scales.projection_name
    assert name("features.28") == "features.28"
    assert name("pre_logits.fc1") == "pre_logits.fc1"
    assert name("pre_logits.fc2") == "pre_logits.fc2"
    assert name("head.fc") is None
    assert name("features.28.input_quantizer") is None
    assert name("pre_logits.act1") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"features.{index}": 0.01 for index in (2, 5, 7, 10, 12)}
    scales["features.7"] = 0.5
    scales["pre_logits.fc1"] = 0.5
    scales["pre_logits.fc2"] = 0.001
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["features.7"]
    assert table["features.7"] is None
    assert table["pre_logits.fc1"] == 0.5  # its own kind
    assert table["features.5"] == 0.01


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
def test_quantized_convolution_dequantizes_to_fp16() -> None:
    """Both Dequantize layers of a convolution output fp16, and no Cast is added."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (1, 64, 8, 8))
    packed, scale = quantize_fp8_weight(
        np.random.default_rng(0).standard_normal((32, 64, 3, 3)).astype("f")
    )
    ctx = quantization.VggQuantContext(
        scales={"m": quantization._LayerScales(0.1, quantization._FP8Weight(packed, scale))}
    )
    out = ctx.convolution(network, activation, "m", np.zeros(32, np.float32), padding=(1, 1))
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in [layer.type for layer in layers]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert out.dtype == trt.float16
    assert tuple(out.shape) == (1, 32, 8, 8)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path, stores FP8 weights and is smaller."""
    indices, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(indices))
    plain, quantized = _Writer(), _Writer()
    model_module.build(_request(model_dir, quantization=None), plain)
    model_module.build(_request(model_dir), quantized)

    plain_plan, fp8_plan = plain.files["engine.plan"], quantized.files["engine.plan"]
    assert len(fp8_plan) < 0.9 * len(plain_plan)
    assert quantized.files["runtime.json"] == plain.files["runtime.json"]
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(fp8_plan)
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "pixel_values",
        "logits",
    }
