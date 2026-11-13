# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_repvgg FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

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
from ..checkpoint import Checkpoint  # noqa: E402
from ..quantization import (  # noqa: E402
    calibrate_timm_repvgg,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
STEM, WIDTH, BLOCKS, CLASSES, IMAGE = 32, 128, (2, 1), 16, 64
SIZE = (IMAGE, IMAGE)


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _batch_norm(rng, tensors, prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = 1.0 + _rand(rng, channels)
    tensors[f"{prefix}.bias"] = _rand(rng, channels)
    tensors[f"{prefix}.running_mean"] = _rand(rng, channels)
    tensors[f"{prefix}.running_var"] = np.abs(_rand(rng, channels)) + 1.0


def _branch(rng, tensors, prefix: str, out_channels: int, in_channels: int, kernel: int) -> None:
    tensors[f"{prefix}.conv.weight"] = _rand(rng, out_channels, in_channels, kernel, kernel)
    _batch_norm(rng, tensors, f"{prefix}.bn", out_channels)


def _write_tiny_repvgg(path: Path, seed: int = 0) -> None:
    """Write a small timm RepVGG checkpoint (64 px input, three blocks) and config."""
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "repvgg_a2",
        "num_classes": CLASSES,
        "num_features": WIDTH,
        "pretrained_cfg": {"input_size": [3, IMAGE, IMAGE]},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {}
    _branch(rng, tensors, "stem.conv_kxk", STEM, 3, 3)
    _branch(rng, tensors, "stem.conv_1x1", STEM, 3, 1)
    channels = STEM
    for stage, count in enumerate(BLOCKS):
        for index in range(count):
            prefix = f"stages.{stage}.{index}"
            _branch(rng, tensors, f"{prefix}.conv_kxk", WIDTH, channels, 3)
            _branch(rng, tensors, f"{prefix}.conv_1x1", WIDTH, channels, 1)
            if index > 0:
                _batch_norm(rng, tensors, f"{prefix}.identity", WIDTH)
            channels = WIDTH
    tensors["head.fc.weight"] = _rand(rng, CLASSES, WIDTH)
    tensors["head.fc.bias"] = _rand(rng, CLASSES)
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's blocks and fused fp16 weights as the build loads them."""
    _write_tiny_repvgg(tmp_path)
    blocks = model_module._layout(Checkpoint.open(tmp_path))
    weights = model_module._weights(Checkpoint.open(tmp_path), blocks, np.float16)
    return blocks, weights, tmp_path


def _all_scales(blocks, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in projection_names(blocks)}


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_fp8_weight_scales_each_output_channel() -> None:
    """Each output channel has scale amax / 448 and its bytes decode back to its weights."""
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((64, 32, 3, 3)).astype(np.float32)
    weight *= np.logspace(-2, 1, 64, dtype=np.float32).reshape(-1, 1, 1, 1)
    packed, scale = quantize_fp8_weight(weight)
    amax = np.abs(weight.reshape(64, -1)).max(axis=1)
    np.testing.assert_allclose(scale, amax / 448.0, rtol=1e-6)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale.reshape(-1, 1, 1, 1)
    error = np.abs(restored - weight).reshape(64, -1).max(axis=1)
    assert (error <= amax * 0.07).all()  # e4m3 half-step of each channel


def test_quantize_fp8_weight_floors_small_channels() -> None:
    """A tiny or all-zero channel gets the smallest normal fp16 scale; nothing becomes NaN."""
    weight = np.full((4, 4, 3, 3), 3.0, dtype=np.float32)
    weight[1] = 1e-3
    weight[2] = 0.0
    packed, scale = quantize_fp8_weight(weight)
    assert scale[1] == scale[2] == np.float32(2.0**-14)
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale.reshape(-1, 1, 1, 1)
    assert not np.isnan(restored).any() and not restored[2].any()
    np.testing.assert_allclose(restored[1], 1e-3, rtol=0.07)


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2, 3, 3), dtype=np.float32))


def test_projection_names_are_the_blocks(tiny) -> None:
    """One fused convolution per block; the stem stays unquantized."""
    blocks, _, _ = tiny
    assert projection_names(blocks) == ["stages.0.0", "stages.0.1", "stages.1.0"]


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    blocks, weights, _ = tiny
    key = scale_key(blocks, SIZE, weights)
    assert key.startswith(f"repvgg-c{STEM}x{WIDTH}x{WIDTH}x{WIDTH}-s64x64-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_repvgg(other, seed=1)
    other_weights = model_module._weights(Checkpoint.open(other), blocks, np.float16)
    assert scale_key(blocks, SIZE, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    blocks, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(blocks, SIZE, weights)


def test_committed_scales_cover_every_block() -> None:
    """Every committed entry has a positive scale, or null, for every block of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        widths = key.split("-c")[1].split("-")[0].split("x")
        assert len(scales) == len(widths) - 1, key
        assert all(name.startswith("stages.") for name in scales), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_fused_weights(tiny, monkeypatch) -> None:
    """Each block gets the FP8 bytes of its fused weight and one scale per output channel."""
    blocks, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(blocks))
    ctx = calibrate_timm_repvgg(blocks, SIZE, weights, "fp8", precision="fp16")

    assert set(ctx.scales) == {"stages.0.0", "stages.0.1", "stages.1.0"}
    entry = ctx.scales["stages.0.1"]
    assert entry.input_scale == pytest.approx(0.02)
    assert entry.weight.packed.shape == (WIDTH, WIDTH, 3, 3)
    assert entry.weight.weight_scale.shape == (WIDTH,)
    packed, scale = quantize_fp8_weight(np.asarray(weights["stages.0.1.weight"], np.float32))
    assert np.array_equal(entry.weight.packed, packed)
    np.testing.assert_array_equal(entry.weight.weight_scale, scale)
    assert not ctx.should_quantize("stem")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that block in fp16 on purpose."""
    blocks, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(blocks, null=("stages.0.1",)))
    ctx = calibrate_timm_repvgg(blocks, SIZE, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize("stages.0.1")
    assert ctx.should_quantize("stages.1.0")


def test_block_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A block missing from the file fails the build and names the calibration tool."""
    blocks, weights, _ = tiny
    scales = _all_scales(blocks)
    del scales["stages.1.0"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"stages\.1\.0.*calibrate_fp8_scales\.py"):
        calibrate_timm_repvgg(blocks, SIZE, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_block_is_refused(tiny, monkeypatch) -> None:
    """A table with more blocks than the model is a mismatch, not a silent partial build."""
    blocks, weights, _ = tiny
    scales = _all_scales(blocks)
    scales["stages.1.5"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"stages\.1\.5"):
        calibrate_timm_repvgg(blocks, SIZE, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    blocks, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(blocks))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_timm_repvgg(blocks, SIZE, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    blocks, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_repvgg(blocks, SIZE, weights, "nvfp4", precision="fp16")


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


def test_already_quantized_checkpoint_is_refused(tiny, monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    blocks, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(blocks))
    tensors = load_file(str(model_dir / "model.safetensors"))
    tensors["stages.0.0.conv_kxk.conv.weight_scale"] = np.ones(1, dtype=np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


def test_quantization_config_is_refused(tmp_path: Path) -> None:
    """A config that declares a quantization is refused before any weight is read."""
    _write_tiny_repvgg(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"] = {"quant_method": "modelopt"}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        quantization.refuse_quantized_checkpoint(tmp_path, [])


def test_build_refuses_an_unknown_model_config(tiny) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_fp8_with_fp32_precision(tiny, monkeypatch) -> None:
    """FP8 on the fp32 engine is rejected before any graph is built."""
    blocks, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(blocks))
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


def test_projection_name_maps_the_block_input() -> None:
    """The 3x3 branch convolution of a block names the block; nothing else is calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("stages.3.0.conv_kxk.conv") == "stages.3.0"
    assert name("stages.3.0.conv_1x1.conv") is None
    assert name("stages.3.0.identity") is None
    assert name("stem.conv_kxk.conv") is None
    assert name("head.fc") is None
    assert name("stages.3.0.conv_kxk.conv.input_quantizer") is None


def test_outlier_blocks_become_null() -> None:
    """A scale above 20x the median of all blocks is stored as null, the others are kept."""
    scales = {f"stages.1.{index}": 0.01 for index in range(6)}
    scales["stages.2.0"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["stages.2.0"]
    assert table["stages.2.0"] is None
    assert table["stages.1.3"] == 0.01


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
def test_quantized_convolution_dequantizes_per_channel_to_fp16() -> None:
    """The weight Dequantize runs per output channel; both Dequantize layers output fp16."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (1, 64, 8, 8))
    packed, scale = quantize_fp8_weight(
        np.random.default_rng(0).standard_normal((32, 64, 3, 3)).astype("f")
    )
    ctx = quantization.RepVggQuantContext(
        scales={"m": quantization._LayerScales(0.1, quantization._FP8Weight(packed, scale))}
    )
    out = ctx.convolution(network, activation, "m", np.zeros(32, np.float32), stride=2)
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in [layer.type for layer in layers]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert sorted(tuple(layer.get_input(1).shape) for layer in dequantize) == [(1,), (32,)]
    assert out.dtype == trt.float16
    assert tuple(out.shape) == (1, 32, 4, 4)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path, stores FP8 weights and is smaller."""
    blocks, _, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(blocks))
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
