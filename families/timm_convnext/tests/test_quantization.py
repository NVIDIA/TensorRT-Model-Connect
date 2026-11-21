# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_convnext FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

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
    DOWNSAMPLE,
    PROJECTIONS,
    calibrate_timm_convnext,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
WIDTHS, DEPTHS, CLASSES, IMAGE = (64, 128), (2, 2), 16, 64
SIZE = (IMAGE, IMAGE)


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_convnext(path: Path, seed: int = 0) -> None:
    """Write a small timm ConvNeXt checkpoint (64 px input, 4 px stem patch) and config."""
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "convnext_tiny",
        "num_classes": CLASSES,
        "pretrained_cfg": {"input_size": [3, IMAGE, IMAGE], "crop_pct": 0.95},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "stem.0.weight": _rand(rng, WIDTHS[0], 3, 4, 4),
        "stem.0.bias": _rand(rng, WIDTHS[0]),
        "stem.1.weight": np.ones(WIDTHS[0], np.float32),
        "stem.1.bias": _rand(rng, WIDTHS[0]),
        "head.norm.weight": np.ones(WIDTHS[-1], np.float32),
        "head.norm.bias": _rand(rng, WIDTHS[-1]),
        "head.fc.weight": _rand(rng, CLASSES, WIDTHS[-1]),
        "head.fc.bias": _rand(rng, CLASSES),
    }
    for stage, depth in enumerate(DEPTHS):
        width = WIDTHS[stage]
        if stage:
            tensors[f"stages.{stage}.downsample.0.weight"] = np.ones(WIDTHS[stage - 1], np.float32)
            tensors[f"stages.{stage}.downsample.0.bias"] = _rand(rng, WIDTHS[stage - 1])
            tensors[f"stages.{stage}.downsample.1.weight"] = _rand(
                rng, width, WIDTHS[stage - 1], 2, 2
            )
            tensors[f"stages.{stage}.downsample.1.bias"] = _rand(rng, width)
        for block in range(depth):
            prefix = f"stages.{stage}.blocks.{block}"
            tensors[f"{prefix}.conv_dw.weight"] = _rand(rng, width, 1, 7, 7)
            tensors[f"{prefix}.conv_dw.bias"] = _rand(rng, width)
            tensors[f"{prefix}.norm.weight"] = np.ones(width, np.float32)
            tensors[f"{prefix}.norm.bias"] = _rand(rng, width)
            tensors[f"{prefix}.mlp.fc1.weight"] = _rand(rng, 4 * width, width)
            tensors[f"{prefix}.mlp.fc1.bias"] = _rand(rng, 4 * width)
            tensors[f"{prefix}.mlp.fc2.weight"] = _rand(rng, width, 4 * width)
            tensors[f"{prefix}.mlp.fc2.bias"] = _rand(rng, width)
            tensors[f"{prefix}.gamma"] = np.full(width, 0.5, np.float32)
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's layout and fp16 weights as the build loads them."""
    _write_tiny_convnext(tmp_path)
    depths = model_module._layout(Checkpoint.open(tmp_path))
    weights = model_module._weights(Checkpoint.open(tmp_path), depths, np.float16)
    return depths, weights, tmp_path


def _all_scales(depths, weights, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in projection_names(depths, weights)}


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_fp8_weight_scale_and_bytes() -> None:
    """The weight scale is amax / 448 and the bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((64, 32, 2, 2)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= np.abs(weight).max() * 0.07  # e4m3 half-step


def test_quantize_fp8_weight_never_overflows_to_nan() -> None:
    """Values at the scale boundary clip to the largest finite fp8 value."""
    packed, _ = quantize_fp8_weight(np.full((4, 4, 1, 1), 3.0, dtype=np.float32))
    assert not np.isnan(packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)).any()


def test_quantize_fp8_weight_rejects_zero_tensor() -> None:
    """An all-zero weight has no usable scale and is refused."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((2, 2, 1, 1), dtype=np.float32))


def test_projection_names_cover_block_mlps_and_downsamples(tiny) -> None:
    """Two MLP convolutions per block and one downsample per stage after the first."""
    depths, weights, _ = tiny
    names = projection_names(depths, weights)
    assert len(names) == sum(depths) * len(PROJECTIONS) + (len(depths) - 1)
    assert f"stages.1.{DOWNSAMPLE}" in names
    assert f"stages.0.{DOWNSAMPLE}" not in names
    assert "stages.1.blocks.1.mlp.fc2" in names
    assert not any("conv_dw" in name or name.startswith("stem") for name in names)


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    depths, weights, _ = tiny
    key = scale_key(depths, SIZE, weights)
    assert key.startswith("convnext-c64x128-d2x2-p4-s64x64-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_convnext(other, seed=1)
    other_weights = model_module._weights(Checkpoint.open(other), depths, np.float16)
    assert scale_key(depths, SIZE, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    depths, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(depths, SIZE, weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed entry has a positive scale, or null, for every convolution of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        depths = [int(value) for value in key.split("-d")[1].split("-")[0].split("x")]
        fake = {f"stages.{stage}.downsample.weight": None for stage in range(1, len(depths))}
        assert set(scales) == set(projection_names(depths, fake)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each convolution gets an FP8 weight in [out, in, kh, kw] layout with its own scale."""
    depths, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(depths, weights))
    ctx = calibrate_timm_convnext(depths, SIZE, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == sum(depths) * len(PROJECTIONS) + 1
    fc1 = ctx.scales["stages.1.blocks.1.mlp.fc1"]
    assert fc1.input_scale == pytest.approx(0.02)
    assert fc1.weight.packed.shape == (4 * WIDTHS[1], WIDTHS[1], 1, 1)
    down = ctx.scales[f"stages.1.{DOWNSAMPLE}"].weight
    assert down.packed.shape == (WIDTHS[1], WIDTHS[0], 2, 2)

    source = np.asarray(weights["stages.1.blocks.1.mlp.fc1.weight"], dtype=np.float32)
    packed, scale = quantize_fp8_weight(source.reshape(*source.shape, 1, 1))
    assert np.array_equal(fc1.weight.packed, packed)
    assert fc1.weight.weight_scale == pytest.approx(scale)
    assert not ctx.should_quantize("stages.0.blocks.0.conv_dw")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that convolution in fp16 on purpose."""
    depths, weights, _ = tiny
    name = "stages.1.blocks.1.mlp.fc2"
    _patch_scales(monkeypatch, _all_scales(depths, weights, null=(name,)))
    ctx = calibrate_timm_convnext(depths, SIZE, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("stages.0.blocks.0.mlp.fc2")
    assert len(ctx.scales) == sum(depths) * len(PROJECTIONS)


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A convolution missing from the file fails the build and names the calibration tool."""
    depths, weights, _ = tiny
    scales = _all_scales(depths, weights)
    del scales["stages.0.blocks.0.mlp.fc2"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(
        ValueError, match=r"stages\.0\.blocks\.0\.mlp\.fc2.*calibrate_fp8_scales\.py"
    ):
        calibrate_timm_convnext(depths, SIZE, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_projection_is_refused(tiny, monkeypatch) -> None:
    """A table with more blocks than the model is a mismatch, not a silent partial build."""
    depths, weights, _ = tiny
    scales = _all_scales(depths, weights)
    scales["stages.0.blocks.9.mlp.fc1"] = 0.02
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"stages\.0\.blocks\.9\.mlp\.fc1"):
        calibrate_timm_convnext(depths, SIZE, weights, "fp8", precision="fp16")


def test_already_quantized_checkpoint_is_refused(tiny, monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    depths, weights, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(depths, weights))
    tensors = load_file(str(model_dir / "model.safetensors"))
    tensors["stages.0.blocks.0.mlp.fc1.weight_scale"] = np.ones(1, dtype=np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


def test_quantization_config_is_refused(tmp_path: Path) -> None:
    """A config that declares a quantization is refused before any weight is read."""
    _write_tiny_convnext(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"] = {"quant_method": "modelopt"}
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        quantization.refuse_quantized_checkpoint(tmp_path, [])


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    depths, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(depths, weights))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_timm_convnext(depths, SIZE, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    depths, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_convnext(depths, SIZE, weights, "nvfp4", precision="fp16")


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
        task="image_to_class_scores",
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
    depths, weights, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(depths, weights))
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


def test_projection_name_matches_only_quantized_modules() -> None:
    """Only the block MLP linears and the downsample convolutions are calibrated."""
    name = calibrate_fp8_scales.projection_name
    for stem in PROJECTIONS:
        assert name(f"stages.2.blocks.5.{stem}") == f"stages.2.blocks.5.{stem}"
    assert name(f"stages.3.{DOWNSAMPLE}") == f"stages.3.{DOWNSAMPLE}"
    assert name("stages.3.downsample.0") is None  # the norm before the downsample
    assert name("stages.2.blocks.5.conv_dw") is None
    assert name("head.fc") is None
    assert name("stem.0") is None
    assert name("stages.0.blocks.0.mlp.fc1.input_quantizer") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"stages.0.blocks.{block}.mlp.fc2": 0.01 for block in range(6)}
    scales.update({f"stages.0.blocks.{block}.mlp.fc1": 0.02 for block in range(6)})
    scales["stages.0.blocks.3.mlp.fc2"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["stages.0.blocks.3.mlp.fc2"]
    assert table["stages.0.blocks.3.mlp.fc2"] is None
    assert table["stages.0.blocks.2.mlp.fc2"] == 0.01
    assert table["stages.0.blocks.3.mlp.fc1"] == 0.02


def test_outlier_kind_ignores_stage_and_block_index() -> None:
    """The median of a kind runs over all stages, and the downsamples form one kind."""
    scales = {f"stages.{stage}.blocks.0.mlp.fc1": 0.1 for stage in range(4)}
    scales.update({f"stages.{stage}.{DOWNSAMPLE}": 0.2 for stage in (1, 2, 3)})
    scales["stages.3.blocks.0.mlp.fc1"] = 5.0
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["stages.3.blocks.0.mlp.fc1"]
    assert table[f"stages.2.{DOWNSAMPLE}"] == 0.2


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
        np.random.default_rng(0).standard_normal((32, 64, 1, 1)).astype("f")
    )
    ctx = quantization.ConvNeXtQuantContext(
        scales={"m": quantization._LayerScales(0.1, quantization._FP8Weight(packed, scale))}
    )
    out = ctx.convolution(network, activation, "m", np.zeros(32, np.float32))
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
    depths, weights, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(depths, weights))
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
