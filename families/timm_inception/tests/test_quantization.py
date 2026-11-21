# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_inception FP8 self-quantization: weight quantizer, scale lookup, context, tool, engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file


try:
    import tensorrt as trt
except ModuleNotFoundError:
    trt = sys.modules.setdefault("tensorrt", SimpleNamespace())

from families.timm_inception import calibrate_fp8_scales, model as model_module, quantization  # noqa: E402
from families.timm_inception.config import ModelConfig  # noqa: E402
from families.timm_inception.quantization import (  # noqa: E402
    FP16_MIN_NORMAL,
    create_quant_context,
    fold_batch_norm,
    load_activation_scales,
    quantize_fp8_weight,
    quantized_convolutions,
    scale_key,
)
from families.timm_inception.weights import _open_safetensors  # noqa: E402


HAS_TRT = hasattr(trt, "Builder")
WIDTH = 16
# The tiny model at 75x75: (prefix, input channels in units of WIDTH, kernel) of every
# convolution. The first convolution reads the 3 image channels. One Mixed block of each
# topology: InceptionA (7x7), B (to 3x3), C, D (to 1x1), E.
_CONVS = (
    ("Conv2d_1a_3x3", 0, (3, 3)),
    ("Conv2d_2a_3x3", 1, (3, 3)),
    ("Conv2d_2b_3x3", 1, (3, 3)),
    ("Conv2d_3b_1x1", 1, (1, 1)),
    ("Conv2d_4a_3x3", 1, (3, 3)),
    ("Mixed_5b.branch1x1", 1, (1, 1)),
    ("Mixed_5b.branch5x5_1", 1, (1, 1)),
    ("Mixed_5b.branch5x5_2", 1, (5, 5)),
    ("Mixed_5b.branch3x3dbl_1", 1, (1, 1)),
    ("Mixed_5b.branch3x3dbl_2", 1, (3, 3)),
    ("Mixed_5b.branch3x3dbl_3", 1, (3, 3)),
    ("Mixed_5b.branch_pool", 1, (1, 1)),
    ("Mixed_6a.branch3x3", 4, (3, 3)),
    ("Mixed_6a.branch3x3dbl_1", 4, (1, 1)),
    ("Mixed_6a.branch3x3dbl_2", 1, (3, 3)),
    ("Mixed_6a.branch3x3dbl_3", 1, (3, 3)),
    ("Mixed_6b.branch1x1", 6, (1, 1)),
    ("Mixed_6b.branch7x7_1", 6, (1, 1)),
    ("Mixed_6b.branch7x7_2", 1, (1, 7)),
    ("Mixed_6b.branch7x7_3", 1, (7, 1)),
    ("Mixed_6b.branch7x7dbl_1", 6, (1, 1)),
    ("Mixed_6b.branch7x7dbl_2", 1, (7, 1)),
    ("Mixed_6b.branch7x7dbl_3", 1, (1, 7)),
    ("Mixed_6b.branch7x7dbl_4", 1, (7, 1)),
    ("Mixed_6b.branch7x7dbl_5", 1, (1, 7)),
    ("Mixed_6b.branch_pool", 6, (1, 1)),
    ("Mixed_7a.branch3x3_1", 4, (1, 1)),
    ("Mixed_7a.branch3x3_2", 1, (3, 3)),
    ("Mixed_7a.branch7x7x3_1", 4, (1, 1)),
    ("Mixed_7a.branch7x7x3_2", 1, (1, 7)),
    ("Mixed_7a.branch7x7x3_3", 1, (7, 1)),
    ("Mixed_7a.branch7x7x3_4", 1, (3, 3)),
    ("Mixed_7b.branch1x1", 6, (1, 1)),
    ("Mixed_7b.branch3x3_1", 6, (1, 1)),
    ("Mixed_7b.branch3x3_2a", 1, (1, 3)),
    ("Mixed_7b.branch3x3_2b", 1, (3, 1)),
    ("Mixed_7b.branch3x3dbl_1", 6, (1, 1)),
    ("Mixed_7b.branch3x3dbl_2", 1, (3, 3)),
    ("Mixed_7b.branch3x3dbl_3a", 1, (1, 3)),
    ("Mixed_7b.branch3x3dbl_3b", 1, (3, 1)),
    ("Mixed_7b.branch_pool", 6, (1, 1)),
)
QUANTIZED = sorted(prefix for prefix, *_ in _CONVS if prefix.split(".")[0] >= "Mixed_6")


def _checkpoint(model_dir: Path, *, image: int = 75) -> Path:
    rng = np.random.default_rng(3)
    tensors: dict[str, np.ndarray] = {}
    for prefix, units, (height, width) in _CONVS:
        inputs = units * WIDTH or 3
        tensors[f"{prefix}.conv.weight"] = (
            rng.standard_normal((WIDTH, inputs, height, width)) * 0.3
        ).astype(np.float32)
        tensors[f"{prefix}.bn.weight"] = rng.uniform(0.5, 1.5, WIDTH).astype(np.float32)
        tensors[f"{prefix}.bn.bias"] = rng.standard_normal(WIDTH).astype(np.float32) * 0.1
        tensors[f"{prefix}.bn.running_mean"] = rng.standard_normal(WIDTH).astype(np.float32) * 0.1
        tensors[f"{prefix}.bn.running_var"] = rng.uniform(0.5, 1.5, WIDTH).astype(np.float32)
    tensors["fc.weight"] = rng.standard_normal((5, 6 * WIDTH)).astype(np.float32)
    tensors["fc.bias"] = rng.standard_normal(5).astype(np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
    config = {
        "architecture": "inception_v3",
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, image, image],
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def _config(model_dir: Path) -> dict:
    """The resolved family config (input size and Mixed block layout) the build uses."""
    raw = ModelConfig.from_dir(model_dir).raw
    reader = _open_safetensors(model_dir)
    return {**model_module._resolve_config(raw), **model_module._discover_layout(reader)}


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's resolved config, reader and directory."""
    model_dir = _checkpoint(tmp_path)
    return _config(model_dir), _open_safetensors(model_dir), model_dir


def _scales(value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in QUANTIZED}


def _context(config, reader, precision="fp16", quantization_name="fp8"):
    return create_quant_context(
        config, reader, quantization_name, precision=precision, bn_eps=model_module._BN_EPS
    )


def test_tiny_layout_has_every_block_topology(tiny) -> None:
    """The tiny checkpoint walks the five Mixed block topologies of the real model."""
    config, _, _ = tiny
    assert [block["kind"] for block in config["blocks"]] == [
        "inception_a",
        "inception_b",
        "inception_c",
        "inception_d",
        "inception_e",
    ]


def test_quantize_fp8_weight_scales_each_output_channel() -> None:
    """Each output channel gets amax / 448 and its bytes decode back to the weight."""
    weight = np.random.default_rng(0).standard_normal((8, 4, 3, 3)).astype(np.float32)
    weight[3] *= 0.1
    packed, scales = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scales.dtype == np.float16 and scales.shape == (8,)
    amax = np.abs(weight.reshape(8, -1)).max(axis=1)
    np.testing.assert_allclose(scales.astype(np.float32), amax / 448.0, rtol=1e-3)
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scales.astype(
        np.float32
    ).reshape(-1, 1, 1, 1)
    error = np.abs(restored - weight).reshape(8, -1).max(axis=1)
    assert np.all(error <= amax * 0.07)  # e4m3 half-step, per channel


def test_quantize_fp8_weight_handles_zero_and_tiny_channels() -> None:
    """An all-zero channel gets scale 1; a tiny one is raised to the smallest normal fp16."""
    weight = np.random.default_rng(1).standard_normal((3, 2, 1, 1)).astype(np.float32)
    weight[0] = 0.0
    weight[1] = 1e-4
    packed, scales = quantize_fp8_weight(weight)
    assert float(scales[0]) == 1.0
    assert float(scales[1]) == pytest.approx(FP16_MIN_NORMAL)
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    assert np.all(decoded[0] == 0.0)
    assert np.allclose(decoded[1] * FP16_MIN_NORMAL, 1e-4, rtol=0.07)


@pytest.mark.parametrize("value", [0.0, np.nan])
def test_quantize_fp8_weight_rejects_unusable_tensors(value) -> None:
    """An all-zero or non-finite weight is refused."""
    with pytest.raises(ValueError):
        quantize_fp8_weight(np.full((2, 2, 1, 1), value, dtype=np.float32))


def test_fold_batch_norm_matches_the_unfolded_layers(tiny) -> None:
    """The folded weight and bias reproduce convolution, then batch norm."""
    _, reader, _ = tiny
    prefix = "Mixed_6b.branch1x1"
    weight, bias = fold_batch_norm(reader, prefix, 1e-3)
    gamma = reader.get_tensor(f"{prefix}.bn.weight")
    variance = reader.get_tensor(f"{prefix}.bn.running_var")
    scale = gamma / np.sqrt(variance + 1e-3)
    raw = reader.get_tensor(f"{prefix}.conv.weight")
    np.testing.assert_allclose(weight, raw * scale.reshape(-1, 1, 1, 1), rtol=1e-6)
    mean = reader.get_tensor(f"{prefix}.bn.running_mean")
    np.testing.assert_allclose(
        bias, reader.get_tensor(f"{prefix}.bn.bias") - mean * scale, rtol=1e-6
    )


def test_quantized_convolutions_skip_the_stem_and_inception_a(tiny) -> None:
    """Only the convolutions of the InceptionB to InceptionE blocks are quantized."""
    config, reader, _ = tiny
    assert quantized_convolutions(config["blocks"], reader.keys()) == QUANTIZED


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path: Path) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    config, reader, _ = tiny
    key = scale_key(config, reader)
    assert key.startswith(f"inception_v3-abcde-n{len(QUANTIZED)}-s75x75-w")
    assert scale_key({**config, "image_size_h": 99}, reader) != key
    other = tmp_path / "other"
    other.mkdir()
    _checkpoint(other)
    tensors = dict(load_file(str(other / "model.safetensors")))
    tensors["Conv2d_1a_3x3.conv.weight"] *= 2.0
    save_file(tensors, str(other / "model.safetensors"))
    assert scale_key(_config(other), _open_safetensors(other)) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    config, reader, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(scale_key(config, reader))


def test_committed_scales_cover_the_quantized_convolutions() -> None:
    """Every committed entry has a positive scale for every convolution from Mixed_6a on."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert key.startswith("inception_v3-aaabccccdee-n68-s299x299-"), key
        assert len(scales) == 68, key
        assert all(name.split(".")[0] >= "Mixed_6a" for name in scales), key
        assert all(value is None or value >= FP16_MIN_NORMAL for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_folded_weights(tiny, monkeypatch) -> None:
    """Each quantized convolution holds its folded weight in FP8 and its folded fp16 bias."""
    config, reader, _ = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _scales())
    ctx = _context(config, reader)
    assert sorted(ctx.convolutions) == QUANTIZED
    assert not ctx.should_quantize("Mixed_5b.branch1x1")
    assert not ctx.should_quantize("Conv2d_4a_3x3")
    weight, bias = fold_batch_norm(reader, "Mixed_7a.branch7x7x3_2", model_module._BN_EPS)
    packed, scales = quantize_fp8_weight(weight)
    conv = ctx.convolutions["Mixed_7a.branch7x7x3_2"]
    assert np.array_equal(conv.packed, packed)
    assert np.array_equal(conv.weight_scales, scales)
    assert np.array_equal(conv.bias, bias.astype(np.float16))
    assert conv.packed.shape == (WIDTH, WIDTH, 1, 7)
    assert conv.input_scale == pytest.approx(0.02, rel=1e-3)


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (outlier input) leaves that convolution in fp16 on purpose."""
    config, reader, _ = tiny
    scales = _scales(null=("Mixed_7b.branch3x3_2a",))
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    ctx = _context(config, reader)
    assert not ctx.should_quantize("Mixed_7b.branch3x3_2a")
    assert len(ctx.convolutions) == len(QUANTIZED) - 1


def test_convolution_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A convolution missing from the file fails the build and names the calibration tool."""
    config, reader, _ = tiny
    scales = _scales()
    del scales["Mixed_6b.branch7x7dbl_5"]
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"branch7x7dbl_5.*calibrate_fp8_scales\.py"):
        _context(config, reader)


def test_scale_for_an_unknown_convolution_is_refused(tiny, monkeypatch) -> None:
    """An InceptionA convolution in the table is a mismatch, not a partial build."""
    config, reader, _ = tiny
    scales = {**_scales(), "Mixed_5b.branch1x1": 0.02}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"Mixed_5b\.branch1x1"):
        _context(config, reader)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), 1e-6])
def test_unusable_scale_is_refused(tiny, monkeypatch, value) -> None:
    """A zero, negative, NaN or sub-fp16 scale is refused."""
    config, reader, _ = tiny
    scales = {**_scales(), "Mixed_6a.branch3x3": value}
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"Mixed_6a\.branch3x3"):
        _context(config, reader)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    config, reader, _ = tiny
    with pytest.raises(ValueError, match="precision fp16"):
        _context(config, reader, precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    config, reader, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        _context(config, reader, quantization_name="nvfp4")


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


@pytest.mark.parametrize("marker", ["tensor", "config", "file"])
def test_build_refuses_an_already_quantized_checkpoint(tiny, marker) -> None:
    """Scale tensors, ``quantization_config`` or ``hf_quant_config.json`` mean already quantized."""
    *_, model_dir = tiny
    if marker == "tensor":
        tensors = dict(load_file(str(model_dir / "model.safetensors")))
        tensors["Mixed_6a.branch3x3.conv.input_scale"] = np.ones(1, dtype=np.float32)
        save_file(tensors, str(model_dir / "model.safetensors"))
    elif marker == "config":
        path = model_dir / "config.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["quantization_config"] = {"quant_method": "modelopt"}
        path.write_text(json.dumps(raw), encoding="utf-8")
    else:
        (model_dir / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


class _FakeBuilderConfig:
    """Stands in for IBuilderConfig; only the attribute the build option touches."""

    profiling_verbosity = None


@pytest.mark.skipif(not HAS_TRT, reason="needs TensorRT")
def test_fp8_build_keeps_layer_details_only_for_quantized_builds() -> None:
    """A quantized build asks for detailed profiling; an unquantized build is left alone."""
    quantized = _FakeBuilderConfig()
    quantization.configure_fp8_build(quantized, object())
    assert quantized.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    plain = _FakeBuilderConfig()
    quantization.configure_fp8_build(plain, None)
    assert plain.profiling_verbosity is None


@pytest.mark.skipif(not HAS_TRT, reason="needs TensorRT")
def test_build_refuses_an_unknown_model_config(tiny) -> None:
    """The committed table has no entry for the tiny model, so the FP8 build raises."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


# --- calibration tool ---------------------------------------------------------------------


def test_convolution_name_maps_timm_modules_to_checkpoint_prefixes() -> None:
    """timm's ``<prefix>.conv`` Conv2d becomes ``<prefix>``; other modules are skipped."""
    name = calibrate_fp8_scales.convolution_name
    assert name("Mixed_6b.branch7x7_2.conv") == "Mixed_6b.branch7x7_2"
    assert name("Mixed_6b.branch7x7_2.bn") is None
    assert name("Conv2d_4a_3x3.conv") is None
    assert name("fc") is None
    assert name("Mixed_6b.branch1x1.conv.input_quantizer") is None


def test_outlier_inputs_become_null() -> None:
    """A scale above 20x the median is stored as null, the others are kept."""
    scales = {f"Mixed_6{letter}.branch1x1": 0.05 for letter in "bcde"}
    scales.update({f"Mixed_7{letter}.branch1x1": 0.05 for letter in "bc"})
    scales["Mixed_7c.branch1x1"] = 2.0
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["Mixed_7c.branch1x1"]
    assert table["Mixed_7c.branch1x1"] is None
    assert table["Mixed_7b.branch1x1"] == 0.05


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
def test_shared_input_gets_one_quantize_and_fp16_dequantize(tiny, monkeypatch) -> None:
    """Two convolutions and the pooling branch reading one tensor share one Quantize."""
    config, reader, _ = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _scales())
    ctx = _context(config, reader)
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tensor = network.add_input("x", trt.float16, (1, 6 * WIDTH, 3, 3))
    pooled = ctx.dequantized_input(network, tensor, "Mixed_6b.branch_pool")
    first = ctx.convolution(network, tensor, "Mixed_6b.branch1x1")
    second = ctx.convolution(network, tensor, "Mixed_6b.branch7x7_1")
    assert pooled is ctx.dequantized_input(network, tensor, "Mixed_6b.branch7x7dbl_1")
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert [layer.type for layer in layers].count(trt.LayerType.QUANTIZE) == 1
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 3  # the shared input and the two weights
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert first.get_output(0).dtype == second.get_output(0).dtype == trt.float16


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_convolutions(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path and keeps the same I/O tensors."""
    *_, model_dir = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _scales())
    plain, quantized = _Writer(), _Writer()
    model_module.build(_request(model_dir, quantization=None), plain)
    model_module.build(_request(model_dir), quantized)
    assert plain.files["runtime.json"] == quantized.files["runtime.json"]

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        quantized.files["engine.plan"]
    )
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "pixel_values",
        "logits",
    }
    information = engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    assert "e4m3" in information.lower() or "fp8" in information.lower()
