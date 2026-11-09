# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_resnet FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

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
    calibrate_timm_resnet,
    conv_names,
    fold_batch_norm,
    load_activation_scales,
    norm_name,
    quantize_fp8_conv_weight,
    scale_key,
)

EPS = 1e-5
CLASSES, IMAGE, STEM = 16, 64, 32


def _rand(rng, *shape, scale=0.05):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def _bn(rng, tensors: dict, prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = 1.0 + _rand(rng, channels)
    tensors[f"{prefix}.bias"] = _rand(rng, channels)
    tensors[f"{prefix}.running_mean"] = _rand(rng, channels)
    tensors[f"{prefix}.running_var"] = 1.0 + np.abs(_rand(rng, channels))


def _write_tiny_resnet(path: Path, seed: int = 0, *, bottleneck: bool = True) -> None:
    """Write a small timm ResNet checkpoint (64 px input, one block per stage) and config."""
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "resnet50" if bottleneck else "resnet18",
        "num_classes": CLASSES,
        "pretrained_cfg": {"input_size": [3, IMAGE, IMAGE]},
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {"conv1.weight": _rand(rng, STEM, 3, 7, 7)}
    _bn(rng, tensors, "bn1", STEM)
    in_ch = STEM
    for stage in range(4):
        width = 32 << stage
        out_ch = 4 * width if bottleneck else width
        prefix = f"layer{stage + 1}.0"
        if bottleneck:
            shapes = {"conv1": (width, in_ch, 1, 1), "conv2": (width, width, 3, 3)}
            shapes["conv3"] = (out_ch, width, 1, 1)
        else:
            shapes = {"conv1": (out_ch, in_ch, 3, 3), "conv2": (out_ch, out_ch, 3, 3)}
        for index, (conv, shape) in enumerate(shapes.items(), start=1):
            tensors[f"{prefix}.{conv}.weight"] = _rand(rng, *shape)
            _bn(rng, tensors, f"{prefix}.bn{index}", shape[0])
        if in_ch != out_ch or stage:
            tensors[f"{prefix}.downsample.0.weight"] = _rand(rng, out_ch, in_ch, 1, 1)
            _bn(rng, tensors, f"{prefix}.downsample.1", out_ch)
        in_ch = out_ch
    tensors["fc.weight"] = _rand(rng, CLASSES, in_ch)
    tensors["fc.bias"] = _rand(rng, CLASSES)
    save_file(tensors, str(path / "model.safetensors"))


def _load(path: Path):
    config = ModelConfig.from_dir(path)
    weights = model_module._TimmResnetModel().load_weights(str(path), config, precision="fp16")
    return config.raw["_timm_resnet_config"], weights


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's layout and weights as the FP8 build loads them."""
    _write_tiny_resnet(tmp_path)
    cfg, weights = _load(tmp_path)
    return cfg, weights, tmp_path


def _all_scales(cfg, weights, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in conv_names(cfg, weights)}


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def test_quantize_fp8_conv_weight_scales_per_output_channel() -> None:
    """Each output channel gets amax / 448 and the bytes decode back to the weight."""
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((8, 4, 3, 3)).astype(np.float32)
    weight[3] *= 100.0  # one channel far larger must not coarsen the others
    packed, scale = quantize_fp8_conv_weight(weight)
    amax = np.abs(weight).reshape(8, -1).max(axis=1)
    np.testing.assert_allclose(scale, amax / 448.0, rtol=1e-6)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale.reshape(-1, 1, 1, 1)
    error = np.abs(restored - weight).reshape(8, -1).max(axis=1)
    assert np.all(error <= amax * 0.07)  # e4m3 half-step relative to each channel


def test_quantize_fp8_conv_weight_zero_channel_and_zero_tensor() -> None:
    """A zero channel gets scale 1 and stays zero; an all-zero weight is refused."""
    weight = np.ones((2, 1, 1, 1), dtype=np.float32)
    weight[1] = 0.0
    packed, scale = quantize_fp8_conv_weight(weight)
    assert scale[1] == 1.0
    assert packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)[1].item() == 0.0
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_conv_weight(np.zeros((2, 2, 1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="4-D"):
        quantize_fp8_conv_weight(np.ones((2, 2), dtype=np.float32))


def test_fold_batch_norm_matches_conv_then_norm() -> None:
    """A 1x1 convolution with the folded weight and shift equals convolution then batch norm."""
    rng = np.random.default_rng(1)
    weight, x = _rand(rng, 6, 4, 1, 1, scale=1.0), _rand(rng, 4, 10, scale=1.0)
    gamma, beta, mean = _rand(rng, 6, scale=1.0), _rand(rng, 6), _rand(rng, 6)
    variance = 1.0 + np.abs(_rand(rng, 6))
    folded, shift = fold_batch_norm(weight, gamma, beta, mean, variance, EPS)
    conv = weight[:, :, 0, 0] @ x
    expected = (conv - mean[:, None]) / np.sqrt(variance[:, None] + EPS) * gamma[:, None]
    expected += beta[:, None]
    np.testing.assert_allclose(
        folded[:, :, 0, 0] @ x + shift[:, None], expected, rtol=1e-5, atol=1e-6
    )


def test_norm_name_follows_the_convolution() -> None:
    """conv<k> is followed by bn<k>, downsample.0 by downsample.1."""
    assert norm_name("layer2.0.conv1") == "layer2.0.bn1"
    assert norm_name("layer4.2.conv3") == "layer4.2.bn3"
    assert norm_name("layer3.0.downsample.0") == "layer3.0.downsample.1"


def test_conv_names_skip_the_first_stage(tiny) -> None:
    """Three convolutions per bottleneck block and the downsample of stages 2 to 4 only."""
    cfg, weights, _ = tiny
    names = conv_names(cfg, weights)
    assert len(names) == 3 * (3 + 1)
    assert "layer2.0.downsample.0" in names and "layer4.0.conv3" in names
    assert not any(name.startswith("layer1.") for name in names)


def test_conv_names_of_basic_blocks(tmp_path: Path) -> None:
    """A basic-block model has two convolutions per block."""
    _write_tiny_resnet(tmp_path, bottleneck=False)
    cfg, weights = _load(tmp_path)
    names = conv_names(cfg, weights)
    assert "layer2.0.conv2" in names and "layer2.0.conv3" not in names
    assert len(names) == 3 * (2 + 1)


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    cfg, weights, _ = tiny
    key = scale_key(cfg, weights)
    assert key.startswith(f"resnet-b1x1x1x1-bottleneck-c{STEM}-w32-g1-s64x64-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_resnet(other, seed=1)
    other_cfg, other_weights = _load(other)
    assert scale_key(other_cfg, other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    cfg, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(cfg, weights)


def test_committed_scales_cover_every_convolution() -> None:
    """Every committed entry has a positive scale, or null, for every convolution of its model."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        blocks = [int(value) for value in key.split("-b")[1].split("-")[0].split("x")]
        cfg = {"blocks": blocks, "bottleneck": "-bottleneck-" in key}
        # Every stage of the committed models starts with a downsample block.
        fake = {f"layer{stage}.0.downsample.0.weight": None for stage in range(1, 5)}
        assert set(scales) == set(conv_names(cfg, fake)), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_folded_weights(tiny, monkeypatch) -> None:
    """Each convolution gets its BN-folded FP8 weight with per-channel scales and its shift."""
    cfg, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(cfg, weights))
    ctx = calibrate_timm_resnet(cfg, weights, "fp8", precision="fp16", eps=EPS)

    assert len(ctx.scales) == 12
    assert ctx.should_quantize("layer2.0.conv1")
    assert not ctx.should_quantize("layer1.0.conv1")  # fp16, see quantization.py
    entry = ctx.scales["layer3.0.downsample.0"]
    assert entry.input_scale == pytest.approx(0.02)
    folded, shift = fold_batch_norm(
        weights["layer3.0.downsample.0.weight"],
        *(weights[f"layer3.0.downsample.1.{leaf}"] for leaf in ("weight", "bias")),
        weights["layer3.0.downsample.1.running_mean"],
        weights["layer3.0.downsample.1.running_var"],
        EPS,
    )
    packed, scale = quantize_fp8_conv_weight(folded)
    assert np.array_equal(entry.weight.packed, packed)
    np.testing.assert_array_equal(entry.weight.weight_scale, scale)
    np.testing.assert_array_equal(entry.weight.bias, shift)


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that convolution in fp16 on purpose."""
    cfg, weights, _ = tiny
    name = "layer3.0.conv2"
    _patch_scales(monkeypatch, _all_scales(cfg, weights, null=(name,)))
    ctx = calibrate_timm_resnet(cfg, weights, "fp8", precision="fp16", eps=EPS)
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("layer2.0.conv2")
    assert len(ctx.scales) == 11


def test_convolution_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A convolution missing from the file fails the build and names the calibration tool."""
    cfg, weights, _ = tiny
    scales = _all_scales(cfg, weights)
    del scales["layer2.0.conv3"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"layer2\.0\.conv3.*calibrate_fp8_scales\.py"):
        calibrate_timm_resnet(cfg, weights, "fp8", precision="fp16", eps=EPS)


def test_scale_for_an_unknown_convolution_is_refused(tiny, monkeypatch) -> None:
    """A table with a first-stage or extra-block entry is a mismatch, not a silent partial build."""
    cfg, weights, _ = tiny
    for extra in ("layer1.0.conv1", "layer2.9.conv1"):
        scales = _all_scales(cfg, weights)
        scales[extra] = 0.02
        _patch_scales(monkeypatch, scales)
        with pytest.raises(ValueError, match=extra.replace(".", r"\.")):
            calibrate_timm_resnet(cfg, weights, "fp8", precision="fp16", eps=EPS)


def test_already_quantized_weights_are_refused(tiny, monkeypatch) -> None:
    """Weights that carry FP8 scales are refused instead of quantized twice."""
    cfg, weights, _ = tiny
    weights["layer2.0.conv1.weight_scale"] = np.ones(1, dtype=np.float32)
    _patch_scales(monkeypatch, _all_scales(cfg, weights))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_timm_resnet(cfg, weights, "fp8", precision="fp16", eps=EPS)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    cfg, weights, _ = tiny
    _patch_scales(monkeypatch, _all_scales(cfg, weights))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_timm_resnet(cfg, weights, "fp8", precision=precision, eps=EPS)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    cfg, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_resnet(cfg, weights, "nvfp4", precision="fp16", eps=EPS)


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
    cfg, weights, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(cfg, weights))
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision="fp32"), _Writer())


def test_build_refuses_other_quantization(tiny) -> None:
    """A quantization other than fp8 or none is rejected."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model_module.build(_request(model_dir, quantization="nvfp4"), _Writer())


def test_build_refuses_a_quantized_checkpoint(tiny) -> None:
    """A checkpoint with a ModelOpt quantization config is refused before quantizing."""
    *_, model_dir = tiny
    (model_dir / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


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
    """Only the convolutions of stages 2 to 4 are calibrated."""
    name = calibrate_fp8_scales.conv_name
    for leaf in ("conv1", "conv2", "conv3", "downsample.0"):
        assert name(f"layer3.5.{leaf}") == f"layer3.5.{leaf}"
    assert name("layer1.0.conv1") is None
    assert name("conv1") is None
    assert name("fc") is None
    assert name("layer2.0.bn1") is None
    assert name("layer2.0.downsample.1") is None
    assert name("layer2.0.conv1.input_quantizer") is None


def test_outlier_convolutions_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"layer3.{block}.conv2": 0.01 for block in range(6)}
    scales.update({f"layer3.{block}.conv3": 0.02 for block in range(6)})
    scales["layer3.3.conv2"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer3.3.conv2"]
    assert table["layer3.3.conv2"] is None
    assert table["layer3.2.conv2"] == 0.01
    assert table["layer3.3.conv3"] == 0.02


def test_outlier_kind_ignores_stage_and_block_index() -> None:
    """The median of a kind runs over all stages and blocks."""
    scales = {f"layer{stage}.0.conv1": 0.1 for stage in (2, 3, 4)}
    scales.update({f"layer{stage}.1.conv1": 0.1 for stage in (2, 3, 4)})
    scales.update({f"layer{stage}.0.downsample.0": 0.2 for stage in (2, 3, 4)})
    scales["layer4.1.conv1"] = 5.0
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["layer4.1.conv1"]
    assert table["layer3.0.downsample.0"] == 0.2


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


def _network_layers(tiny, monkeypatch, quantization_mode):
    """Layers of the network the family builds (the plan build itself is skipped)."""
    cfg, weights, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(cfg, weights))
    seen = {}

    def capture(self, network, config):
        seen["layers"] = [network.get_layer(i) for i in range(network.num_layers)]
        return b"plan"

    monkeypatch.setattr(trt.Builder, "build_serialized_network", capture)
    model_module.build(_request(model_dir, quantization=quantization_mode), _Writer())
    return seen["layers"]


@pytest.mark.gpu
@pytest.mark.trt
def test_block_input_is_quantized_once(tiny, monkeypatch) -> None:
    """conv1, the downsample and the residual identity share one Quantize per block."""
    layers = _network_layers(tiny, monkeypatch, "fp8")
    kinds = [layer.type for layer in layers]
    # Three quantized bottleneck blocks: block input (shared), conv2 input, conv3 input.
    assert kinds.count(trt.LayerType.QUANTIZE) == 3 * 3
    # One activation and one weight Dequantize per quantized convolution, plus the shared one.
    assert kinds.count(trt.LayerType.DEQUANTIZE) == 3 * 3 + 12
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert trt.LayerType.CAST in kinds  # only the input and output casts of the fp16 graph
    assert kinds.count(trt.LayerType.CAST) == 2


@pytest.mark.gpu
@pytest.mark.trt
def test_unquantized_network_has_no_qdq(tiny, monkeypatch) -> None:
    """Without quantization the network has no Quantize or Dequantize layer."""
    kinds = [layer.type for layer in _network_layers(tiny, monkeypatch, None)]
    assert trt.LayerType.QUANTIZE not in kinds and trt.LayerType.DEQUANTIZE not in kinds


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path, stores FP8 weights and is smaller."""
    cfg, weights, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(cfg, weights))
    plain, quantized = _Writer(), _Writer()
    model_module.build(_request(model_dir, quantization=None), plain)
    model_module.build(_request(model_dir), quantized)

    plain_plan, fp8_plan = plain.files["engine.plan"], quantized.files["engine.plan"]
    assert len(fp8_plan) < 0.9 * len(plain_plan)
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(fp8_plan)
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "pixel_values",
        "logits",
    }
