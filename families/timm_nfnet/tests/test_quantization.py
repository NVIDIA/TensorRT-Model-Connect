# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_nfnet FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

trt = pytest.importorskip("tensorrt")
if not hasattr(trt, "Builder"):  # test_model.py stands in a placeholder module without TensorRT
    pytest.skip("TensorRT is not installed", allow_module_level=True)

from .. import calibrate_fp8_scales, model as model_module, quantization  # noqa: E402
from ..checkpoint import Checkpoint  # noqa: E402
from ..quantization import (  # noqa: E402
    calibrate_timm_nfnet,
    conv_names,
    load_activation_scales,
    quantize_fp8_conv_weight,
    scale_key,
)

CLASSES, IMAGE, WIDTH = 5, 64, 16


def _conv(rng, tensors: dict, prefix: str, out_ch: int, in_ch: int, k: int) -> None:
    tensors[f"{prefix}.weight"] = rng.standard_normal((out_ch, in_ch, k, k)).astype(np.float32)
    tensors[f"{prefix}.gain"] = (1.0 + 0.1 * rng.standard_normal((out_ch, 1, 1, 1))).astype(
        np.float32
    )
    tensors[f"{prefix}.bias"] = (0.1 * rng.standard_normal(out_ch)).astype(np.float32)


def _write_tiny_nfnet(path: Path, seed: int = 0) -> Path:
    """A small dm_nfnet checkpoint: four stem convolutions, stages of depth 1 and 2.

    The first stage has one group in conv2 / conv2b (quantized); the second stage has two
    groups (kept fp16), as in dm_nfnet where the group width is 128.
    """
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "dm_nfnet_f0",
        "num_classes": CLASSES,
        "pretrained_cfg": {"input_size": [3, IMAGE, IMAGE], "crop_pct": 0.9},
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, np.ndarray] = {}
    for position in range(1, 5):
        _conv(rng, tensors, f"stem.conv{position}", WIDTH, 3 if position == 1 else WIDTH, 3)
    for stage, depth in enumerate((1, 2)):
        per_group = WIDTH if stage == 0 else WIDTH // 2
        for index in range(depth):
            prefix = f"stages.{stage}.{index}"
            _conv(rng, tensors, f"{prefix}.conv1", WIDTH, WIDTH, 1)
            _conv(rng, tensors, f"{prefix}.conv2", WIDTH, per_group, 3)
            _conv(rng, tensors, f"{prefix}.conv2b", WIDTH, per_group, 3)
            _conv(rng, tensors, f"{prefix}.conv3", WIDTH, WIDTH, 1)
            tensors[f"{prefix}.attn_last.fc1.weight"] = rng.standard_normal(
                (4, WIDTH, 1, 1)
            ).astype(np.float32)
            tensors[f"{prefix}.attn_last.fc1.bias"] = np.zeros(4, dtype=np.float32)
            tensors[f"{prefix}.attn_last.fc2.weight"] = rng.standard_normal(
                (WIDTH, 4, 1, 1)
            ).astype(np.float32)
            tensors[f"{prefix}.attn_last.fc2.bias"] = np.zeros(WIDTH, dtype=np.float32)
            tensors[f"{prefix}.skipinit_gain"] = np.array(0.5, dtype=np.float32)
            if index == 0:
                _conv(rng, tensors, f"{prefix}.downsample.conv", WIDTH, WIDTH, 1)
    _conv(rng, tensors, "final_conv", 2 * WIDTH, WIDTH, 1)
    tensors["head.fc.weight"] = rng.standard_normal((CLASSES, 2 * WIDTH)).astype(np.float32)
    tensors["head.fc.bias"] = np.zeros(CLASSES, dtype=np.float32)
    save_file(tensors, str(path / "model.safetensors"))
    return path


def _load(path: Path):
    checkpoint = Checkpoint.open(path)
    config = model_module._preprocess_config(model_module._read_config(path))
    return config, model_module._layout(checkpoint), checkpoint


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's config, layout and checkpoint as the FP8 build reads them."""
    model_dir = _write_tiny_nfnet(tmp_path / "model")
    return (*_load(model_dir), model_dir)


def _all_scales(blocks, checkpoint, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in conv_names(blocks, checkpoint)}


def _patch_scales(monkeypatch, scales) -> None:
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)


def _context(tiny, **kwargs):
    config, blocks, checkpoint, _ = tiny
    values = dict(precision="fp16", standardise=model_module.standardise)
    values.update(kwargs)
    return calibrate_timm_nfnet(config, blocks, checkpoint, "fp8", **values)


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


def test_conv_names_keep_the_image_stem_and_grouped_convolutions_out(tiny) -> None:
    """Dense convolutions only: no first stem convolution, no grouped 3x3, no gate."""
    _, blocks, checkpoint, _ = tiny
    names = conv_names(blocks, checkpoint)
    assert names[:3] == ["stem.conv2", "stem.conv3", "stem.conv4"]
    assert names[-1] == "final_conv"
    assert "stages.0.0.conv2" in names and "stages.0.0.conv2b" in names  # one group
    assert "stages.1.0.conv2" not in names and "stages.1.1.conv2b" not in names  # two groups
    assert "stages.1.0.downsample.conv" in names and "stages.1.1.conv3" in names
    assert not any("attn_last" in name for name in names)
    # stem 3 + stage 0 (conv1, conv2, conv2b, conv3, downsample) + stage 1 (3 + 2) + final
    assert len(names) == 3 + 5 + 5 + 1


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path: Path) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    config, blocks, checkpoint, _ = tiny
    key = scale_key(config, blocks, checkpoint)
    assert key.startswith(f"nfnet-d1x2-c{WIDTH}-w{WIDTH}-s{IMAGE}x{IMAGE}-w")
    other = _load(_write_tiny_nfnet(tmp_path / "other", seed=1))
    assert scale_key(*other) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    config, blocks, checkpoint, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(config, blocks, checkpoint)


def test_committed_scales_are_positive_and_named() -> None:
    """Every committed entry has a positive scale, or null, for dense convolutions only."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        assert key.startswith("nfnet-d"), key
        depths = [int(value) for value in key.split("-d")[1].split("-")[0].split("x")]
        blocks = sum(depths)
        # conv1, conv3 per block, a downsample per stage, the stem and the final convolution
        assert len(scales) >= 2 * blocks + len(depths) + 2, key
        assert "stem.conv1" not in scales and "final_conv" in scales, key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_context_quantizes_the_standardised_weights(tiny, monkeypatch) -> None:
    """Each convolution gets its standardised FP8 weight with per-channel scales."""
    _, blocks, checkpoint, _ = tiny
    _patch_scales(monkeypatch, _all_scales(blocks, checkpoint))
    ctx = _context(tiny)
    assert len(ctx.scales) == 14
    assert ctx.should_quantize("stem.conv2") and not ctx.should_quantize("stem.conv1")
    assert not ctx.should_quantize("stages.1.0.conv2")
    entry = ctx.scales["stages.1.0.downsample.conv"]
    assert entry.input_scale == pytest.approx(0.02)
    standardised = model_module.standardise(
        checkpoint.tensor("stages.1.0.downsample.conv.weight"),
        checkpoint.tensor("stages.1.0.downsample.conv.gain"),
        np.float32,
    )
    packed, scale = quantize_fp8_conv_weight(standardised)
    assert np.array_equal(entry.weight.packed, packed)
    np.testing.assert_array_equal(entry.weight.weight_scale, scale)


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that convolution in fp16 on purpose."""
    _, blocks, checkpoint, _ = tiny
    name = "stages.1.1.conv1"
    _patch_scales(monkeypatch, _all_scales(blocks, checkpoint, null=(name,)))
    ctx = _context(tiny)
    assert not ctx.should_quantize(name)
    assert ctx.should_quantize("stages.1.0.conv1")
    assert len(ctx.scales) == 13


def test_convolution_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A convolution missing from the file fails the build and names the calibration tool."""
    _, blocks, checkpoint, _ = tiny
    scales = _all_scales(blocks, checkpoint)
    del scales["stages.1.0.conv3"]
    _patch_scales(monkeypatch, scales)
    with pytest.raises(ValueError, match=r"stages\.1\.0\.conv3.*calibrate_fp8_scales\.py"):
        _context(tiny)


def test_scale_for_an_unknown_convolution_is_refused(tiny, monkeypatch) -> None:
    """A grouped, image-stem or extra-block entry is a mismatch, not a silent partial build."""
    _, blocks, checkpoint, _ = tiny
    for extra in ("stages.1.0.conv2", "stem.conv1", "stages.1.9.conv1"):
        scales = _all_scales(blocks, checkpoint)
        scales[extra] = 0.02
        _patch_scales(monkeypatch, scales)
        with pytest.raises(ValueError, match=extra.replace(".", r"\.")):
            _context(tiny)


def test_bad_scale_values_are_refused(tiny, monkeypatch) -> None:
    """A non-positive scale or one below the smallest normal fp16 value is refused."""
    _, blocks, checkpoint, _ = tiny
    for value, message in ((0.0, "finite and positive"), (1e-6, "smallest normal fp16")):
        scales = _all_scales(blocks, checkpoint)
        scales["final_conv"] = value
        _patch_scales(monkeypatch, scales)
        with pytest.raises(ValueError, match=message):
            _context(tiny)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    _, blocks, checkpoint, _ = tiny
    _patch_scales(monkeypatch, _all_scales(blocks, checkpoint))
    with pytest.raises(ValueError, match="precision fp16"):
        _context(tiny, precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    config, blocks, checkpoint, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_nfnet(
            config,
            blocks,
            checkpoint,
            "nvfp4",
            precision="fp16",
            standardise=model_module.standardise,
        )


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
    model_dir = tiny[-1]
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_fp8_with_fp32_precision(tiny, monkeypatch) -> None:
    """FP8 on the fp32 engine is rejected before any graph is built."""
    _, blocks, checkpoint, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(blocks, checkpoint))
    with pytest.raises(ValueError, match="precision fp16"):
        model_module.build(_request(model_dir, precision="fp32"), _Writer())


def test_build_refuses_other_quantization(tiny) -> None:
    """A quantization other than fp8 or none is rejected."""
    model_dir = tiny[-1]
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model_module.build(_request(model_dir, quantization="nvfp4"), _Writer())


def test_build_refuses_a_quantized_checkpoint(tiny) -> None:
    """A checkpoint with a ModelOpt quantization config is refused before quantizing."""
    model_dir = tiny[-1]
    (model_dir / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already quantized"):
        model_module.build(_request(model_dir), _Writer())


def test_build_refuses_a_checkpoint_with_fp8_scales(tmp_path: Path) -> None:
    """Weight or input scales in the tensors also mark an already quantized checkpoint."""
    model_dir = _write_tiny_nfnet(tmp_path / "model")
    from safetensors.numpy import load_file

    tensors = load_file(str(model_dir / "model.safetensors"))
    tensors["stem.conv2.weight_scale"] = np.ones(1, dtype=np.float32)
    save_file(tensors, str(model_dir / "model.safetensors"))
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


def test_conv_kind_drops_the_stage_block_and_stem_index() -> None:
    """Stem convolutions are one kind; block convolutions are grouped by leaf name."""
    kind = calibrate_fp8_scales.conv_kind
    assert kind("stem.conv2") == kind("stem.conv4") == "stem.conv"
    assert kind("stages.2.5.conv1") == "conv1"
    assert kind("stages.0.0.conv2b") == "conv2b"
    assert kind("stages.3.0.downsample.conv") == "downsample.conv"
    assert kind("final_conv") == "final_conv"


def test_outlier_convolutions_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"stages.2.{block}.conv1": 0.01 for block in range(6)}
    scales.update({f"stages.2.{block}.conv3": 0.02 for block in range(6)})
    scales["stages.2.3.conv1"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["stages.2.3.conv1"]
    assert table["stages.2.3.conv1"] is None
    assert table["stages.2.2.conv1"] == 0.01
    assert table["stages.2.3.conv3"] == 0.02


def test_almost_zero_input_becomes_null() -> None:
    """A scale below the smallest normal fp16 value is stored as null, not as a subnormal."""
    scales = {f"stages.2.{block}.conv3": 0.05 for block in range(6)}
    scales["stages.2.5.conv3"] = 3e-5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["stages.2.5.conv3"]
    assert table["stages.2.5.conv3"] is None and table["stages.2.4.conv3"] == 0.05


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
    _, blocks, checkpoint, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(blocks, checkpoint))
    seen = {}

    def capture(self, network, config):
        seen["layers"] = [network.get_layer(i) for i in range(network.num_layers)]
        return b"plan"

    monkeypatch.setattr(trt.Builder, "build_serialized_network", capture)
    model_module.build(_request(model_dir, quantization=quantization_mode), _Writer())
    return seen["layers"]


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_network_has_one_qdq_per_convolution_input(tiny, monkeypatch) -> None:
    """14 FP8 convolutions; conv1 and the unpooled downsample share one Quantize."""
    layers = _network_layers(tiny, monkeypatch, "fp8")
    kinds = [layer.type for layer in layers]
    # stages.0.0: conv1 and the stride-1 downsample read the same quantized input.
    assert kinds.count(trt.LayerType.QUANTIZE) == 14 - 1
    # One activation Dequantize per Quantize and one weight Dequantize per FP8 convolution.
    assert kinds.count(trt.LayerType.DEQUANTIZE) == (14 - 1) + 14
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    convolutions = [layer for layer in layers if layer.type == trt.LayerType.CONVOLUTION]
    # 4 stem + 3 blocks x 4 + 3 blocks x 2 gate + downsample x 2 + final
    assert len(convolutions) == 4 + 12 + 6 + 2 + 1


@pytest.mark.gpu
@pytest.mark.trt
def test_unquantized_network_has_no_qdq(tiny, monkeypatch) -> None:
    """Without quantization the network has no Quantize or Dequantize layer."""
    kinds = [layer.type for layer in _network_layers(tiny, monkeypatch, None)]
    assert trt.LayerType.QUANTIZE not in kinds and trt.LayerType.DEQUANTIZE not in kinds


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path and keeps its inputs and outputs."""
    _, blocks, checkpoint, model_dir = tiny
    _patch_scales(monkeypatch, _all_scales(blocks, checkpoint))
    plain, quantized = _Writer(), _Writer()
    model_module.build(_request(model_dir, quantization=None), plain)
    model_module.build(_request(model_dir), quantized)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        quantized.files["engine.plan"]
    )
    assert engine is not None
    assert {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)} == {
        "pixel_values",
        "logits",
    }
    assert quantized.files["runtime.json"] == plain.files["runtime.json"]
