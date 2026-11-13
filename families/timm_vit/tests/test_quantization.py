# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm_vit FP8 self-quantization: weight quantizer, scale lookup, context, calibration tool, engine."""

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
    calibrate_timm_vit,
    load_activation_scales,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)

# Wide enough that FP8 weight savings dominate the per-layer Q/DQ plan overhead.
HIDDEN, MLP, LAYERS, HEADS, CLASSES = 128, 256, 2, 4, 16


def _rand(rng, *shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny_vit(path: Path, seed: int = 0) -> None:
    """Write a small timm ViT checkpoint (224 px, 16 px patches) and its config."""
    rng = np.random.default_rng(seed)
    config = {
        "architecture": "vit_base_patch16_224",
        "input_size": [3, 224, 224],
        "patch_size": 16,
        "num_features": HIDDEN,
        "depth": LAYERS,
        "num_heads": HEADS,
        "mlp_ratio": MLP / HIDDEN,
        "num_classes": CLASSES,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "patch_embed.proj.weight": _rand(rng, HIDDEN, 3, 16, 16),
        "patch_embed.proj.bias": _rand(rng, HIDDEN),
        "cls_token": _rand(rng, 1, 1, HIDDEN),
        "pos_embed": _rand(rng, 1, 197, HIDDEN),
        "norm.weight": np.ones(HIDDEN, np.float32),
        "norm.bias": _rand(rng, HIDDEN),
        "head.weight": _rand(rng, CLASSES, HIDDEN),
        "head.bias": _rand(rng, CLASSES),
    }
    for layer in range(LAYERS):
        prefix = f"blocks.{layer}"
        for norm in ("norm1", "norm2"):
            tensors[f"{prefix}.{norm}.weight"] = np.ones(HIDDEN, np.float32)
            tensors[f"{prefix}.{norm}.bias"] = _rand(rng, HIDDEN)
        tensors[f"{prefix}.attn.qkv.weight"] = _rand(rng, 3 * HIDDEN, HIDDEN)
        tensors[f"{prefix}.attn.qkv.bias"] = _rand(rng, 3 * HIDDEN)
        tensors[f"{prefix}.attn.proj.weight"] = _rand(rng, HIDDEN, HIDDEN)
        tensors[f"{prefix}.attn.proj.bias"] = _rand(rng, HIDDEN)
        tensors[f"{prefix}.mlp.fc1.weight"] = _rand(rng, MLP, HIDDEN)
        tensors[f"{prefix}.mlp.fc1.bias"] = _rand(rng, MLP)
        tensors[f"{prefix}.mlp.fc2.weight"] = _rand(rng, HIDDEN, MLP)
        tensors[f"{prefix}.mlp.fc2.bias"] = _rand(rng, HIDDEN)
    save_file(tensors, str(path / "model.safetensors"))


@pytest.fixture
def tiny(tmp_path: Path):
    """The tiny model's config, resolved geometry and fp16 weights as the build loads them."""
    _write_tiny_vit(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model_module._TimmVitModel().load_weights(str(tmp_path), config, precision="fp16")
    return config, config.raw["_timm_vit_config"], weights, tmp_path


def _all_scales(vit_cfg: dict, value: float = 0.02, null: tuple[str, ...] = ()):
    return {name: (None if name in null else value) for name in projection_names(vit_cfg)}


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


def test_scale_key_names_the_model_config_and_weights(tiny, tmp_path_factory) -> None:
    """The key holds the config values and numeric weight sums, so other weights get another key."""
    _, vit_cfg, weights, _ = tiny
    key = scale_key(vit_cfg, weights)
    assert key.startswith(f"vit-h{HIDDEN}-l{LAYERS}-i{MLP}-n{HEADS}-p16x16-s224x224-w")
    other = tmp_path_factory.mktemp("other")
    _write_tiny_vit(other, seed=1)
    other_config = ModelConfig.from_dir(other)
    other_weights = model_module._TimmVitModel().load_weights(
        str(other), other_config, precision="fp16"
    )
    assert scale_key(other_config.raw["_timm_vit_config"], other_weights) != key


def test_unknown_model_config_is_refused(tiny) -> None:
    """A model config without committed scales fails and names the calibration tool."""
    _, vit_cfg, weights, _ = tiny
    with pytest.raises(NotImplementedError, match=r"calibrate_fp8_scales\.py"):
        load_activation_scales(vit_cfg, weights)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed entry has a positive scale, or null, for every projection of its blocks."""
    path = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    assert table
    for key, scales in table.items():
        depth = int(key.split("-l")[1].split("-")[0])
        assert set(scales) == set(projection_names({"depth": depth})), key
        assert all(value is None or value > 0 for value in scales.values()), key
        assert any(value is not None for value in scales.values()), key


def test_every_attention_and_mlp_linear_is_a_quantized_module() -> None:
    """qkv, the attention output projection and both MLP projections are quantized."""
    assert PROJECTIONS == ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def test_context_quantizes_the_loaded_weights(tiny, monkeypatch) -> None:
    """Each module gets an FP8 weight of the loaded shape; qkv gives three, one per q, k, v."""
    _, vit_cfg, weights, _ = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(vit_cfg))
    ctx = calibrate_timm_vit(vit_cfg, weights, "fp8", precision="fp16")

    assert len(ctx.scales) == LAYERS * len(PROJECTIONS)
    qkv = ctx.scales["blocks.1.attn.qkv"]
    assert qkv.input_scale == pytest.approx(0.02)
    assert len(qkv.weights) == 3
    assert all((w.in_features, w.out_features) == (HIDDEN, HIDDEN) for w in qkv.weights)
    fc2 = ctx.scales["blocks.0.mlp.fc2"].weights[0]
    assert (fc2.in_features, fc2.out_features) == (MLP, HIDDEN)

    source = np.asarray(weights["blocks.1.attn.qkv.weight"], dtype=np.float32)
    k_part = np.split(source, 3, axis=1)[1]
    packed, scale = quantize_fp8_weight(k_part)
    assert np.array_equal(qkv.weights[1].packed, packed)
    assert qkv.weights[1].weight_scale == pytest.approx(scale)
    assert ctx.should_quantize("blocks.0.attn.proj")


def test_null_entry_stays_unquantized(tiny, monkeypatch) -> None:
    """A null scale (massive activation) leaves that projection in fp16 on purpose."""
    _, vit_cfg, weights, _ = tiny
    scales = _all_scales(vit_cfg, null=("blocks.1.mlp.fc2",))
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    ctx = calibrate_timm_vit(vit_cfg, weights, "fp8", precision="fp16")
    assert not ctx.should_quantize("blocks.1.mlp.fc2")
    assert ctx.should_quantize("blocks.0.mlp.fc2")
    assert len(ctx.scales) == LAYERS * len(PROJECTIONS) - 1


def test_projection_without_a_scale_is_refused(tiny, monkeypatch) -> None:
    """A projection missing from the file fails the build and names the calibration tool."""
    _, vit_cfg, weights, _ = tiny
    scales = _all_scales(vit_cfg)
    del scales["blocks.0.mlp.fc2"]
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"blocks\.0\.mlp\.fc2.*calibrate_fp8_scales\.py"):
        calibrate_timm_vit(vit_cfg, weights, "fp8", precision="fp16")


def test_scale_for_an_unknown_projection_is_refused(tiny, monkeypatch) -> None:
    """A table with more blocks than the model is a mismatch, not a silent partial build."""
    _, vit_cfg, weights, _ = tiny
    scales = _all_scales(vit_cfg)
    scales["blocks.9.mlp.fc1"] = 0.02
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: scales)
    with pytest.raises(ValueError, match=r"blocks\.9\.mlp\.fc1"):
        calibrate_timm_vit(vit_cfg, weights, "fp8", precision="fp16")


def test_already_quantized_checkpoint_is_refused(tiny, monkeypatch) -> None:
    """A checkpoint that carries FP8 scales is refused instead of quantized twice."""
    _, vit_cfg, weights, _ = tiny
    weights["blocks.0.mlp.fc1.weight_scale"] = np.ones(1, dtype=np.float32)
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(vit_cfg))
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_timm_vit(vit_cfg, weights, "fp8", precision="fp16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_needs_fp16_precision(tiny, monkeypatch, precision) -> None:
    """FP8 is built on the fp16 engine only."""
    _, vit_cfg, weights, _ = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(vit_cfg))
    with pytest.raises(ValueError, match="precision fp16"):
        calibrate_timm_vit(vit_cfg, weights, "fp8", precision=precision)


def test_unsupported_quantization_is_refused(tiny) -> None:
    """Only fp8 is implemented."""
    _, vit_cfg, weights, _ = tiny
    with pytest.raises(NotImplementedError, match="fp8"):
        calibrate_timm_vit(vit_cfg, weights, "nvfp4", precision="fp16")


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


def test_build_refuses_fp8_with_tensor_parallelism(tiny) -> None:
    """Tensor-parallel FP8 is not implemented."""
    *_, model_dir = tiny
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        model_module.build(_request(model_dir, tensor_parallel_size=2), _Writer())


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
    """Only the four linears of each block are calibrated; the head and norms are not."""
    name = calibrate_fp8_scales.projection_name
    for stem in PROJECTIONS:
        assert name(f"blocks.11.{stem}") == f"blocks.11.{stem}"
    assert name("head") is None
    assert name("blocks.0.norm1") is None
    assert name("blocks.0.attn.qkv.input_quantizer") is None
    assert name("patch_embed.proj") is None


def test_outlier_projections_become_null() -> None:
    """A scale above 20x the median of its kind is stored as null, the others are kept."""
    scales = {f"blocks.{layer}.mlp.fc2": 0.01 for layer in range(6)}
    scales.update({f"blocks.{layer}.attn.proj": 0.02 for layer in range(6)})
    scales["blocks.3.mlp.fc2"] = 0.5
    table, excluded = calibrate_fp8_scales.split_outlier_scales(scales)
    assert excluded == ["blocks.3.mlp.fc2"]
    assert table["blocks.3.mlp.fc2"] is None
    assert table["blocks.2.mlp.fc2"] == 0.01
    assert table["blocks.3.attn.proj"] == 0.02


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
def test_dequantize_layers_output_fp16() -> None:
    """Both Dequantize layers of a projection output fp16, and no Cast is added."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    activation = network.add_input("x", trt.float16, (1, 8, 64))
    packed, scale = quantize_fp8_weight(
        np.random.default_rng(0).standard_normal((64, 32)).astype("f")
    )
    ctx = quantization.TimmVitQuantContext(
        scales={
            "m": quantization._LayerScales(0.1, (quantization._FP8Weight(packed, scale, 64, 32),))
        }
    )
    out = ctx.matmul(network, ctx.quantize_input(network, activation, "m"), "m")
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    assert trt.LayerType.CAST not in [layer.type for layer in layers]
    dequantize = [layer for layer in layers if layer.type == trt.LayerType.DEQUANTIZE]
    assert len(dequantize) == 2
    assert all(layer.get_output(0).dtype == trt.float16 for layer in dequantize)
    assert out.dtype == trt.float16


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_fp8_weights(tiny, monkeypatch) -> None:
    """The quantized engine builds from the real build path, stores FP8 weights and is smaller."""
    config, vit_cfg, weights, model_dir = tiny
    monkeypatch.setattr(quantization, "load_activation_scales", lambda *_: _all_scales(vit_cfg))
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
