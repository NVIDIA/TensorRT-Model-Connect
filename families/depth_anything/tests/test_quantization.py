# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth Anything FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect import BuildRequest  # noqa: E402

from .. import calibrate_fp8_scales, model  # noqa: E402
from .. import config as config_module  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    LAYER_PROJECTIONS,
    calibrate_depth_anything,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, PATCH, GRID = 128, 512, 2, 4, 14, 4
NECK = (16, 32, 64, 128)
FUSION, HEAD = 32, 16


def _tiny_config() -> dict:
    """The geometry ``config.resolve`` returns for a tiny checkpoint."""
    raw = {
        "model_type": "depth_anything",
        "backbone_config": {
            "hidden_size": D,
            "num_attention_heads": HEADS,
            "image_size": GRID * PATCH,
            "patch_size": PATCH,
            "out_indices": [1, 1, 2, 2],
        },
        "reassemble_factors": [4, 2, 1, 0.5],
        "neck_hidden_sizes": list(NECK),
        "fusion_hidden_size": FUSION,
        "head_hidden_size": HEAD,
        "reassemble_hidden_size": D,
    }
    cfg = config_module.resolve(raw)
    cfg["num_hidden_layers"] = LAYERS
    return cfg


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor the engine reads, at a tiny model config."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    weights = {
        "backbone.embeddings.cls_token": normal(1, 1, D),
        "backbone.embeddings.position_embeddings": normal(1, GRID * GRID + 1, D),
        "backbone.embeddings.patch_embeddings.projection.weight": normal(D, 3, PATCH, PATCH),
        "backbone.embeddings.patch_embeddings.projection.bias": normal(D),
        "backbone.layernorm.weight": np.ones(D, np.float32),
        "backbone.layernorm.bias": normal(D),
    }
    for layer in range(LAYERS):
        prefix = f"backbone.encoder.layer.{layer}"
        for norm in ("norm1", "norm2"):
            weights[f"{prefix}.{norm}.weight"] = np.ones(D, np.float32)
            weights[f"{prefix}.{norm}.bias"] = normal(D)
        for scale in ("layer_scale1", "layer_scale2"):
            weights[f"{prefix}.{scale}.lambda1"] = np.full(D, 0.5, np.float32)
        for name in ("query", "key", "value"):
            weights[f"{prefix}.attention.attention.{name}.weight"] = normal(D, D)
            weights[f"{prefix}.attention.attention.{name}.bias"] = normal(D)
        weights[f"{prefix}.attention.output.dense.weight"] = normal(D, D)
        weights[f"{prefix}.attention.output.dense.bias"] = normal(D)
        weights[f"{prefix}.mlp.fc1.weight"] = normal(FFN, D)
        weights[f"{prefix}.mlp.fc1.bias"] = normal(FFN)
        weights[f"{prefix}.mlp.fc2.weight"] = normal(D, FFN)
        weights[f"{prefix}.mlp.fc2.bias"] = normal(D)
    for index, width in enumerate(NECK):
        prefix = f"neck.reassemble_stage.layers.{index}"
        weights[f"{prefix}.projection.weight"] = normal(width, D, 1, 1)
        weights[f"{prefix}.projection.bias"] = normal(width)
        weights[f"neck.convs.{index}.weight"] = normal(FUSION, width, 3, 3)
    weights["neck.reassemble_stage.layers.0.resize.weight"] = normal(NECK[0], NECK[0], 4, 4)
    weights["neck.reassemble_stage.layers.0.resize.bias"] = normal(NECK[0])
    weights["neck.reassemble_stage.layers.1.resize.weight"] = normal(NECK[1], NECK[1], 2, 2)
    weights["neck.reassemble_stage.layers.1.resize.bias"] = normal(NECK[1])
    weights["neck.reassemble_stage.layers.3.resize.weight"] = normal(NECK[3], NECK[3], 3, 3)
    weights["neck.reassemble_stage.layers.3.resize.bias"] = normal(NECK[3])
    for index in range(4):
        prefix = f"neck.fusion_stage.layers.{index}"
        for unit in ("residual_layer1", "residual_layer2"):
            for conv in ("convolution1", "convolution2"):
                weights[f"{prefix}.{unit}.{conv}.weight"] = normal(FUSION, FUSION, 3, 3)
                weights[f"{prefix}.{unit}.{conv}.bias"] = normal(FUSION)
        weights[f"{prefix}.projection.weight"] = normal(FUSION, FUSION, 1, 1)
        weights[f"{prefix}.projection.bias"] = normal(FUSION)
    weights["head.conv1.weight"] = normal(FUSION // 2, FUSION, 3, 3)
    weights["head.conv1.bias"] = normal(FUSION // 2)
    weights["head.conv2.weight"] = normal(HEAD, FUSION // 2, 3, 3)
    weights["head.conv2.bias"] = normal(HEAD)
    weights["head.conv3.weight"] = normal(1, HEAD, 1, 1)
    weights["head.conv3.bias"] = np.full(1, 1.0, np.float32)
    return weights


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float | None]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights, their model config, and activation scales for every projection."""
    cfg = _tiny_config()
    scales = _all_scales(cfg)
    monkeypatch.setattr(
        quantization_module, "load_activation_scales", lambda _cfg, _weights: scales
    )
    return _tiny_weights(), cfg, scales


def test_fp8_weight_scale_is_amax_over_448() -> None:
    """The scale is max|w| / 448 and the stored bytes round-trip to the weight."""
    weight = np.random.default_rng(1).normal(0.0, 0.1, (16, 32)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448.0)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    assert np.abs(restored - weight).max() <= 0.07 * np.abs(weight).max()


def test_fp8_weight_rejects_all_zero() -> None:
    """A weight with no positive amax has no scale."""
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 4), np.float32))


def test_every_backbone_projection_is_quantized(tiny) -> None:
    """Six projections per layer get an FP8 weight of the checkpoint's shape."""
    weights, cfg, _ = tiny
    ctx = calibrate_depth_anything(weights, cfg, "fp8")
    assert len(ctx.scales) == LAYERS * len(LAYER_PROJECTIONS) == len(projections(cfg))
    entry = ctx.scales["backbone.encoder.layer.1.mlp.fc1"]
    assert entry.weight.packed.shape == (FFN, D)
    assert (entry.weight.out_features, entry.weight.in_features) == (FFN, D)
    assert ctx.has_fp8_weights
    assert ctx.should_quantize("backbone.encoder.layer.0.attention.output.dense")
    assert not ctx.should_quantize("backbone.embeddings.patch_embeddings.projection")
    assert not ctx.should_quantize("neck.convs.0")
    assert not ctx.should_quantize("head.conv1")


def test_outlier_projection_stays_unquantized(tiny) -> None:
    """A null entry is an explicit decision: that projection is built in the work precision."""
    weights, cfg, scales = tiny
    scales["backbone.encoder.layer.1.mlp.fc2"] = None
    ctx = calibrate_depth_anything(weights, cfg, "fp8")
    assert not ctx.should_quantize("backbone.encoder.layer.1.mlp.fc2")
    assert ctx.matmul("backbone.encoder.layer.1.mlp.fc2") is None
    assert ctx.should_quantize("backbone.encoder.layer.0.mlp.fc2")


def test_projection_without_an_entry_is_an_error(tiny) -> None:
    """A projection missing from the table raises; there is no silent fallback."""
    weights, cfg, scales = tiny
    del scales["backbone.encoder.layer.0.attention.attention.key"]
    with pytest.raises(ValueError, match="calibrate_fp8_scales.py"):
        calibrate_depth_anything(weights, cfg, "fp8")


def test_entry_without_a_projection_is_an_error(tiny) -> None:
    """A table written for another depth raises."""
    weights, cfg, scales = tiny
    scales["backbone.encoder.layer.7.mlp.fc1"] = 0.05
    with pytest.raises(ValueError, match="without a projection"):
        calibrate_depth_anything(weights, cfg, "fp8")


def test_non_positive_scale_is_an_error(tiny) -> None:
    """A zero scale would divide by zero in Quantize."""
    weights, cfg, scales = tiny
    scales["backbone.encoder.layer.0.mlp.fc1"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        calibrate_depth_anything(weights, cfg, "fp8")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg, _ = tiny
    weights["backbone.encoder.layer.0.mlp.fc1.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_depth_anything(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_depth_anything(weights, cfg, "nvfp4")


def test_unknown_model_config_names_the_calibration_tool() -> None:
    """A model config with no committed scales fails and says how to add them."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        calibrate_depth_anything(_tiny_weights(), _tiny_config(), "fp8")


def test_scale_key_comes_from_the_resolved_config() -> None:
    """The key names width, depth, heads, patch, image size and a weight fingerprint."""
    weights = _tiny_weights()
    total = float(weights["backbone.encoder.layer.0.attention.attention.query.weight"].sum())
    assert scale_key(_tiny_config(), weights) == (
        f"depth_anything-h{D}-l{LAYERS}-n{HEADS}-p{PATCH}-s{PATCH * GRID}-w{total:.4e}"
    )


def test_scale_key_tells_checkpoints_of_one_model_config_apart() -> None:
    """Two checkpoints with the same model config (for example V1 and V2) get different keys."""
    cfg = _tiny_config()
    assert scale_key(cfg, _tiny_weights(0)) != scale_key(cfg, _tiny_weights(1))
    assert scale_key(cfg, _tiny_weights(0)) == scale_key(cfg, _tiny_weights(0))


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has an entry per projection; outliers are null."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(
            r"depth_anything-h(\d+)-l(\d+)-n(\d+)-p(\d+)-s(\d+)-w-?\d\.\d{4}e[+-]\d+", key
        )
        assert match, key
        layers = int(match.group(2))
        assert len(scales) == layers * len(LAYER_PROJECTIONS), key
        for layer in range(layers):
            for stem in LAYER_PROJECTIONS:
                assert f"backbone.encoder.layer.{layer}.{stem}" in scales, (key, layer, stem)
        values = [v for v in scales.values() if v is not None]
        assert values, key
        assert all(np.isfinite(v) and v > 0 for v in values), key
        assert len(values) >= 0.9 * len(scales), f"{key}: too many projections left unquantized"


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only backbone encoder projections are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("backbone.encoder.layer.3.mlp.fc1") == "backbone.encoder.layer.3.mlp.fc1"
    assert name("backbone.encoder.layer.0.attention.attention.query") is not None
    assert name("backbone.encoder.layer.0.attention.output.dense") is not None
    assert name("backbone.encoder.layer.0.attention.attention") is None
    assert name("backbone.embeddings.patch_embeddings.projection") is None
    assert name("neck.convs.0") is None
    assert name("head.conv1") is None


def test_calibration_tool_marks_outliers_against_the_median_of_their_kind() -> None:
    """A scale above 20x the median of the same projection in other layers becomes None."""
    cfg = {**_tiny_config(), "num_hidden_layers": 5}
    scales = {name: 0.01 for name in projections(cfg)}
    scales["backbone.encoder.layer.4.mlp.fc2"] = 0.5
    scales["backbone.encoder.layer.4.mlp.fc1"] = 0.15
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert marked["backbone.encoder.layer.4.mlp.fc2"] is None
    assert marked["backbone.encoder.layer.4.mlp.fc1"] == 0.15
    assert marked["backbone.encoder.layer.0.mlp.fc2"] == 0.01
    assert sum(v is None for v in marked.values()) == 1


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The fetch and calibrate commands reach their functions with the parsed arguments."""
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_images", lambda *args: calls.append(args))
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: calls.append(args))
    monkeypatch.setattr(
        "sys.argv", ["calibrate", "fetch", "--out", str(tmp_path / "images"), "--count", "10"]
    )
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "calibrate", "--model-dir", str(tmp_path), "--images", str(tmp_path)],
    )
    calibrate_fp8_scales.main()
    assert calls == [(tmp_path / "images", 10), (tmp_path, tmp_path, 200)]


class _Engine:
    """A deserialized plan that can be run on CUDA tensors."""

    def __init__(self, plan: bytes) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def run(self, torch, value):
        out = torch.empty(tuple(self.engine.get_tensor_shape("predicted_depth")), device="cuda")
        self.context.set_tensor_address("pixel_values", value.data_ptr())
        self.context.set_tensor_address("predicted_depth", out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(-1)


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_quantized_engine_builds_and_matches_unquantized(tiny, precision) -> None:
    """The FP8 engine builds, is smaller, and tracks the unquantized engine."""
    torch = pytest.importorskip("torch")
    weights, cfg, _ = tiny
    ctx = calibrate_depth_anything(weights, cfg, "fp8")
    size = cfg["image_size"]
    pixels = torch.randn(1, 3, size, size, device="cuda")

    plain = model.build_engine(weights, cfg, precision=precision)
    quant = model.build_engine(weights, cfg, precision=precision, quant_ctx=ctx)
    assert len(quant) < len(plain)

    reference = _Engine(plain).run(torch, pixels)
    actual = _Engine(quant).run(torch, pixels)
    assert torch.isfinite(actual).all()
    error = (actual - reference).norm() / reference.norm()
    assert error < 0.2, error.item()


def test_fp8_build_options_keep_layer_names(tiny) -> None:
    """FP8 builds ask for detailed profiling and drop the dual-GEMM route when it is listed."""
    weights, cfg, _ = tiny
    ctx = calibrate_depth_anything(weights, cfg, "fp8")

    class _Config:
        profiling_verbosity = None
        build_route = None
        all_build_routes = "peep:match_dual_gemm"

    config = _Config()
    model._apply_quant_options(config, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    assert config.build_route == "-peep:match_dual_gemm=off"

    plain = _Config()
    model._apply_quant_options(plain, None)
    assert plain.profiling_verbosity is None and plain.build_route is None


def _request(tmp_path: Path, **overrides) -> BuildRequest:
    fields = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "b",
        "family": "depth_anything",
        "task": "monocular_depth",
        "precision": "fp16",
        "quantization": "fp8",
    }
    fields.update(overrides)
    return BuildRequest(**fields)


def test_build_refuses_other_quantization_formats(tmp_path: Path) -> None:
    """Only fp8 is accepted."""
    with pytest.raises(NotImplementedError, match="supports quantization fp8"):
        model.build(_request(tmp_path, quantization="nvfp4"), None)
