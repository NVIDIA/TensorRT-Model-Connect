# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv2 FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

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
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    LAYER_PROJECTIONS,
    calibrate_dinov2,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, PATCH, GRID = 128, 512, 2, 4, 14, 4


def _tiny_config() -> dict:
    """The geometry ``config.resolve`` would return for a tiny checkpoint."""
    return {
        "hidden_size": D,
        "intermediate_size": FFN,
        "num_layers": LAYERS,
        "num_heads": HEADS,
        "head_dim": D // HEADS,
        "eps": 1e-6,
        "patch_size": PATCH,
        "trained_grid": GRID,
        "grid": GRID,
        "image_size": GRID * PATCH,
    }


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor the engine reads, at a tiny geometry."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    weights = {
        "embeddings.cls_token": normal(1, 1, D),
        "embeddings.position_embeddings": normal(1, GRID * GRID + 1, D),
        "embeddings.patch_embeddings.projection.weight": normal(D, 3, PATCH, PATCH),
        "embeddings.patch_embeddings.projection.bias": normal(D),
        "layernorm.weight": np.ones(D, np.float32),
        "layernorm.bias": normal(D),
    }
    for layer in range(LAYERS):
        prefix = f"encoder.layer.{layer}"
        for norm in ("norm1", "norm2"):
            weights[f"{prefix}.{norm}.weight"] = np.ones(D, np.float32)
            weights[f"{prefix}.{norm}.bias"] = normal(D)
        for scale in ("layer_scale1", "layer_scale2"):
            weights[f"{prefix}.{scale}.lambda1"] = normal(D)
        for name in ("query", "key", "value"):
            weights[f"{prefix}.attention.attention.{name}.weight"] = normal(D, D)
            weights[f"{prefix}.attention.attention.{name}.bias"] = normal(D)
        weights[f"{prefix}.attention.output.dense.weight"] = normal(D, D)
        weights[f"{prefix}.attention.output.dense.bias"] = normal(D)
        weights[f"{prefix}.mlp.fc1.weight"] = normal(FFN, D)
        weights[f"{prefix}.mlp.fc1.bias"] = normal(FFN)
        weights[f"{prefix}.mlp.fc2.weight"] = normal(D, FFN)
        weights[f"{prefix}.mlp.fc2.bias"] = normal(D)
    return weights


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float | None]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(monkeypatch):
    """Tiny weights, their geometry, and activation scales for every projection."""
    cfg = _tiny_config()
    scales = _all_scales(cfg)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
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


def test_every_projection_is_quantized(tiny) -> None:
    """Six projections per layer get an FP8 weight of the checkpoint's shape."""
    weights, cfg, _ = tiny
    ctx = calibrate_dinov2(weights, cfg, "fp8")
    assert len(ctx.scales) == LAYERS * len(LAYER_PROJECTIONS) == len(projections(cfg))
    entry = ctx.scales["encoder.layer.1.mlp.fc1"]
    assert entry.weight.packed.shape == (FFN, D)
    assert (entry.weight.out_features, entry.weight.in_features) == (FFN, D)
    assert ctx.has_fp8_weights
    assert ctx.should_quantize("encoder.layer.0.attention.output.dense")
    assert not ctx.should_quantize("embeddings.patch_embeddings.projection")
    assert not ctx.should_quantize("layernorm")


def test_outlier_projection_stays_unquantized(tiny) -> None:
    """A null entry is an explicit decision: that projection is built in float32."""
    weights, cfg, scales = tiny
    scales["encoder.layer.1.mlp.fc2"] = None
    ctx = calibrate_dinov2(weights, cfg, "fp8")
    assert not ctx.should_quantize("encoder.layer.1.mlp.fc2")
    assert ctx.matmul("encoder.layer.1.mlp.fc2") is None
    assert ctx.should_quantize("encoder.layer.0.mlp.fc2")


def test_projection_without_an_entry_is_an_error(tiny) -> None:
    """A projection missing from the table raises; there is no silent float32 fallback."""
    weights, cfg, scales = tiny
    del scales["encoder.layer.0.attention.attention.key"]
    with pytest.raises(ValueError, match="calibrate_fp8_scales.py"):
        calibrate_dinov2(weights, cfg, "fp8")


def test_entry_without_a_projection_is_an_error(tiny) -> None:
    """A table written for another depth raises."""
    weights, cfg, scales = tiny
    scales["encoder.layer.7.mlp.fc1"] = 0.05
    with pytest.raises(ValueError, match="without a projection"):
        calibrate_dinov2(weights, cfg, "fp8")


def test_non_positive_scale_is_an_error(tiny) -> None:
    """A zero scale would divide by zero in Quantize."""
    weights, cfg, scales = tiny
    scales["encoder.layer.0.mlp.fc1"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        calibrate_dinov2(weights, cfg, "fp8")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg, _ = tiny
    weights["encoder.layer.0.mlp.fc1.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_dinov2(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg, _ = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_dinov2(weights, cfg, "nvfp4")


def test_unknown_model_config_names_the_calibration_tool() -> None:
    """A model config with no committed scales fails and says how to add them."""
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        calibrate_dinov2(_tiny_weights(), _tiny_config(), "fp8")


def test_scale_key_comes_from_the_resolved_config() -> None:
    """The key names width, depth, MLP width, heads, patch and image size."""
    assert scale_key(_tiny_config()) == (
        f"dinov2-h{D}-l{LAYERS}-i{FFN}-n{HEADS}-p{PATCH}-s{PATCH * GRID}"
    )


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has an entry per projection; outliers are null."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"dinov2-h(\d+)-l(\d+)-i(\d+)-n(\d+)-p(\d+)-s(\d+)", key)
        assert match, key
        layers = int(match.group(2))
        assert len(scales) == layers * len(LAYER_PROJECTIONS), key
        for layer in range(layers):
            for stem in LAYER_PROJECTIONS:
                assert f"encoder.layer.{layer}.{stem}" in scales, (key, layer, stem)
        values = [v for v in scales.values() if v is not None]
        assert values, key
        assert all(np.isfinite(v) and v > 0 for v in values), key
        assert len(values) >= 0.9 * len(scales), f"{key}: too many projections left float32"


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only encoder projections are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("encoder.layer.3.mlp.fc1") == "encoder.layer.3.mlp.fc1"
    assert name("encoder.layer.0.attention.attention.query") is not None
    assert name("encoder.layer.0.attention.output.dense") is not None
    assert name("encoder.layer.0.attention.attention") is None
    assert name("embeddings.patch_embeddings.projection") is None
    assert name("layernorm") is None


def test_calibration_tool_marks_outliers_against_the_median_of_their_kind() -> None:
    """A scale above 20x the median of the same projection in other layers becomes None."""
    scales = {name: 0.01 for name in projections({**_tiny_config(), "num_layers": 5})}
    scales["encoder.layer.4.mlp.fc2"] = 0.5
    scales["encoder.layer.4.mlp.fc1"] = 0.15
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert marked["encoder.layer.4.mlp.fc2"] is None
    assert marked["encoder.layer.4.mlp.fc1"] == 0.15
    assert marked["encoder.layer.0.mlp.fc2"] == 0.01
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

    def run(self, torch, value, output: str):
        out = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda")
        self.context.set_tensor_address("pixel_values", value.data_ptr())
        self.context.set_tensor_address(output, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(-1)


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_engine_builds_and_matches_float32(tiny) -> None:
    """The FP8 engine builds, is smaller, and tracks the float32 engine."""
    torch = pytest.importorskip("torch")
    weights, cfg, _ = tiny
    ctx = calibrate_dinov2(weights, cfg, "fp8")
    size = cfg["image_size"]
    pixels = torch.randn(1, 3, size, size, device="cuda")

    plain = model.build_engine(weights, cfg)
    quant = model.build_engine(weights, cfg, quant_ctx=ctx)
    assert len(quant) < len(plain)

    reference = _Engine(plain).run(torch, pixels, "last_hidden_state")
    actual = _Engine(quant).run(torch, pixels, "last_hidden_state")
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
    assert cosine > 0.9, cosine


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names(tiny) -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    weights, cfg, _ = tiny
    ctx = calibrate_dinov2(weights, cfg, "fp8")
    _, _, config = model._builder(False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = model._builder(False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED


def _request(tmp_path: Path, **overrides) -> BuildRequest:
    fields = {
        "model_dir": tmp_path,
        "output_path": tmp_path / "b",
        "family": "dinov2",
        "task": "image_to_token_and_pooled_features",
        "precision": "fp32",
        "quantization": "fp8",
    }
    fields.update(overrides)
    return BuildRequest(**fields)


def test_build_requires_float32_for_fp8(tmp_path: Path) -> None:
    """fp8 keeps the family's float32 rule: the engine still works in float32 around Q/DQ."""
    with pytest.raises(NotImplementedError, match="fp32 only"):
        model.build(_request(tmp_path, precision="fp16"), None)


def test_build_refuses_other_quantization_formats(tmp_path: Path) -> None:
    """Only fp8 is accepted."""
    with pytest.raises(NotImplementedError, match="supports quantization fp8"):
        model.build(_request(tmp_path, quantization="nvfp4"), None)
