# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth Anything FP8 and NVFP4 quantization: weight scaling, projection coverage, calibration
tool, engine build."""

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
    NVFP4_ACTIVATION_SCALES_FILENAME,
    NVFP4_PROJECTIONS,
    _FP4Weight,
    _FP8Weight,
    calibrate_depth_anything,
    dequantize_nvfp4_weight,
    nvfp4_projections,
    projections,
    quantize_fp8_weight,
    quantize_nvfp4_weight,
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


@pytest.fixture
def tiny_nvfp4(tiny, monkeypatch):
    """``tiny`` plus NVFP4 input scales for every fc1 (the FP8 scale / 6).

    The tiny model config is added to the NVFP4 model configs for these tests only.
    """
    weights, cfg, scales = tiny
    monkeypatch.setitem(quantization_module.NVFP4_MODEL_CONFIGS, D, "Tiny")
    fp4_scales = {name: 0.05 / 6 for name in nvfp4_projections(cfg)}
    monkeypatch.setattr(
        quantization_module, "load_nvfp4_activation_scales", lambda _cfg, _weights: fp4_scales
    )
    return weights, cfg, scales, fp4_scales


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
    """Only fp8 and nvfp4 are implemented."""
    weights, cfg, _ = tiny
    with pytest.raises(NotImplementedError, match="'fp8', 'nvfp4'"):
        calibrate_depth_anything(weights, cfg, "int4_awq")


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


@pytest.mark.parametrize("quantization", ["int4_awq", "w4a16", "fp8_block"])
def test_build_refuses_other_quantization_formats(tmp_path: Path, quantization: str) -> None:
    """Only fp8, nvfp4 and none are accepted."""
    with pytest.raises(NotImplementedError, match="supports quantization fp8, nvfp4 or none"):
        model.build(_request(tmp_path, quantization=quantization), None)


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_build_refuses_other_precisions_for_quantized_builds(tmp_path: Path, quantization) -> None:
    """fp8 and nvfp4 take the same work precisions as the unquantized build (fp16, fp32)."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "depth_anything", "backbone_config": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unsupported depth_anything precision: bf16"):
        model.build(_request(tmp_path, quantization=quantization, precision="bf16"), None)


# ---------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------

_E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _reference_nvfp4(weight: np.ndarray):
    """Independent NVFP4 reference: nearest E2M1 value, ties to the even code."""
    weight = weight.astype(np.float32)
    rows, cols = weight.shape
    ws2 = np.float32(np.abs(weight).max()) * np.float32(1.0 / 2688)
    blocks = weight.reshape(rows, cols // 16, 16)
    scale = np.minimum(np.abs(blocks).max(-1) / np.float32(6) / ws2, np.float32(448))
    scale = scale.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1), scale).astype(np.float32)
    scaled = blocks / (scale[..., None] * ws2)
    distance = np.abs(np.abs(scaled)[..., None] - _E2M1)
    nearest = distance.argmin(-1)
    upper = np.minimum(nearest + 1, 7)
    tie = np.take_along_axis(distance, upper[..., None], -1)[..., 0] == distance.min(-1)
    index = np.where(tie & (upper != nearest) & (upper % 2 == 0), upper, nearest)
    code = (index | ((scaled < 0) << 3)).astype(np.uint8).reshape(rows, cols)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.astype(ml_dtypes.float8_e4m3fn).view(np.uint8), float(ws2)


def test_nvfp4_weight_matches_an_independent_reference() -> None:
    """Packed values, block scales and weight_scale_2 match a direct nearest-value reference."""
    rng = np.random.default_rng(3)
    weight = rng.normal(0.0, 0.05, (48, 64)).astype(np.float32)
    weight[5, 16:32] = 0.0  # an all-zero block gets block scale 1
    weight[7, :16] = np.linspace(-1.0, 1.0, 16, dtype=np.float32) * 3.0
    packed, block_scale, ws2 = quantize_nvfp4_weight(weight)
    ref_packed, ref_scale, ref_ws2 = _reference_nvfp4(weight)
    assert packed.shape == (48, 32) and packed.dtype == np.uint8
    assert block_scale.shape == (48, 4) and block_scale.dtype == np.uint8
    assert ws2 == ref_ws2
    np.testing.assert_array_equal(block_scale, ref_scale)
    np.testing.assert_array_equal(packed, ref_packed)
    assert block_scale.view(ml_dtypes.float8_e4m3fn)[5, 1].astype(np.float32) == 1.0


def test_nvfp4_weight_scale_2_is_amax_times_reciprocal_of_2688() -> None:
    """weight_scale_2 is amax * fp32(1 / 2688), the ModelOpt operation."""
    weight = np.random.default_rng(4).normal(0.0, 0.3, (16, 32)).astype(np.float32)
    _, _, ws2 = quantize_nvfp4_weight(weight)
    expected = np.float32(np.abs(weight).max()) * np.float32(1.0 / 2688)
    assert ws2 == float(expected)


def test_nvfp4_rounds_ties_to_the_even_code() -> None:
    """Midpoints go to the neighbour with the even code; the sign is bit 3."""
    scaled = np.array(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 0.26, 4.9, -0.25, -5.0, -6.0, 0.0, 7.0],
        np.float32,
    )
    codes = quantization_module._e2m1_codes(scaled)
    # magnitudes:  0    1.0  1.0  2.0  2.0  4.0  4.0  6.0  0.5  4.0
    expected = [0, 2, 2, 4, 4, 6, 6, 7, 1, 6, 0 | 8, 6 | 8, 7 | 8, 0, 7]
    np.testing.assert_array_equal(codes, np.array(expected, np.uint8))


def test_nvfp4_weight_round_trip_error_is_bounded() -> None:
    """Each value is within half an E2M1 step (at most 1 of the block's scale unit) of the weight."""
    weight = np.random.default_rng(5).normal(0.0, 0.05, (32, 64)).astype(np.float32)
    packed, block_scale, ws2 = quantize_nvfp4_weight(weight)
    restored = dequantize_nvfp4_weight(packed, block_scale, ws2)
    unit = np.repeat(
        block_scale.view(ml_dtypes.float8_e4m3fn).astype(np.float32), 16, axis=-1
    ) * np.float32(ws2)
    assert (np.abs(restored - weight) <= unit * 1.0 + 1e-7).all()
    assert np.linalg.norm(restored - weight) / np.linalg.norm(weight) < 0.15


@pytest.mark.parametrize(
    "weight, match",
    [
        (np.zeros((4, 32), np.float32), "amax"),
        (np.ones((4, 24), np.float32), "multiple of 16"),
        (np.ones((32,), np.float32), r"\[out, in\]"),
    ],
)
def test_nvfp4_weight_rejects_bad_input(weight, match) -> None:
    """No scale for an all-zero weight; 16-value blocks need in_features % 16 == 0."""
    with pytest.raises(ValueError, match=match):
        quantize_nvfp4_weight(weight)


def test_nvfp4_recipe_is_fc1_only() -> None:
    """The NVFP4 recipe quantizes only the first MLP projection to NVFP4."""
    assert NVFP4_PROJECTIONS == ("mlp.fc1",)
    assert set(NVFP4_PROJECTIONS) <= set(LAYER_PROJECTIONS)


def test_nvfp4_fc1_is_nvfp4_and_the_rest_is_fp8(tiny_nvfp4) -> None:
    """fc1 of every layer is NVFP4 with the NVFP4 scale; the other five stay the FP8 recipe."""
    weights, cfg, scales, fp4_scales = tiny_nvfp4
    ctx = calibrate_depth_anything(weights, cfg, "nvfp4")
    fp8 = calibrate_depth_anything(weights, cfg, "fp8")
    assert set(ctx.scales) == set(projections(cfg))
    for name, entry in ctx.scales.items():
        if name.endswith(".mlp.fc1"):
            assert isinstance(entry.weight, _FP4Weight), name
            assert entry.input_scale == fp4_scales[name]
            assert entry.weight.packed.shape == (FFN, D // 2)
            assert entry.weight.block_scale.shape == (FFN, D // 16)
            expected = quantize_nvfp4_weight(weights[f"{name}.weight"])
            np.testing.assert_array_equal(entry.weight.packed, expected[0])
            np.testing.assert_array_equal(entry.weight.block_scale, expected[1])
            assert entry.weight.weight_scale_2 == expected[2]
        else:
            assert isinstance(entry.weight, _FP8Weight), name
            assert entry.input_scale == scales[name]
            np.testing.assert_array_equal(entry.weight.packed, fp8.scales[name].weight.packed)
            assert entry.weight.weight_scale == fp8.scales[name].weight.weight_scale
    assert ctx.has_fp4_weights and ctx.has_fp8_weights
    assert not fp8.has_fp4_weights and fp8.has_fp8_weights
    assert sum(isinstance(e.weight, _FP4Weight) for e in ctx.scales.values()) == LAYERS


def test_nvfp4_null_entry_keeps_fc1_unquantized(tiny_nvfp4) -> None:
    """A null NVFP4 entry is an explicit decision: that fc1 is built in the work precision."""
    weights, cfg, _, fp4_scales = tiny_nvfp4
    fp4_scales["backbone.encoder.layer.1.mlp.fc1"] = None
    ctx = calibrate_depth_anything(weights, cfg, "nvfp4")
    assert not ctx.should_quantize("backbone.encoder.layer.1.mlp.fc1")
    assert isinstance(ctx.scales["backbone.encoder.layer.0.mlp.fc1"].weight, _FP4Weight)
    assert isinstance(ctx.scales["backbone.encoder.layer.1.mlp.fc2"].weight, _FP8Weight)


def test_nvfp4_table_without_an_fc1_entry_is_an_error(tiny_nvfp4) -> None:
    """An fc1 missing from the NVFP4 table raises and names the calibration tool."""
    weights, cfg, _, fp4_scales = tiny_nvfp4
    del fp4_scales["backbone.encoder.layer.0.mlp.fc1"]
    with pytest.raises(ValueError, match="calibrate_fp8_scales.py"):
        calibrate_depth_anything(weights, cfg, "nvfp4")


def test_nvfp4_table_with_another_projection_is_an_error(tiny_nvfp4) -> None:
    """An NVFP4 entry for a projection outside the recipe (here fc2) raises."""
    weights, cfg, _, fp4_scales = tiny_nvfp4
    fp4_scales["backbone.encoder.layer.0.mlp.fc2"] = 0.01
    with pytest.raises(ValueError, match="without an NVFP4 projection"):
        calibrate_depth_anything(weights, cfg, "nvfp4")


def test_nvfp4_non_positive_scale_is_an_error(tiny_nvfp4) -> None:
    """A zero NVFP4 global scale would divide by zero in the dynamic quantize."""
    weights, cfg, _, fp4_scales = tiny_nvfp4
    fp4_scales["backbone.encoder.layer.1.mlp.fc1"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        calibrate_depth_anything(weights, cfg, "nvfp4")


def test_nvfp4_already_quantized_checkpoint_is_rejected(tiny_nvfp4) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg, _, _ = tiny_nvfp4
    weights["backbone.encoder.layer.0.mlp.fc2.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized; nvfp4"):
        calibrate_depth_anything(weights, cfg, "nvfp4")


def test_nvfp4_unknown_model_config_names_the_calibration_tool(tiny, monkeypatch) -> None:
    """A model config with no committed NVFP4 scales fails and says how to add them."""
    weights, cfg, _ = tiny
    monkeypatch.setitem(quantization_module.NVFP4_MODEL_CONFIGS, D, "Tiny")
    with pytest.raises(NotImplementedError, match="NVFP4 activation scales"):
        calibrate_depth_anything(weights, cfg, "nvfp4")


# Backbone geometry of the V2 checkpoints: hidden size, heads, out_indices, neck widths.
_V2_CONFIGS = {
    "small": (384, 6, [3, 6, 9, 12], [48, 96, 192, 384]),
    "base": (768, 12, [3, 6, 9, 12], [96, 192, 384, 768]),
    "large": (1024, 16, [5, 12, 18, 24], [256, 512, 1024, 1024]),
}


def _write_v2_config(folder: Path, model: str) -> None:
    """A Depth Anything V2 ``config.json`` of the ``model`` config (no weights)."""
    hidden, heads, out_indices, neck = _V2_CONFIGS[model]
    raw = {
        "model_type": "depth_anything",
        "backbone_config": {
            "model_type": "dinov2",
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "image_size": 518,
            "patch_size": 14,
            "out_indices": out_indices,
        },
        "reassemble_factors": [4, 2, 1, 0.5],
        "neck_hidden_sizes": neck,
        "fusion_hidden_size": 64,
        "head_hidden_size": 32,
        "reassemble_hidden_size": hidden,
    }
    (folder / "config.json").write_text(json.dumps(raw), encoding="utf-8")


class _ReachedCheckpoint(Exception):
    """Raised in place of reading the checkpoint: the build got past every request check."""


def _stop_at_checkpoint(monkeypatch) -> None:
    def stop(_model_dir):
        raise _ReachedCheckpoint

    monkeypatch.setattr(model, "_load_tensors", stop)


def test_nvfp4_model_configs_are_base_and_large() -> None:
    """NVFP4 is shipped for Base (hidden size 768) and Large (1024) only."""
    assert quantization_module.NVFP4_MODEL_CONFIGS == {768: "Base", 1024: "Large"}


def test_nvfp4_is_rejected_for_the_small_model_config(tmp_path: Path, monkeypatch) -> None:
    """Small + nvfp4 fails before the checkpoint is read and names the allowed model configs."""
    _write_v2_config(tmp_path, "small")
    _stop_at_checkpoint(monkeypatch)
    message = (
        r"nvfp4 supports only the Base \(hidden size 768\) and Large \(hidden size 1024\) "
        r"model configs; this checkpoint has hidden size 384 \(Small\)"
    )
    with pytest.raises(NotImplementedError, match=message):
        model.build(_request(tmp_path, quantization="nvfp4"), None)
    cfg = config_module.resolve(json.loads((tmp_path / "config.json").read_text()))
    with pytest.raises(NotImplementedError, match=message):
        calibrate_depth_anything(_tiny_weights(), cfg, "nvfp4")


@pytest.mark.parametrize("model_config", ["base", "large"])
def test_nvfp4_is_accepted_for_base_and_large(tmp_path: Path, monkeypatch, model_config) -> None:
    """Base and Large + nvfp4 pass every request check and go on to read the checkpoint."""
    _write_v2_config(tmp_path, model_config)
    _stop_at_checkpoint(monkeypatch)
    cfg = config_module.resolve(json.loads((tmp_path / "config.json").read_text()))
    quantization_module.check_nvfp4_model_config(cfg)
    with pytest.raises(_ReachedCheckpoint):
        model.build(_request(tmp_path, quantization="nvfp4"), None)


@pytest.mark.parametrize("model_config", ["small", "base", "large"])
@pytest.mark.parametrize("quantization", ["none", "fp8"])
def test_none_and_fp8_are_accepted_for_every_model_config(
    tmp_path: Path, monkeypatch, model_config, quantization
) -> None:
    """The NVFP4 model-config rule does not touch the none and fp8 builds."""
    _write_v2_config(tmp_path, model_config)
    _stop_at_checkpoint(monkeypatch)
    with pytest.raises(_ReachedCheckpoint):
        model.build(_request(tmp_path, quantization=quantization), None)


def test_committed_nvfp4_scales_are_the_fp8_scales_over_6() -> None:
    """Every FP8 model config has an NVFP4 table: one entry per fc1, the FP8 scale / 6."""
    folder = Path(quantization_module.__file__).parent
    fp8 = json.loads((folder / ACTIVATION_SCALES_FILENAME).read_text())
    fp4 = json.loads((folder / NVFP4_ACTIVATION_SCALES_FILENAME).read_text())
    assert set(fp4) == set(fp8)
    for key, scales in fp4.items():
        layers = len(fp8[key]) // len(LAYER_PROJECTIONS)
        assert set(scales) == {f"backbone.encoder.layer.{i}.mlp.fc1" for i in range(layers)}, key
        for name, value in scales.items():
            assert value is not None and value > 0, (key, name)
            assert value == fp8[key][name] / 6, (key, name)
    assert fp4 == {key: calibrate_fp8_scales.nvfp4_scales(value) for key, value in fp8.items()}


def test_calibration_tool_writes_nvfp4_from_fp8(monkeypatch) -> None:
    """The nvfp4 command derives the NVFP4 table from the FP8 table; null stays null."""
    fp8 = {
        "backbone.encoder.layer.0.mlp.fc1": 0.6,
        "backbone.encoder.layer.0.mlp.fc2": 0.3,
        "backbone.encoder.layer.1.mlp.fc1": None,
    }
    assert calibrate_fp8_scales.nvfp4_scales(fp8) == {
        "backbone.encoder.layer.0.mlp.fc1": 0.6 / 6,
        "backbone.encoder.layer.1.mlp.fc1": None,
    }
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "write_nvfp4_from_fp8", lambda: calls.append("nvfp4"))
    monkeypatch.setattr("sys.argv", ["calibrate", "nvfp4"])
    calibrate_fp8_scales.main()
    assert calls == ["nvfp4"]


def test_nvfp4_build_options_match_fp8(tiny_nvfp4) -> None:
    """NVFP4 builds keep FP8 projections, so they use the FP8 build options."""
    weights, cfg, _, _ = tiny_nvfp4
    ctx = calibrate_depth_anything(weights, cfg, "nvfp4")

    class _Config:
        profiling_verbosity = None
        build_route = None
        all_build_routes = "peep:match_dual_gemm"

    config = _Config()
    model._apply_quant_options(config, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    assert config.build_route == "-peep:match_dual_gemm=off"


def _network_layer_types(weights, cfg, precision, ctx) -> dict:
    """Count of each layer type in the network the family defines (no engine build)."""
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    work_np, work_trt = (
        (np.float16, trt.float16) if precision == "fp16" else (np.float32, trt.float32)
    )
    model._build_network(network, weights, cfg, work_np, work_trt, ctx)
    counts: dict = {}
    for index in range(network.num_layers):
        kind = network.get_layer(index).type
        counts[kind] = counts.get(kind, 0) + 1
    return counts


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_nvfp4_network_has_one_dynamic_quantize_per_fc1(tiny_nvfp4, precision) -> None:
    """One dynamic quantize per layer (fc1); five static FP8 Quantize layers per layer."""
    weights, cfg, _, _ = tiny_nvfp4
    nvfp4 = _network_layer_types(
        weights, cfg, precision, calibrate_depth_anything(weights, cfg, "nvfp4")
    )
    fp8 = _network_layer_types(
        weights, cfg, precision, calibrate_depth_anything(weights, cfg, "fp8")
    )
    assert nvfp4.get(trt.LayerType.DYNAMIC_QUANTIZE, 0) == LAYERS
    assert nvfp4.get(trt.LayerType.QUANTIZE, 0) == 5 * LAYERS
    assert fp8.get(trt.LayerType.DYNAMIC_QUANTIZE, 0) == 0
    assert fp8.get(trt.LayerType.QUANTIZE, 0) == 6 * LAYERS


def _blackwell(torch) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_nvfp4_engine_builds_and_matches_unquantized(tiny_nvfp4, precision) -> None:
    """The NVFP4 engine builds on Blackwell and tracks the unquantized one.

    Plan size is not checked here: at this tiny model config the kernels dominate the plan.
    """
    torch = pytest.importorskip("torch")
    if not _blackwell(torch):
        pytest.skip("NVFP4 needs a Blackwell GPU")
    weights, cfg, _, _ = tiny_nvfp4
    size = cfg["image_size"]
    pixels = torch.randn(1, 3, size, size, device="cuda")

    plain = model.build_engine(weights, cfg, precision=precision)
    fp4 = model.build_engine(
        weights,
        cfg,
        precision=precision,
        quant_ctx=calibrate_depth_anything(weights, cfg, "nvfp4"),
    )

    reference = _Engine(plain).run(torch, pixels)
    actual = _Engine(fp4).run(torch, pixels)
    assert torch.isfinite(actual).all()
    error = (actual - reference).norm() / reference.norm()
    assert error < 0.3, error.item()
