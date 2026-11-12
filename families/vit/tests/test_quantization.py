# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViT FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, config as config_module, model  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    calibrate_vit,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, PATCH, GRID, CLASSES = 128, 256, 2, 4, 16, 4, 10
PREFIX = "vit."


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor the family reads, at a tiny geometry."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def norm(prefix):
        return {f"{prefix}.weight": np.ones(D, np.float32), f"{prefix}.bias": normal(D)}

    def linear(prefix, out, inp):
        return {f"{prefix}.weight": normal(out, inp), f"{prefix}.bias": normal(out)}

    weights = {
        f"{PREFIX}embeddings.cls_token": normal(1, 1, D),
        f"{PREFIX}embeddings.position_embeddings": normal(1, GRID * GRID + 1, D),
        f"{PREFIX}embeddings.patch_embeddings.projection.weight": normal(D, 3, PATCH, PATCH),
        f"{PREFIX}embeddings.patch_embeddings.projection.bias": normal(D),
    }
    weights.update(norm(f"{PREFIX}layernorm"))
    weights.update(linear("classifier", CLASSES, D))
    for layer in range(LAYERS):
        prefix = f"{PREFIX}encoder.layer.{layer}"
        weights.update(norm(f"{prefix}.layernorm_before"))
        weights.update(norm(f"{prefix}.layernorm_after"))
        for name in ("query", "key", "value"):
            weights.update(linear(f"{prefix}.attention.attention.{name}", D, D))
        weights.update(linear(f"{prefix}.attention.output.dense", D, D))
        weights.update(linear(f"{prefix}.intermediate.dense", FFN, D))
        weights.update(linear(f"{prefix}.output.dense", D, FFN))
    return weights


def _config(tmp_path: Path, weights: dict) -> dict:
    """Resolve the tiny geometry the way ``build`` does."""
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["ViTForImageClassification"], "num_attention_heads": HEADS})
    )
    return config_module.resolve(tmp_path, weights)


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny weights, their resolved geometry, and activation scales for every projection."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    return weights, cfg


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
    weights, cfg = tiny
    ctx = calibrate_vit(weights, cfg, "fp8")
    assert len(ctx.scales) == LAYERS * 6 == len(projections(cfg))
    entry = ctx.scales[f"{PREFIX}encoder.layer.1.intermediate.dense"]
    assert entry.weight.packed.shape == (FFN, D)
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("classifier")
    assert not ctx.should_quantize(f"{PREFIX}embeddings.patch_embeddings.projection")


def test_split_projections_share_the_input_scale_but_not_the_weight_scale(tiny) -> None:
    """Query, key and value read one input; each keeps its own weight scale."""
    weights, cfg = tiny
    ctx = calibrate_vit(weights, cfg, "fp8")
    entries = [
        ctx.scales[f"{PREFIX}encoder.layer.0.attention.attention.{name}"]
        for name in ("query", "key", "value")
    ]
    assert len({e.input_scale for e in entries}) == 1
    for name, entry in zip(("query", "key", "value"), entries):
        weight = weights[f"{PREFIX}encoder.layer.0.attention.attention.{name}.weight"]
        assert entry.weight.weight_scale == pytest.approx(float(np.abs(weight).max()) / 448.0)


def test_null_scale_keeps_the_projection_unquantized(tmp_path: Path, monkeypatch) -> None:
    """An outlier projection is stored as null and stays in the work precision."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    outlier = f"{PREFIX}encoder.layer.0.output.dense"
    scales[outlier] = None
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    ctx = calibrate_vit(weights, cfg, "fp8")
    assert not ctx.should_quantize(outlier)
    assert ctx.matmul(outlier) is None
    assert ctx.should_quantize(f"{PREFIX}encoder.layer.0.intermediate.dense")


def test_missing_scale_raises_and_names_the_tool(tmp_path: Path, monkeypatch) -> None:
    """A table without an entry for a projection is an error, not a silent fallback."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    del scales[f"{PREFIX}encoder.layer.0.attention.attention.key"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="calibrate_fp8_scales"):
        calibrate_vit(weights, cfg, "fp8")


def test_scale_without_projection_raises(tmp_path: Path, monkeypatch) -> None:
    """A table entry that matches no projection means the table is for another model."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    scales[f"{PREFIX}encoder.layer.9.output.dense"] = 0.05
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="without a projection"):
        calibrate_vit(weights, cfg, "fp8")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg = tiny
    weights[f"{PREFIX}encoder.layer.0.intermediate.dense.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_vit(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_vit(weights, cfg, "nvfp4")


def test_unknown_model_config_names_the_calibration_tool(tmp_path: Path) -> None:
    """A model config with no committed scales fails with the available configs and the tool."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales") as error:
        calibrate_vit(weights, cfg, "fp8")
    assert "no calibrated FP8 activation scales" in str(error.value)


def test_scale_key_comes_from_the_resolved_config(tmp_path: Path) -> None:
    """The key names the hidden size, layers, MLP size, patch and input size."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    assert scale_key(cfg, weights).startswith(
        f"vit-h{D}-l{LAYERS}-i{FFN}-p{PATCH}-s{PATCH * GRID}-w"
    )


def test_scale_key_separates_checkpoints_of_one_model_config(tmp_path: Path) -> None:
    """A fine-tune shares the model config with its base model, but not the scales."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    other = _tiny_weights(seed=1)
    assert scale_key(cfg, weights) == scale_key(cfg, _tiny_weights())
    assert scale_key(cfg, weights) != scale_key(cfg, other)


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has an entry per projection; non-null scales are positive."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"vit-h(\d+)-l(\d+)-i(\d+)-p(\d+)-s(\d+)-w\S+-\S+", key)
        assert match, key
        layers = int(match.group(2))
        assert len(scales) == layers * len(quantization_module.LAYER_PROJECTIONS), key
        for layer in range(layers):
            for stem in quantization_module.LAYER_PROJECTIONS:
                assert f"{PREFIX}encoder.layer.{layer}.{stem}" in scales, (key, layer, stem)
        values = [v for v in scales.values() if v is not None]
        assert all(np.isfinite(v) and v > 0 for v in values), key
        # Outliers are the exception: at most a few projections stay unquantized.
        assert len(values) >= 0.9 * len(scales), key


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only the six projections of the encoder layers are calibrated, under either HF naming."""
    name = calibrate_fp8_scales.projection_name
    # transformers 5.x module paths
    assert name("vit.layers.3.mlp.fc2") == "encoder.layer.3.output.dense"
    assert name("vit.layers.0.mlp.fc1") == "encoder.layer.0.intermediate.dense"
    assert name("vit.layers.0.attention.q_proj") == "encoder.layer.0.attention.attention.query"
    assert name("vit.layers.0.attention.k_proj") == "encoder.layer.0.attention.attention.key"
    assert name("vit.layers.0.attention.v_proj") == "encoder.layer.0.attention.attention.value"
    assert name("vit.layers.11.attention.o_proj") == "encoder.layer.11.attention.output.dense"
    # transformers 4.x module paths are the checkpoint names
    assert name("vit.encoder.layer.3.output.dense") == "encoder.layer.3.output.dense"
    assert name("vit.encoder.layer.0.attention.attention.query") is not None
    for other in (
        "classifier",
        "vit.embeddings.patch_embeddings.projection",
        "vit.layers.0.attention",
        "vit.layers.0.layernorm_before",
        "vit.layernorm",
        "vit.pooler.dense",
    ):
        assert name(other) is None, other


def test_outliers_are_more_than_twenty_times_the_median_of_their_kind() -> None:
    """The outlier rule compares a projection with the same projection of the other layers."""
    scales = {}
    for layer in range(5):
        for stem in quantization_module.LAYER_PROJECTIONS:
            scales[f"{PREFIX}encoder.layer.{layer}.{stem}"] = 0.01
    scales[f"{PREFIX}encoder.layer.2.output.dense"] = 0.21
    scales[f"{PREFIX}encoder.layer.3.intermediate.dense"] = 0.19
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert [name for name, value in marked.items() if value is None] == [
        f"{PREFIX}encoder.layer.2.output.dense"
    ]


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
    assert calls == [(tmp_path / "images", 10), (tmp_path, tmp_path, 500)]


class _Engine:
    """A deserialized plan that can be run on CUDA tensors."""

    def __init__(self, plan: bytes) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def run(self, torch, name: str, value, output: str):
        out = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda")
        self.context.set_tensor_address(name, value.data_ptr())
        self.context.set_tensor_address(output, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(-1)


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_engine_builds_and_matches_float32(tiny) -> None:
    """The FP8 engine on the fp16 work type builds, is smaller, and tracks the float32 engine."""
    torch = pytest.importorskip("torch")
    weights, cfg = tiny
    ctx = calibrate_vit(weights, cfg, "fp8")
    size = cfg["image_size"]
    pixels = torch.randn(1, 3, size, size, device="cuda")

    plain = model.build_engine(weights, cfg)
    quant = model.build_engine(weights, cfg, quant_ctx=ctx, dtype=np.float16)
    assert len(quant) < len(plain)

    reference = _Engine(plain).run(torch, "pixel_values", pixels, "logits")
    actual = _Engine(quant).run(torch, "pixel_values", pixels, "logits")
    cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
    assert cosine > 0.9, cosine


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names(tiny) -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    ctx = quantization_module.VitQuantContext(scales={"x": object()})
    _, _, config = model._builder(False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = model._builder(False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED


def _fp8_request(tmp_path: Path, precision: str):
    from tensorrt_model_connect import BuildRequest

    return BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b",
        family="vit",
        task="image_to_class_scores",
        precision=precision,
        quantization="fp8",
    )


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fp8_requires_fp16(tmp_path: Path, precision: str) -> None:
    """fp8 runs on the fp16 graph only; fp32 and bf16 raise before the checkpoint is read."""
    with pytest.raises(ValueError, match="requires precision=fp16"):
        model.build(_fp8_request(tmp_path, precision), None)


def test_fp8_with_fp16_reaches_the_checkpoint(tmp_path: Path) -> None:
    """fp16 with fp8 passes the precision check and goes on to read the checkpoint."""
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="safetensors"):
        model.build(_fp8_request(tmp_path, "fp16"), None)


@pytest.mark.parametrize("marker", ["config", "hf_quant_config"])
def test_fp8_refuses_an_already_quantized_checkpoint(tmp_path: Path, marker: str) -> None:
    """Self-quantization needs the original checkpoint, not a quantized export."""
    if marker == "config":
        (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"x": 1}}))
    else:
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "hf_quant_config.json").write_text("{}")
    with pytest.raises(ValueError, match="already quantized"):
        model.build(_fp8_request(tmp_path, "fp16"), None)
