# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

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
    calibrate_clip,
    projections,
    quantize_fp8_weight,
    scale_key,
)

D, FFN, LAYERS, HEADS, PATCH, GRID, VOCAB, WINDOW, PROJ = 128, 256, 2, 4, 16, 4, 100, 8, 64
TOWERS = ("vision_model", "text_model")


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor both towers read, at a tiny geometry."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def norm(prefix):
        return {f"{prefix}.weight": np.ones(D, np.float32), f"{prefix}.bias": normal(D)}

    def linear(prefix, out, inp):
        return {f"{prefix}.weight": normal(out, inp), f"{prefix}.bias": normal(out)}

    weights = {"logit_scale": np.asarray([2.6592], np.float32)}
    for tower in TOWERS:
        for layer in range(LAYERS):
            prefix = f"{tower}.encoder.layers.{layer}"
            weights.update(norm(f"{prefix}.layer_norm1"))
            weights.update(norm(f"{prefix}.layer_norm2"))
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                weights.update(linear(f"{prefix}.self_attn.{name}", D, D))
            weights.update(linear(f"{prefix}.mlp.fc1", FFN, D))
            weights.update(linear(f"{prefix}.mlp.fc2", D, FFN))
    weights["text_model.embeddings.token_embedding.weight"] = normal(VOCAB, D)
    weights["text_model.embeddings.position_embedding.weight"] = normal(WINDOW, D)
    weights.update(norm("text_model.final_layer_norm"))
    weights["text_projection.weight"] = normal(PROJ, D)
    weights["vision_model.embeddings.patch_embedding.weight"] = normal(D, 3, PATCH, PATCH)
    weights["vision_model.embeddings.class_embedding"] = normal(D)
    weights["vision_model.embeddings.position_embedding.weight"] = normal(GRID * GRID + 1, D)
    weights.update(norm("vision_model.pre_layrnorm"))
    weights.update(norm("vision_model.post_layernorm"))
    weights["visual_projection.weight"] = normal(PROJ, D)
    return weights


def _config(tmp_path: Path) -> dict:
    """Resolve the tiny geometry the way ``build`` does."""
    tower = {
        "hidden_size": D,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "intermediate_size": FFN,
        "hidden_act": "quick_gelu",
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "clip",
                "projection_dim": PROJ,
                "text_config": dict(tower, vocab_size=VOCAB, max_position_embeddings=WINDOW),
                "vision_config": dict(tower, image_size=PATCH * GRID, patch_size=PATCH),
            }
        )
    )
    return config_module.resolve(tmp_path)


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny weights, their resolved geometry, and activation scales for every projection."""
    weights = _tiny_weights()
    cfg = _config(tmp_path)
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


def test_every_projection_of_both_towers_is_quantized(tiny) -> None:
    """Six projections per layer in each tower get an FP8 weight of the checkpoint's shape."""
    weights, cfg = tiny
    ctx = calibrate_clip(weights, cfg, "fp8")
    assert len(ctx.scales) == 2 * LAYERS * 6 == len(projections(cfg))
    entry = ctx.scales["vision_model.encoder.layers.1.mlp.fc1"]
    assert entry.weight.packed.shape == (FFN, D)
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("text_projection")
    assert not ctx.should_quantize("visual_projection")


def test_null_scale_keeps_an_outlier_projection_unquantized(tmp_path: Path, monkeypatch) -> None:
    """A projection stored as null (an outlier) stays in the work precision."""
    weights = _tiny_weights()
    cfg = _config(tmp_path)
    scales = _all_scales(cfg)
    scales["text_model.encoder.layers.0.mlp.fc2"] = None
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    ctx = calibrate_clip(weights, cfg, "fp8")
    assert not ctx.should_quantize("text_model.encoder.layers.0.mlp.fc2")
    assert ctx.matmul("text_model.encoder.layers.0.mlp.fc2") is None
    assert ctx.should_quantize("text_model.encoder.layers.0.mlp.fc1")


def test_missing_scale_raises_and_names_the_tool(tmp_path: Path, monkeypatch) -> None:
    """A projection with no entry is an error, not a silent fallback."""
    weights = _tiny_weights()
    cfg = _config(tmp_path)
    scales = _all_scales(cfg)
    del scales["text_model.encoder.layers.0.self_attn.k_proj"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="calibrate_fp8_scales"):
        calibrate_clip(weights, cfg, "fp8")


def test_entry_without_projection_raises(tmp_path: Path, monkeypatch) -> None:
    """A table with an entry the model does not have is stale and rejected."""
    weights = _tiny_weights()
    cfg = _config(tmp_path)
    scales = _all_scales(cfg)
    scales["vision_model.encoder.layers.9.mlp.fc1"] = 0.1
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="without a projection"):
        calibrate_clip(weights, cfg, "fp8")


def test_invalid_scales_raise(tmp_path: Path, monkeypatch) -> None:
    """Non-positive or non-finite scales, and fp16-unsafe tiny scales, are rejected."""
    weights = _tiny_weights()
    cfg = _config(tmp_path)
    name = "vision_model.encoder.layers.0.mlp.fc1"
    for bad in (0.0, float("nan")):
        scales = _all_scales(cfg)
        scales[name] = bad
        monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
        with pytest.raises(ValueError, match="finite and positive"):
            calibrate_clip(weights, cfg, "fp8")
    scales = _all_scales(cfg)
    scales[name] = 1e-6
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="fp16"):
        calibrate_clip(weights, cfg, "fp8", precision="fp16")
    assert calibrate_clip(weights, cfg, "fp8", precision="fp32").has_fp8_weights


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg = tiny
    weights["vision_model.encoder.layers.0.mlp.fc1.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_clip(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_clip(weights, cfg, "nvfp4")


def test_unknown_model_config_is_reported(tmp_path: Path) -> None:
    """A config with no committed scales fails with the available keys and the tool name."""
    cfg = _config(tmp_path)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        calibrate_clip(_tiny_weights(), cfg, "fp8")


def test_scale_key_comes_from_the_resolved_geometry(tmp_path: Path) -> None:
    """The key names both towers, the patch, the image size and the prompt window."""
    cfg = _config(tmp_path)
    expected = f"clip-v{D}x{LAYERS}m{FFN}p{PATCH}i{PATCH * GRID}-t{D}x{LAYERS}m{FFN}w{WINDOW}"
    assert scale_key(cfg) == expected


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has an entry per projection, and positive finite scales."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    pattern = r"clip-v(\d+)x(\d+)m(\d+)p(\d+)i(\d+)-t(\d+)x(\d+)m(\d+)w(\d+)"
    for key, scales in table.items():
        match = re.fullmatch(pattern, key)
        assert match, key
        vision_layers, text_layers = int(match.group(2)), int(match.group(7))
        assert len(scales) == (vision_layers + text_layers) * 6, key
        for value in scales.values():
            assert value is None or (np.isfinite(value) and value > 0), key
        assert any(v is not None for v in scales.values()), key
        for tower, layers in (("vision_model", vision_layers), ("text_model", text_layers)):
            for layer in range(layers):
                for stem in quantization_module.LAYER_PROJECTIONS:
                    assert f"{tower}.encoder.layers.{layer}.{stem}" in scales, (key, tower, layer)


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only encoder projections of the two towers are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name("vision_model.encoder.layers.3.mlp.fc1") == "vision_model.encoder.layers.3.mlp.fc1"
    assert name("text_model.encoder.layers.0.self_attn.out_proj") is not None
    assert name("visual_projection") is None
    assert name("text_projection") is None
    assert name("vision_model.embeddings.patch_embedding") is None


def test_calibration_tool_marks_outliers_per_kind() -> None:
    """A scale above 20x the median of its kind in its tower is stored as null."""
    scales = {}
    for tower in TOWERS:
        for layer in range(5):
            for stem in quantization_module.LAYER_PROJECTIONS:
                scales[f"{tower}.encoder.layers.{layer}.{stem}"] = 0.1
    scales["text_model.encoder.layers.2.mlp.fc2"] = 2.5
    scales["vision_model.encoder.layers.2.mlp.fc2"] = 1.9
    marked = calibrate_fp8_scales.mark_outliers(scales)
    assert marked["text_model.encoder.layers.2.mlp.fc2"] is None
    assert marked["vision_model.encoder.layers.2.mlp.fc2"] == 1.9
    assert sum(v is None for v in marked.values()) == 1


def test_calibration_tool_splits_validation_images() -> None:
    """Calibration and evaluation images are disjoint and cover every class evenly."""
    splits = [calibrate_fp8_scales.split_of(i) for i in range(50000)]
    assert splits.count("calib") == 500 and splits.count("eval") == 1000
    assert not {i for i, s in enumerate(splits) if s == "calib"} & {
        i for i, s in enumerate(splits) if s == "eval"
    }


def test_calibration_tool_cli_parses_both_commands(monkeypatch, tmp_path: Path) -> None:
    """The fetch and calibrate commands reach their functions with the parsed arguments."""
    calls = []
    monkeypatch.setattr(calibrate_fp8_scales, "fetch_images", lambda *args: calls.append(args))
    monkeypatch.setattr(calibrate_fp8_scales, "calibrate", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["calibrate", "fetch", "--out", str(tmp_path / "images")])
    calibrate_fp8_scales.main()
    monkeypatch.setattr(
        "sys.argv",
        ["calibrate", "calibrate", "--model-dir", str(tmp_path), "--images", str(tmp_path)],
    )
    calibrate_fp8_scales.main()
    assert calls == [(tmp_path / "images",), (tmp_path, tmp_path, 500)]


class _Engine:
    """A deserialized plan that can be run on CUDA tensors."""

    def __init__(self, plan: bytes) -> None:
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()

    def run(self, torch, inputs: dict, output: str):
        out = torch.empty(tuple(self.engine.get_tensor_shape(output)), device="cuda")
        for name, value in inputs.items():
            self.context.set_tensor_address(name, value.data_ptr())
        self.context.set_tensor_address(output, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(-1)

    def layer_kinds(self) -> list[str]:
        inspector = self.engine.create_engine_inspector()
        info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        return [json.dumps(layer) for layer in info["Layers"]]


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_quantized_engines_build_and_match_unquantized(tiny, precision) -> None:
    """Both FP8 engines build, are smaller, are finite, and track the unquantized engines."""
    torch = pytest.importorskip("torch")
    weights, cfg = tiny
    ctx = calibrate_clip(weights, cfg, "fp8", precision=precision)
    size = cfg["vision"]["image_size"]
    pixels = torch.randn(1, 3, size, size, device="cuda")
    ids = torch.randint(0, VOCAB, (WINDOW,), dtype=torch.int32, device="cuda")
    eos = torch.tensor([WINDOW - 3], dtype=torch.int32, device="cuda")

    common = {"precision": precision}
    plain_vision = model.build_vision_engine(weights, cfg, image_size=size, **common)
    plain_text = model.build_text_engine(weights, cfg, max_text_len=WINDOW, **common)
    quant_vision = model.build_vision_engine(weights, cfg, image_size=size, quant_ctx=ctx, **common)
    quant_text = model.build_text_engine(weights, cfg, max_text_len=WINDOW, quant_ctx=ctx, **common)
    assert len(quant_vision) < len(plain_vision)
    assert len(quant_text) < len(plain_text)

    cases = (
        ({"pixel_values": pixels}, "image_embeds", plain_vision, quant_vision),
        ({"input_ids": ids, "eos_index": eos}, "text_embeds", plain_text, quant_text),
    )
    for inputs, output, plain, quant in cases:
        reference = _Engine(plain).run(torch, inputs, output)
        actual = _Engine(quant).run(torch, inputs, output)
        assert torch.isfinite(actual).all(), output
        cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
        assert cosine > 0.9, (output, cosine)
    kinds = _Engine(quant_vision).layer_kinds()
    assert any('"FP8"' in kind for kind in kinds)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names(tiny) -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    weights, cfg = tiny
    ctx = calibrate_clip(weights, cfg, "fp8")
    _, _, config = model._builder("fp16", False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = model._builder("fp16", False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED


def test_build_rejects_unknown_quantization(tmp_path: Path) -> None:
    """Only fp8 or none are accepted."""
    from tensorrt_model_connect import BuildRequest

    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "b",
        family="clip",
        task="text_image_to_relevance",
        precision="fp16",
        quantization="nvfp4",
    )
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model.build(request, None)
