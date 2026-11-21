# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OWLv2 FP8 quantization: weight scaling, projection coverage, calibration tool, engine build."""

from __future__ import annotations

import json
import re
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from .. import calibrate_fp8_scales, model  # noqa: E402
from .. import config as config_module  # noqa: E402
from .. import quantization as quantization_module  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    DEFAULT_PARTS,
    HEAD_PROJECTIONS,
    LAYER_STEMS,
    calibrate_owlv2,
    projections,
    quantize_fp8_weight,
    scale_key,
    split_outlier_scales,
)

D, FFN, LAYERS, HEADS, PATCH, GRID, VOCAB, WINDOW, QUERY = 128, 256, 2, 4, 16, 4, 100, 8, 64
VISION = "owlv2.vision_model"
TEXT = "owlv2.text_model"


def _tiny_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with every tensor both engines read, at a tiny model config."""
    rng = np.random.default_rng(seed)

    def normal(*shape):
        return rng.normal(0.0, 0.05, shape).astype(np.float32)

    def norm(prefix):
        return {f"{prefix}.weight": np.ones(D, np.float32), f"{prefix}.bias": normal(D)}

    def linear(prefix, out, inp):
        return {f"{prefix}.weight": normal(out, inp), f"{prefix}.bias": normal(out)}

    weights: dict[str, np.ndarray] = {}
    for tower in (VISION, TEXT):
        for layer in range(LAYERS):
            prefix = f"{tower}.encoder.layers.{layer}"
            weights.update(norm(f"{prefix}.layer_norm1"))
            weights.update(norm(f"{prefix}.layer_norm2"))
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                weights.update(linear(f"{prefix}.self_attn.{name}", D, D))
            weights.update(linear(f"{prefix}.mlp.fc1", FFN, D))
            weights.update(linear(f"{prefix}.mlp.fc2", D, FFN))
    weights[f"{TEXT}.embeddings.token_embedding.weight"] = normal(VOCAB, D)
    weights[f"{TEXT}.embeddings.position_embedding.weight"] = normal(WINDOW, D)
    weights.update(norm(f"{TEXT}.final_layer_norm"))
    weights["owlv2.text_projection.weight"] = normal(QUERY, D)
    weights[f"{VISION}.embeddings.patch_embedding.weight"] = normal(D, 3, PATCH, PATCH)
    weights[f"{VISION}.embeddings.class_embedding"] = normal(D)
    weights[f"{VISION}.embeddings.position_embedding.weight"] = normal(GRID * GRID + 1, D)
    weights.update(norm(f"{VISION}.pre_layernorm"))
    weights.update(norm(f"{VISION}.post_layernorm"))
    weights.update(norm("layer_norm"))
    weights.update(linear("box_head.dense0", D, D))
    weights.update(linear("box_head.dense1", D, D))
    weights.update(linear("box_head.dense2", 4, D))
    weights.update(linear("class_head.dense0", QUERY, D))
    weights.update(linear("class_head.logit_shift", 1, D))
    weights.update(linear("class_head.logit_scale", 1, D))
    return weights


def _config(tmp_path: Path, weights: dict) -> dict:
    """Resolve the tiny model config the way ``build`` does."""
    stated = {"num_attention_heads": HEADS}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "owlv2",
                "text_config": dict(stated),
                "vision_config": dict(stated, image_size=PATCH * GRID, patch_size=PATCH),
            }
        )
    )
    return config_module.resolve(tmp_path, {key: value.shape for key, value in weights.items()})


def _all_scales(cfg: dict, value: float = 0.05) -> dict[str, float | None]:
    return {name: value for name in projections(cfg)}


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch):
    """Tiny weights, their resolved config, and activation scales for every projection."""
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


def test_default_part_is_the_text_tower_only(tiny) -> None:
    """Six projections per text layer get an FP8 weight of the checkpoint's shape."""
    weights, cfg = tiny
    ctx = calibrate_owlv2(weights, cfg, "fp8")
    assert DEFAULT_PARTS == ("text",)
    assert len(ctx.scales) == LAYERS * 6
    entry = ctx.scales[f"{TEXT}.encoder.layers.1.mlp.fc1"]
    assert entry.weight.packed.shape == (FFN, D)
    assert not ctx.should_quantize(f"{VISION}.encoder.layers.1.mlp.fc1")
    assert ctx.has_fp8_weights
    assert not ctx.should_quantize("box_head.dense0")
    assert not ctx.should_quantize("owlv2.text_projection")


def test_parts_select_what_is_quantized(tiny) -> None:
    """The three parts can be quantized on their own."""
    weights, cfg = tiny
    vision = calibrate_owlv2(weights, cfg, "fp8", ("vision",))
    assert set(vision.scales) == {n for n in projections(cfg) if n.startswith(VISION)}
    heads = calibrate_owlv2(weights, cfg, "fp8", ("heads",))
    assert set(heads.scales) == set(HEAD_PROJECTIONS)
    assert not heads.should_quantize("box_head.dense2")
    assert not heads.should_quantize("class_head.logit_scale")
    with pytest.raises(ValueError, match="unknown OWLv2 FP8 parts"):
        calibrate_owlv2(weights, cfg, "fp8", ("decoder",))


def test_null_scale_keeps_the_projection_in_fp16(tmp_path: Path, monkeypatch) -> None:
    """An outlier projection stored as null is not quantized and has no matmul override."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    name = f"{TEXT}.encoder.layers.0.self_attn.k_proj"
    scales[name] = None
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    ctx = calibrate_owlv2(weights, cfg, "fp8")
    assert not ctx.should_quantize(name)
    assert ctx.matmul(name) is None
    assert ctx.should_quantize(f"{TEXT}.encoder.layers.0.self_attn.q_proj")


def test_projection_without_entry_is_an_error(tmp_path: Path, monkeypatch) -> None:
    """A projection missing from the table fails and names the calibration tool."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    scales = _all_scales(cfg)
    del scales[f"{TEXT}.encoder.layers.0.mlp.fc2"]
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="calibrate_fp8_scales.py"):
        calibrate_owlv2(weights, cfg, "fp8")


def test_non_positive_scale_is_rejected(tiny, monkeypatch) -> None:
    """A zero or negative activation scale is refused."""
    weights, cfg = tiny
    scales = _all_scales(cfg, value=0.0)
    monkeypatch.setattr(quantization_module, "load_activation_scales", lambda _cfg: scales)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_owlv2(weights, cfg, "fp8")


def test_already_quantized_checkpoint_is_rejected(tiny) -> None:
    """A checkpoint that already stores weight scales is not quantized a second time."""
    weights, cfg = tiny
    weights[f"{TEXT}.encoder.layers.0.mlp.fc1.weight_scale"] = np.ones((1,), np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_owlv2(weights, cfg, "fp8")


def test_unsupported_format_is_rejected(tiny) -> None:
    """Only fp8 is implemented."""
    weights, cfg = tiny
    with pytest.raises(NotImplementedError, match="nvfp4"):
        calibrate_owlv2(weights, cfg, "nvfp4")


def test_unknown_model_config_names_the_calibration_tool(tmp_path: Path) -> None:
    """A model config with no committed scales fails with the available ones."""
    weights = _tiny_weights()
    cfg = _config(tmp_path, weights)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        calibrate_owlv2(weights, cfg, "fp8")


def test_scale_key_comes_from_the_resolved_model_config(tmp_path: Path) -> None:
    """The key names both towers, the MLP sizes, the patch, the image size and the prompt."""
    cfg = _config(tmp_path, _tiny_weights())
    expected = f"owlv2-v{D}x{LAYERS}i{FFN}p{PATCH}s{PATCH * GRID}-t{D}x{LAYERS}i{FFN}w{WINDOW}"
    assert scale_key(cfg) == expected


def test_outliers_are_split_per_kind() -> None:
    """A scale above 20x the median of its kind becomes None; other kinds are unaffected."""
    scales = {f"{VISION}.encoder.layers.{i}.mlp.fc2": 0.01 for i in range(5)}
    scales[f"{VISION}.encoder.layers.2.mlp.fc2"] = 0.5
    scales.update({f"{VISION}.encoder.layers.{i}.mlp.fc1": 0.5 for i in range(5)})
    split = split_outlier_scales(scales)
    assert split[f"{VISION}.encoder.layers.2.mlp.fc2"] is None
    assert split[f"{VISION}.encoder.layers.1.mlp.fc2"] == 0.01
    assert all(split[f"{VISION}.encoder.layers.{i}.mlp.fc1"] == 0.5 for i in range(5))
    assert split_outlier_scales({"box_head.dense0": 3.0}) == {"box_head.dense0": 3.0}


def test_committed_scales_cover_every_projection() -> None:
    """Every committed model config has an entry (a scale or null) for each projection."""
    table = json.loads(
        (Path(quantization_module.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert table
    for key, scales in table.items():
        match = re.fullmatch(r"owlv2-v(\d+)x(\d+)i(\d+)p(\d+)s(\d+)-t(\d+)x(\d+)i(\d+)w(\d+)", key)
        assert match, key
        vision_layers, text_layers = int(match.group(2)), int(match.group(7))
        assert len(scales) == (vision_layers + text_layers) * len(LAYER_STEMS) + len(
            HEAD_PROJECTIONS
        ), key
        assert all(v is None or (np.isfinite(v) and v > 0) for v in scales.values()), key
        for prefix, layers in ((VISION, vision_layers), (TEXT, text_layers)):
            for layer in range(layers):
                for stem in LAYER_STEMS:
                    assert f"{prefix}.encoder.layers.{layer}.{stem}" in scales, (key, prefix, layer)
        for name in HEAD_PROJECTIONS:
            assert name in scales, (key, name)


def test_calibration_tool_maps_module_names_to_checkpoint_names() -> None:
    """Only tower projections and the three head projections are calibrated."""
    name = calibrate_fp8_scales.projection_name
    assert name(f"{VISION}.encoder.layers.3.mlp.fc1") == f"{VISION}.encoder.layers.3.mlp.fc1"
    assert name(f"{TEXT}.encoder.layers.0.self_attn.out_proj") is not None
    assert name("box_head.dense1") == "box_head.dense1"
    assert name("class_head.dense0") == "class_head.dense0"
    assert name("box_head.dense2") is None
    assert name("class_head.logit_scale") is None
    assert name("objectness_head.dense0") is None
    assert name(f"{VISION}.embeddings.patch_embedding") is None


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

    def run(self, torch, inputs: dict, output: str):
        """Run once and return the flattened ``output``; every output tensor needs an address."""
        outputs = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                outputs[name] = torch.empty(shape, device="cuda")
                self.context.set_tensor_address(name, outputs[name].data_ptr())
        for name, value in inputs.items():
            self.context.set_tensor_address(name, value.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return outputs[output].reshape(-1)


@pytest.mark.gpu
@pytest.mark.trt
def test_quantized_engines_build_and_match_fp16(tiny) -> None:
    """Both FP8 engines build, are smaller, and track the fp16 engines."""
    torch = pytest.importorskip("torch")
    weights, cfg = tiny
    ctx = calibrate_owlv2(weights, cfg, "fp8", ("vision", "text", "heads"))
    pixels = torch.randn(1, 3, PATCH * GRID, PATCH * GRID, device="cuda")
    ids = torch.randint(1, VOCAB, (WINDOW,), dtype=torch.int32, device="cuda")
    eos = torch.tensor([WINDOW - 1], dtype=torch.int32, device="cuda")

    plain_vision = model.build_vision_engine(weights, cfg, precision="fp16")
    plain_text = model.build_text_engine(weights, cfg, max_text_len=WINDOW, precision="fp16")
    quant_vision = model.build_vision_engine(weights, cfg, precision="fp16", quant_ctx=ctx)
    quant_text = model.build_text_engine(
        weights, cfg, max_text_len=WINDOW, precision="fp16", quant_ctx=ctx
    )
    assert len(quant_vision) < len(plain_vision)
    assert len(quant_text) < len(plain_text)

    cases = (
        ({"pixel_values": pixels}, "class_embeds", plain_vision, quant_vision),
        ({"pixel_values": pixels}, "pred_boxes", plain_vision, quant_vision),
        ({"input_ids": ids, "eos_index": eos}, "query_embeds", plain_text, quant_text),
    )
    for inputs, output, plain, quant in cases:
        reference = _Engine(plain).run(torch, inputs, output)
        actual = _Engine(quant).run(torch, inputs, output)
        assert torch.isfinite(actual).all(), output
        cosine = torch.nn.functional.cosine_similarity(reference, actual, dim=0).item()
        assert cosine > 0.9, (output, cosine)


@pytest.mark.gpu
@pytest.mark.trt
def test_fp8_build_options_keep_layer_names() -> None:
    """FP8 builds ask for detailed profiling so the plan inspector shows kernel names."""
    ctx = quantization_module.Owlv2QuantContext(scales={"x": object()})
    _, _, config = model._builder("fp16", False, ctx)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
    _, _, plain = model._builder("fp16", False)
    assert plain.profiling_verbosity != trt.ProfilingVerbosity.DETAILED


def test_build_rejects_formats_other_than_fp8(tmp_path: Path) -> None:
    """Only fp8 or no quantization is accepted by the family build."""
    from tensorrt_model_connect import BuildRequest

    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "bundle",
        family="owlv2",
        task="image_text_to_boxes",
        precision="fp16",
        quantization="nvfp4",
    )
    with pytest.raises(NotImplementedError, match="fp8 or none"):
        model.build(request, None)
