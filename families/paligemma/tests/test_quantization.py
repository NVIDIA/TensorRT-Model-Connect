# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of the PaliGemma decoder: recipe, scales, rejected configs, tiny engine."""

import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from families.paligemma import calibrate_scales, quantization
from families.paligemma.config import ModelConfig
from families.paligemma.model import build
from families.paligemma.quantization import (
    ACTIVATION_SCALES_FILENAME,
    PROJECTIONS,
    DecoderQuantization,
    decoder_quantization,
    final_norm_head,
    quantize_fp8_weight,
    scale_key,
)
from tensorrt_model_connect import BuildRequest

from .test_model import PALIGEMMA_3B_PT_224, _Writer

SCALES = json.loads((Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text())
TEXT = ModelConfig.from_dict(PALIGEMMA_3B_PT_224).text
KEY = "paligemma-h2048-l18-i16384"


class _Weights:
    """Tensor names and a final norm weight; by default the calibrated checkpoint's first values."""

    def __init__(self, names=(), norm=None):
        self._names = list(names)
        if norm is None:
            norm = np.zeros(TEXT.hidden_size, np.float32)
            norm[:8] = SCALES[KEY]["final_norm_head"]
        self._norm = norm

    def names(self):
        return self._names

    def __getitem__(self, name):
        assert name == "text.norm.weight", name
        return self._norm


def test_fp8_weight_scale_is_amax_over_448_and_round_trips():
    weight = np.random.default_rng(0).standard_normal((32, 48)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448)
    import ml_dtypes

    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    # e4m3 keeps 3 mantissa bits: relative error at most 2**-4 above the subnormal range.
    np.testing.assert_allclose(restored, weight, rtol=2**-4, atol=scale * 2**-9)
    assert np.abs(restored).max() == pytest.approx(np.abs(weight).max())


def test_fp8_weight_without_a_finite_amax_is_rejected():
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 16), np.float32))


def test_scale_file_covers_every_quantized_projection_of_the_published_model():
    assert list(SCALES) == [KEY] == [scale_key(TEXT)]
    head = SCALES[KEY]["final_norm_head"]
    assert len(head) == 8 and all(float(np.float32(value)) == value for value in head)
    entry = SCALES[KEY]["scales"]
    assert set(entry) == {
        f"layer.{layer}.{stem}" for layer in range(TEXT.num_hidden_layers) for stem in PROJECTIONS
    }
    values = [value for value in entry.values() if value is not None]
    assert all(np.isfinite(value) and value > 0 for value in values)
    # The last down projection's input is an outlier (28.8x the median of its kind).
    assert [name for name, value in entry.items() if value is None] == ["layer.17.mlp.down_proj"]


def test_outlier_rule_drops_only_scales_above_20x_the_median_of_their_kind():
    scales = {f"layer.{layer}.{stem}": 1.0 for layer in range(4) for stem in PROJECTIONS}
    scales["layer.3.mlp.down_proj"] = 21.0
    scales["layer.2.self_attn.o_proj"] = 19.0
    kept = calibrate_scales.drop_outliers(scales)
    assert kept["layer.3.mlp.down_proj"] is None
    assert kept["layer.2.self_attn.o_proj"] == 19.0
    assert sum(value is None for value in kept.values()) == 1


def test_calibration_matches_recipe_projections_of_both_module_layouts():
    pattern = calibrate_scales._PROJECTION
    for name in (
        "language_model.model.layers.3.mlp.down_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ):
        assert pattern.search(name).groups() == ("3", "mlp.down_proj")
    for name in (
        "model.language_model.layers.3.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.3.mlp.fc1",
        "model.language_model.layers.3.mlp.down_proj.input_quantizer",
    ):
        assert pattern.search(name) is None


def test_fp8_recipe_quantizes_o_gate_up_down_and_keeps_qkv_bf16():
    recipe = decoder_quantization(TEXT, _Weights(), "fp8")
    expected = {
        f"text.layers.{name[len('layer.') :]}": value
        for name, value in SCALES[KEY]["scales"].items()
        if value is not None
    }
    assert recipe.input_scales == expected and len(expected) == 4 * 18 - 1
    assert "text.layers.17.mlp.down_proj" not in recipe.input_scales
    assert not any(
        stem in name for name in recipe.input_scales for stem in ("q_proj", "k_proj", "v_proj")
    )
    for stem in PROJECTIONS:
        kind = [value for name, value in recipe.input_scales.items() if name.endswith(stem)]
        assert max(kind) < 20 * statistics.median(kind)


def test_missing_or_invalid_scale_entries_are_rejected(monkeypatch):
    for broken, error in (
        (lambda entry: entry.pop("layer.3.mlp.up_proj"), "no entry"),
        (lambda entry: entry.update({"layer.3.mlp.up_proj": 0.0}), "positive"),
        (lambda entry: entry.update({"layer.3.mlp.up_proj": float("nan")}), "positive"),
    ):
        entry = dict(SCALES[KEY]["scales"])
        broken(entry)
        monkeypatch.setattr(quantization, "load_activation_scales", lambda text, weights: entry)
        with pytest.raises(ValueError, match=error):
            decoder_quantization(TEXT, _Weights(), "fp8")


def test_other_checkpoints_and_quantized_checkpoints_are_rejected():
    other = ModelConfig.from_dict(
        {
            **PALIGEMMA_3B_PT_224,
            "text_config": {**PALIGEMMA_3B_PT_224["text_config"], "num_hidden_layers": 12},
        }
    ).text
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(other, _Weights(), "fp8")
    # Same decoder config, other weights (for example the 448-pixel checkpoint).
    norm = _Weights()["text.norm.weight"].copy()
    norm[3] += 1e-3
    assert final_norm_head(_Weights(norm=norm)) != SCALES[KEY]["final_norm_head"]
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(TEXT, _Weights(norm=norm), "fp8")
    with pytest.raises(ValueError, match="already quantized"):
        decoder_quantization(TEXT, _Weights(["text.layers.0.mlp.up_proj.weight_scale"]), "fp8")
    with pytest.raises(ValueError, match="quantization"):
        decoder_quantization(TEXT, _Weights(), "nvfp4")


@pytest.mark.parametrize(
    "routes, route",
    [("peep:match_dual_gemm,peep:other", "-peep:match_dual_gemm=off"), ("peep:other", "")],
)
def test_fused_gate_up_kernel_is_turned_off_when_the_knob_is_listed(routes, route):
    trt = pytest.importorskip("tensorrt")

    class Config:
        all_build_routes = routes
        build_route = ""
        profiling_verbosity = None

    config = Config()
    DecoderQuantization({}).configure(config)
    assert config.build_route == route
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


@pytest.mark.parametrize(
    "change",
    [
        {"quantization": "int8"},
        {"quantization": "nvfp4"},
        {"quantization": "fp8", "precision": "fp32"},
    ],
)
def test_build_rejects_unsupported_quantization_requests(tmp_path, change):
    (tmp_path / "config.json").write_text(json.dumps(PALIGEMMA_3B_PT_224))
    request = BuildRequest(
        **{
            "model_dir": tmp_path,
            "output_path": tmp_path / "x.bundle",
            "family": "paligemma",
            "task": "images_text_to_text",
            "precision": "bf16",
            **change,
        }
    )
    with pytest.raises(ValueError, match="quantization"):
        build(request, _Writer())


def test_tiny_fp8_decoder_builds_with_fp8_weights_and_tracks_bf16(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT build requires CUDA")
    from families.paligemma.checkpoint import CheckpointWeights
    from families.paligemma.graph import build_decoder

    config = transformers.PaliGemmaConfig(
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 28,
            "patch_size": 14,
            "vision_use_head": False,
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 128,
        },
        image_token_index=127,
        projection_dim=64,
        hidden_size=64,
    )
    torch.manual_seed(0)
    transformers.PaliGemmaForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(
        tmp_path
    )
    model_config = ModelConfig.from_dir(tmp_path)
    weights = CheckpointWeights(tmp_path)
    scales = {f"layer.{layer}.{stem}": 0.05 for layer in range(2) for stem in PROJECTIONS}
    scales["layer.1.mlp.down_proj"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda text, weights: scales)
    recipe = decoder_quantization(model_config.text, weights, "fp8")
    assert len(recipe.input_scales) == 7

    capacity = 8
    plans = {
        name: build_decoder(model_config, weights, capacity, precision="bf16", quantization=quant)
        for name, quant in (("bf16", None), ("fp8", recipe))
    }
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    rows = torch.randn(3, 64, generator=torch.Generator().manual_seed(1)).to("cuda", torch.bfloat16)
    logits = {}
    for name, plan in plans.items():
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        tensors = {
            "inputs_embeds": rows,
            "rope_cos": torch.ones(3, 16, device="cuda"),
            "rope_sin": torch.zeros(3, 16, device="cuda"),
            "attention_mask": torch.cat(
                [torch.full((1, 3, capacity), -1e30), torch.zeros(1, 3, 3)], -1
            ).cuda(),
        }
        for layer in range(2):
            for kind in ("k", "v"):
                tensors[f"cache_{kind}_{layer}"] = torch.zeros(
                    1, capacity, 16, dtype=torch.bfloat16, device="cuda"
                )
                tensors[f"present_{kind}_{layer}"] = torch.empty(
                    1, 3, 16, dtype=torch.bfloat16, device="cuda"
                )
        tensors["logits"] = torch.empty(1, 128, device="cuda")
        for tensor_name, tensor in tensors.items():
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor_name, tuple(tensor.shape))
            context.set_tensor_address(tensor_name, tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        logits[name] = tensors["logits"][0].clone()
        if name == "fp8":
            info = (
                engine.create_engine_inspector()
                .get_engine_information(trt.LayerInformationFormat.JSON)
                .lower()
            )
            assert "fp8" in info or "e4m3" in info
    assert torch.isfinite(logits["fp8"]).all()
    cosine = torch.nn.functional.cosine_similarity(logits["fp8"], logits["bf16"], dim=0)
    assert cosine > 0.98
