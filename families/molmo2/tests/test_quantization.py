# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 self-quantization of the Molmo2 decoder: recipe, scales, rejected configs, tiny engine."""

from dataclasses import replace
import json
from pathlib import Path
import statistics

import ml_dtypes
import numpy as np
import pytest

from families.molmo2 import calibrate_scales, quantization
from families.molmo2.model import build, validate_request
from families.molmo2.quantization import (
    ACTIVATION_SCALES_FILENAME,
    IDENTITY_TENSOR,
    PROJECTIONS,
    DecoderQuantization,
    checkpoint_identity,
    decoder_quantization,
    quantize_fp8_weight,
    scale_key,
)
from families.molmo2.tests.test_support import metadata
from tensorrt_model_connect.build import BuildRequest

SCALES = json.loads((Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text())
KEY = "molmo2-h4096-l36-i12288"
TEXT = {"hidden_size": 4096, "num_hidden_layers": 36, "intermediate_size": 12288}
CONFIG = {"text_config": TEXT}


class _Weights:
    """Tensor names and a final norm weight; by default the calibrated checkpoint's first values."""

    def __init__(self, names=(), norm=None):
        self.index = {name: "model.safetensors" for name in names}
        if norm is None:
            norm = np.ones(TEXT["hidden_size"], np.float32)
            norm[:8] = SCALES[KEY]["identity"]
        self._norm = norm

    def __call__(self, name):
        assert name == IDENTITY_TENSOR, name
        return self._norm


def test_fp8_weight_scale_is_amax_over_448_and_round_trips():
    weight = np.random.default_rng(0).standard_normal((32, 48)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448)
    restored = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    # e4m3 keeps 3 mantissa bits: relative error at most 2**-4 above the subnormal range.
    np.testing.assert_allclose(restored, weight, rtol=2**-4, atol=scale * 2**-9)
    assert np.abs(restored).max() == pytest.approx(np.abs(weight).max())


def test_fp8_weight_without_a_finite_amax_is_rejected():
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 16), np.float32))


def test_scale_file_covers_every_projection_of_the_published_model():
    assert list(SCALES) == [KEY] == [scale_key(TEXT)]
    entry = SCALES[KEY]
    assert entry["checkpoint"] == "allenai/Molmo2-8B@e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
    identity = entry["identity"]
    assert len(identity) == 8
    assert all(float(np.float32(v).astype(ml_dtypes.bfloat16)) == v for v in identity)
    assert set(entry["scales"]) == {
        f"layer.{layer}.{stem}" for layer in range(36) for stem in PROJECTIONS
    }
    values = [value for value in entry["scales"].values() if value is not None]
    assert all(np.isfinite(value) and value > 0 for value in values)
    # Three down-projection (ff_out) inputs are outliers (126x, 65x and 36x the median).
    assert sorted(name for name, value in entry["scales"].items() if value is None) == [
        "layer.16.mlp.ff_out",
        "layer.35.mlp.ff_out",
        "layer.6.mlp.ff_out",
    ]


def test_outlier_rule_drops_only_scales_above_20x_the_median_of_their_kind():
    scales = {f"layer.{layer}.{stem}": 1.0 for layer in range(4) for stem in PROJECTIONS}
    scales["layer.3.mlp.ff_out"] = 21.0
    scales["layer.2.self_attn.attn_out"] = 19.0
    kept = calibrate_scales.drop_outliers(scales)
    assert kept["layer.3.mlp.ff_out"] is None
    assert kept["layer.2.self_attn.attn_out"] == 19.0
    assert sum(value is None for value in kept.values()) == 1


def test_calibration_matches_decoder_projections_only():
    name = calibrate_scales.projection_name
    assert name("model.transformer.blocks.3.mlp.ff_out") == "layer.3.mlp.ff_out"
    assert name("transformer.blocks.12.self_attn.att_proj") == "layer.12.self_attn.att_proj"
    for other in (
        "model.transformer.blocks.3.self_attn.q_norm",
        "model.vision_backbone.image_vit.transformer.resblocks.3.attention.wo",
        "model.vision_backbone.image_projector.w2",
        "model.transformer.blocks.3.mlp.ff_out.input_quantizer",
        "lm_head",
    ):
        assert name(other) is None


def test_fp8_recipe_quantizes_every_decoder_projection_except_outliers():
    recipe = decoder_quantization(CONFIG, _Weights(), "fp8")
    expected = {
        f"model.transformer.blocks.{name[len('layer.') :]}": value
        for name, value in SCALES[KEY]["scales"].items()
        if value is not None
    }
    assert recipe.input_scales == expected and len(expected) == 4 * 36 - 3
    for layer in (6, 16, 35):
        assert f"model.transformer.blocks.{layer}.mlp.ff_out" not in recipe.input_scales
    for stem in PROJECTIONS:
        kind = [value for name, value in recipe.input_scales.items() if name.endswith(stem)]
        assert max(kind) < 20 * statistics.median(kind)


def test_missing_or_invalid_scale_entries_are_rejected(monkeypatch):
    for broken, error in (
        (lambda entry: entry.pop("layer.3.mlp.ff_proj"), "no entry"),
        (lambda entry: entry.update({"layer.3.mlp.ff_proj": 0.0}), "positive"),
        (lambda entry: entry.update({"layer.3.mlp.ff_proj": float("nan")}), "positive"),
    ):
        entry = dict(SCALES[KEY]["scales"])
        broken(entry)
        monkeypatch.setattr(quantization, "load_activation_scales", lambda text, identity: entry)
        with pytest.raises(ValueError, match=error):
            decoder_quantization(CONFIG, _Weights(), "fp8")


def test_other_checkpoints_and_quantized_checkpoints_are_rejected():
    other = {"text_config": {**TEXT, "num_hidden_layers": 28}}
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(other, _Weights(), "fp8")
    # Same decoder config, other weights (for example a fine-tuned checkpoint).
    norm = _Weights()(IDENTITY_TENSOR).copy()
    norm[3] *= 1.01
    assert checkpoint_identity(_Weights(norm=norm)) != SCALES[KEY]["identity"]
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(CONFIG, _Weights(norm=norm), "fp8")
    quantized = ["model.transformer.blocks.0.mlp.ff_out.weight_scale"]
    with pytest.raises(ValueError, match="already quantized"):
        decoder_quantization(CONFIG, _Weights(quantized), "fp8")
    with pytest.raises(ValueError, match="already quantized"):
        decoder_quantization(
            {**CONFIG, "quantization_config": {"quant_method": "compressed-tensors"}},
            _Weights(),
            "fp8",
        )
    with pytest.raises(ValueError, match="quantization"):
        decoder_quantization(CONFIG, _Weights(), "nvfp4")


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


def _request(**change):
    return {
        "backend": "trt",
        "context_parallel_size": 1,
        "dynamic_kv_cache": False,
        "fp32_layers": [],
        "image_height": None,
        "image_width": None,
        "max_batch_size": 1,
        "mtp_seq_len": None,
        "precision": "bf16",
        "quantization": None,
        "task": "vision_language_generation",
        "tensor_parallel_size": 1,
        "video_num_frames": None,
        **change,
    }


def test_fp8_is_accepted_for_bf16_only():
    validate_request(metadata(), **_request(quantization="fp8"))
    validate_request(metadata(), **_request(quantization="none"))
    with pytest.raises(ValueError, match="quantization=fp8 requires precision=bf16"):
        validate_request(metadata(), **_request(quantization="fp8", precision="fp32"))


@pytest.mark.parametrize("value", ["nvfp4", "int8", "int4_awq"])
def test_build_rejects_other_quantization_formats(tmp_path, value):
    (tmp_path / "config.json").write_text(json.dumps(metadata()))
    request = BuildRequest(
        tmp_path, Path("output.bundle"), "molmo2", "vision_language_generation", "bf16"
    )
    with pytest.raises(ValueError, match="quantization"):
        build(replace(request, quantization=value), None)


TINY = {
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "rope_theta": 10000.0,
    "layer_norm_eps": 1e-6,
    "vocab_size": 128,
}


class _TinyWeights:
    """Random BF16-valued decoder weights of the tiny config, by checkpoint name."""

    def __init__(self):
        rng = np.random.default_rng(0)
        hidden, inner, kv = TINY["hidden_size"], TINY["intermediate_size"], 2 * 16
        tensors = {"model.transformer.ln_f.weight": 1 + 0.1 * rng.standard_normal(hidden)}
        tensors["lm_head.weight"] = 0.2 * rng.standard_normal((128, hidden))
        for layer in range(2):
            p = f"model.transformer.blocks.{layer}."
            tensors[p + "attn_norm.weight"] = 1 + 0.1 * rng.standard_normal(hidden)
            tensors[p + "ff_norm.weight"] = 1 + 0.1 * rng.standard_normal(hidden)
            tensors[p + "self_attn.q_norm.weight"] = 1 + 0.1 * rng.standard_normal(16)
            tensors[p + "self_attn.k_norm.weight"] = 1 + 0.1 * rng.standard_normal(16)
            tensors[p + "self_attn.att_proj.weight"] = 0.1 * rng.standard_normal(
                (hidden + 2 * kv, hidden)
            )
            tensors[p + "self_attn.attn_out.weight"] = 0.1 * rng.standard_normal((hidden, hidden))
            tensors[p + "mlp.ff_proj.weight"] = 0.1 * rng.standard_normal((2 * inner, hidden))
            tensors[p + "mlp.ff_out.weight"] = 0.1 * rng.standard_normal((hidden, inner))
        self.tensors = {
            name: value.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
            for name, value in tensors.items()
        }
        self.index = dict.fromkeys(self.tensors, "model.safetensors")

    def __call__(self, name):
        return self.tensors[name]


def test_tiny_fp8_decoder_builds_with_fp8_weights_and_tracks_bf16(monkeypatch):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT build requires CUDA")
    from families.molmo2.model import build_decoder

    config = {"text_config": TINY}
    weights = _TinyWeights()
    scales = {f"layer.{layer}.{stem}": 0.02 for layer in range(2) for stem in PROJECTIONS}
    scales["layer.1.mlp.ff_out"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda text, identity: scales)
    recipe = decoder_quantization(config, weights, "fp8")
    assert len(recipe.input_scales) == 7

    maximum, rows = 16, 3
    plans = {
        name: build_decoder(config, weights, "bf16", maximum, quantization=quant)
        for name, quant in (("bf16", None), ("fp8", recipe))
    }
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    generator = torch.Generator().manual_seed(1)
    embeddings = torch.randn(rows, 64, generator=generator).cuda()
    mask = torch.full((1, rows, 1 + rows), -1e30)
    mask[0, :, 1:] = torch.triu(torch.full((rows, rows), -1e30), diagonal=1)
    logits = {}
    for name, plan in plans.items():
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        tensors = {
            "embeddings": embeddings,
            "positions": torch.arange(rows, dtype=torch.int32).cuda(),
            "attention_mask": mask.cuda(),
            "logits": torch.empty(1, 128, device="cuda"),
        }
        for layer in range(2):
            for kind in ("k", "v"):
                tensors[f"past_{kind}_{layer}"] = torch.zeros(
                    1, 32, dtype=torch.bfloat16, device="cuda"
                )
                tensors[f"new_{kind}_{layer}"] = torch.empty(
                    rows, 32, dtype=torch.bfloat16, device="cuda"
                )
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
