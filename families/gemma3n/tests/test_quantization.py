# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of the Gemma 3n decoder: recipe, scales, rejected configs, tiny engine."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

pytest.importorskip("tensorrt")

from families.gemma3n import calibrate_scales, quantization  # noqa: E402
from families.gemma3n.config import ModelConfig, TextConfig  # noqa: E402
from families.gemma3n.model import build  # noqa: E402
from families.gemma3n.quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    PROJECTIONS,
    DecoderQuantization,
    decoder_quantization,
    final_norm_head,
    quantize_fp8_weight,
    scale_key,
)
from tensorrt_model_connect import BuildRequest  # noqa: E402

SCALES = json.loads((Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text())
# The published E4B text config (defaults of the reference config).
E4B = TextConfig.from_raw({"model_type": "gemma3n_text"})
KEY = "gemma3n-h2048-l35-i16384"
NORM = "model.language_model.norm.weight"


class _Checkpoint:
    """Tensor names and a final norm weight; by default the calibrated checkpoint's first values."""

    def __init__(self, names=(), norm=None):
        if norm is None:
            norm = np.zeros(E4B.hidden_size, np.float32)
            norm[:8] = SCALES[KEY]["final_norm_head"]
        self.mapping = {name: "model.safetensors" for name in (*names, NORM)}
        self._norm = norm

    def get(self, name):
        assert name == NORM, name
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


def test_scale_file_covers_every_quantized_projection_of_the_published_model():
    assert list(SCALES) == [KEY] == [scale_key(E4B)]
    head = SCALES[KEY]["final_norm_head"]
    assert len(head) == 8 and all(float(np.float32(value)) == value for value in head)
    entry = SCALES[KEY]["scales"]
    assert (
        set(entry)
        == set(calibrate_scales.expected_names(E4B))
        == {f"layer.{layer}.{stem}" for layer in range(35) for stem in PROJECTIONS}
    )
    values = [value for value in entry.values() if value is not None]
    assert all(np.isfinite(value) and value > 0 for value in values)
    assert sorted(name for name, value in entry.items() if value is None) == OUTLIERS


OUTLIERS = sorted(
    [
        "layer.0.mlp.down_proj",
        "layer.1.mlp.down_proj",
        "layer.4.mlp.down_proj",
        "layer.4.mlp.gate_proj",
        "layer.4.mlp.up_proj",
    ]
)


def test_outlier_rule_drops_only_scales_above_20x_the_median_of_their_kind():
    scales = {f"layer.{layer}.{stem}": 1.0 for layer in range(4) for stem in PROJECTIONS}
    scales["layer.3.mlp.down_proj"] = 21.0
    scales["layer.2.mlp.gate_proj"] = 19.0
    kept = calibrate_scales.drop_outliers(scales)
    assert kept["layer.3.mlp.down_proj"] is None
    assert kept["layer.2.mlp.gate_proj"] == 19.0
    assert sum(value is None for value in kept.values()) == 1


def test_calibration_matches_the_recipe_projections_only():
    pattern = calibrate_scales._PROJECTION
    for name in (
        "model.language_model.layers.3.mlp.down_proj",
        "model.language_model.layers.21.mlp.gate_proj",
    ):
        assert pattern.search(name).group(1) in ("3", "21")
    for name in (
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.laurel.linear_left",
        "model.language_model.layers.3.per_layer_input_gate",
        "model.language_model.layers.3.altup.modality_router",
        "model.vision_tower.timm_model.blocks.3.0.attn.query.proj",
        "model.language_model.layers.3.mlp.down_proj.input_quantizer",
    ):
        assert pattern.search(name) is None


def test_fp8_recipe_quantizes_gate_up_down_and_keeps_attention_bf16():
    recipe = decoder_quantization(E4B, _Checkpoint(), "fp8")
    expected = {
        f"model.language_model.layers.{name[len('layer.') :]}": value
        for name, value in SCALES[KEY]["scales"].items()
        if value is not None
    }
    assert recipe.input_scales == expected and len(expected) == 35 * 3 - len(OUTLIERS)
    assert not any("self_attn" in name for name in recipe.input_scales)
    assert "model.language_model.layers.4.mlp.gate_proj" not in recipe.input_scales
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
        monkeypatch.setattr(quantization, "load_activation_scales", lambda text, ckpt: entry)
        with pytest.raises(ValueError, match=error):
            decoder_quantization(E4B, _Checkpoint(), "fp8")


def test_other_checkpoints_and_quantized_checkpoints_are_rejected():
    # Another config (the E2B layout: 30 layers, 8192-wide MLP).
    other = TextConfig.from_raw(
        {
            "model_type": "gemma3n_text",
            "num_hidden_layers": 30,
            "intermediate_size": 8192,
            "num_kv_shared_layers": 10,
        }
    )
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(other, _Checkpoint(), "fp8")
    # Same config, other weights.
    norm = _Checkpoint().get(NORM).copy()
    norm[3] += 1e-3
    assert final_norm_head(_Checkpoint(norm=norm)) != SCALES[KEY]["final_norm_head"]
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(E4B, _Checkpoint(norm=norm), "fp8")
    with pytest.raises(ValueError, match="already quantized"):
        decoder_quantization(
            E4B, _Checkpoint(["model.language_model.layers.0.mlp.up_proj.weight_scale"]), "fp8"
        )
    for other_format in ("int8", "nvfp4"):
        with pytest.raises(ValueError, match="quantization"):
            decoder_quantization(E4B, _Checkpoint(), other_format)


@pytest.mark.parametrize(
    "routes, route",
    [("peep:match_dual_gemm,peep:other", "-peep:match_dual_gemm=off"), ("peep:other", "")],
)
def test_fused_gate_up_kernel_is_turned_off_when_the_knob_is_listed(routes, route):
    import tensorrt as trt

    class Config:
        all_build_routes = routes
        build_route = ""
        profiling_verbosity = None

    config = Config()
    DecoderQuantization({}).configure(config)
    assert config.build_route == route
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


@pytest.mark.parametrize(
    "change, error",
    [
        ({"quantization": "int8"}, "quantization must be"),
        ({"quantization": "nvfp4"}, "quantization must be"),
        ({"quantization": "fp8", "precision": "fp32"}, "BF16"),
    ],
)
def test_build_rejects_unsupported_quantization_requests(tmp_path, change, error):
    request = BuildRequest(
        **{
            "model_dir": tmp_path,
            "output_path": tmp_path / "x.bundle",
            "family": "gemma3n",
            "task": "images_text_to_text",
            "precision": "bf16",
            **change,
        }
    )
    with pytest.raises(ValueError, match=error):
        build(request, None)


# --- tiny engine ------------------------------------------------------------------


def _tiny_checkpoint():
    """A tiny Gemma 3n config and its text-side tensors (random, BF16), from the reference modules."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.gemma3n import modeling_gemma3n as reference

    text = transformers.Gemma3nTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=512,
        vocab_size_per_layer_input=256,
        hidden_size_per_layer_input=16,
        laurel_rank=8,
        num_kv_shared_layers=2,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"] * 2,
        activation_sparsity_pattern=[0.95, 0.0, 0.0, 0.0],
        max_position_embeddings=64,
    )
    vision = transformers.Gemma3nVisionConfig(hidden_size=2048, vocab_size=128, vocab_offset=256)
    audio = transformers.Gemma3nAudioConfig(hidden_size=32, vocab_size=128, vocab_offset=384)
    torch.manual_seed(0)
    tensors = {}
    modules = (
        ("model.language_model.", reference.Gemma3nTextModel(text)),
        ("model.embed_vision.", reference.Gemma3nMultimodalEmbedder(vision, text)),
        ("model.embed_audio.", reference.Gemma3nMultimodalEmbedder(audio, text)),
    )
    for prefix, module in modules:
        for name, value in module.state_dict().items():
            value = value.float()
            if name.endswith("norm.weight") or name.endswith("correct_output_scale"):
                value = 1.0 + 0.1 * torch.randn_like(value)
            tensors[prefix + name] = value.numpy().astype(ml_dtypes.bfloat16)
    raw = {
        "model_type": "gemma3n",
        "architectures": ["Gemma3nForConditionalGeneration"],
        "text_config": text.to_dict(),
        "vision_config": {**vision.to_dict(), "architecture": "mobilenetv5_300m_enc"},
        "audio_config": audio.to_dict(),
        "image_token_id": 300,
        "boi_token_id": 2,
        "eoi_token_id": 3,
    }
    return ModelConfig.from_raw(raw), tensors


class _Tensors:
    def __init__(self, tensors):
        self.tensors = tensors
        self.mapping = {name: "model.safetensors" for name in tensors}

    def get(self, name):
        return self.tensors[name]

    def get_f32(self, name, shape=None):
        value = self.tensors[name].astype(np.float32)
        assert shape is None or value.shape == tuple(shape)
        return value


def test_tiny_fp8_decoder_builds_with_fp8_weights_and_tracks_bf16(monkeypatch):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT build requires CUDA")
    from families.gemma3n.graph import Builder
    from families.gemma3n.model import _decoder_profiles, _engine
    from families.gemma3n.text_graph import TextGraph

    config, tensors = _tiny_checkpoint()
    checkpoint = _Tensors(tensors)
    scales = {name: 0.05 for name in calibrate_scales.expected_names(config.text)}
    scales["layer.3.mlp.down_proj"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda text, ckpt: scales)
    recipe = decoder_quantization(config.text, checkpoint, "fp8")
    assert len(recipe.input_scales) == 4 * 3 - 1

    capacity, rows = 16, 5
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    runtime = trt.Runtime(logger)
    logits = {}
    for name, quant in (("bf16", None), ("fp8", recipe)):
        network = builder.create_network(flags)
        graph = Builder(network, "bf16", checkpoint)
        TextGraph(graph, config, capacity, quant).build()
        plan = _engine(
            builder, network, "bf16", _decoder_profiles(config, capacity), False, "decoder", quant
        )
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        mask = torch.full((rows, capacity + rows), float("-inf"))
        mask[:, capacity:] = torch.triu(torch.full((rows, rows), float("-inf")), 1)
        tensors_in = {
            "token_id": torch.tensor([2, 17, 300, 41, 99], dtype=torch.int32),
            "position_id": torch.arange(rows, dtype=torch.int32),
            "attention_mask": mask,
            "media_features": torch.randn(rows, 64, generator=torch.Generator().manual_seed(1)).to(
                torch.bfloat16
            ),
            "media_mask": torch.tensor([0, 0, 1, 0, 0], dtype=torch.int32),
        }
        for layer in config.text.cache_layers():
            for kind in ("k", "v"):
                tensors_in[f"cache_{kind}_{layer}"] = torch.zeros(
                    capacity, 32, dtype=torch.bfloat16
                )
                tensors_in[f"present_{kind}_{layer}"] = torch.empty(rows, 32, dtype=torch.bfloat16)
        tensors_in["logits"] = torch.empty(1, 512)
        tensors_in = {key: value.cuda().contiguous() for key, value in tensors_in.items()}
        for tensor_name, tensor in tensors_in.items():
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor_name, tuple(tensor.shape))
            context.set_tensor_address(tensor_name, tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        logits[name] = tensors_in["logits"][0].clone()
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
