# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of the Qwen2.5 Omni Thinker: recipe, scales, rejected configs, tiny engines."""

import io
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from families.qwen2_5_omni import calibrate_scales, quantization
from families.qwen2_5_omni.model import build
from families.qwen2_5_omni.quantization import (
    ACTIVATION_SCALES_FILENAME,
    PROJECTIONS,
    ThinkerQuantContext,
    final_norm_head,
    quantize_fp8_weight,
    scale_key,
    thinker_quantization,
)
from tensorrt_model_connect import BuildRequest

SCALES = json.loads((Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text())
KEY = "qwen2_5_omni-h3584-l28-i18944"
OMNI_7B = {"hidden_size": 3584, "num_hidden_layers": 28, "intermediate_size": 18944}


def _weights(norm=None):
    """A final norm weight (by default with the calibrated checkpoint's first values) and small
    q, k, v weights whose amax is 1, 3 and 2."""
    if norm is None:
        norm = np.ones(3584, np.float32)
        norm[:8] = SCALES[KEY]["final_norm_head"]
    weights = {"model.norm.weight": norm}
    for layer in range(28):
        for stem, amax in (("q", 1.0), ("k", 3.0), ("v", 2.0)):
            weights[f"model.layers.{layer}.self_attn.{stem}_proj.weight"] = np.full(
                (4, 16), -amax, np.float32
            )
    return weights


def _decode_fp8(packed):
    import torch

    return torch.from_numpy(packed).view(torch.float8_e4m3fn).float().numpy()


def test_fp8_weight_scale_is_amax_over_448_and_round_trips():
    weight = np.random.default_rng(0).standard_normal((32, 48)).astype(np.float32)
    packed, scale = quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(float(np.abs(weight).max()) / 448)
    restored = _decode_fp8(packed) * scale
    # e4m3 keeps 3 mantissa bits: relative error at most 2**-4 above the subnormal range.
    np.testing.assert_allclose(restored, weight, rtol=2**-4, atol=scale * 2**-9)
    assert np.abs(restored).max() == pytest.approx(np.abs(weight).max())


def test_fp8_weight_without_a_finite_amax_is_rejected():
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_weight(np.zeros((4, 16), np.float32))


def test_fp8_weight_with_a_shared_amax_uses_the_shared_scale():
    weight = np.full((2, 16), 0.5, np.float32)
    packed, scale = quantize_fp8_weight(weight, amax=2.0)
    assert scale == pytest.approx(2.0 / 448)
    # 0.5 / (2 / 448) = 112 is exact in e4m3.
    assert np.all(_decode_fp8(packed) == 112)


def test_scale_file_covers_every_projection_of_the_published_model():
    assert list(SCALES) == [KEY] == [scale_key(OMNI_7B)]
    head = SCALES[KEY]["final_norm_head"]
    assert len(head) == 8 and all(float(np.float32(value)) == value for value in head)
    entry = SCALES[KEY]["scales"]
    assert set(entry) == {
        f"model.layers.{layer}.{stem}" for layer in range(28) for stem in PROJECTIONS
    }
    values = [value for value in entry.values() if value is not None]
    assert all(np.isfinite(value) and value > 0 for value in values)
    # The down projection of layer 26 sees the massive activations (56x the median).
    assert [name for name, value in entry.items() if value is None] == [
        "model.layers.26.mlp.down_proj"
    ]


def test_outlier_rule_drops_only_scales_above_20x_the_median_of_their_kind():
    scales = {f"model.layers.{layer}.{stem}": 1.0 for layer in range(4) for stem in PROJECTIONS}
    scales["model.layers.3.mlp.down_proj"] = 21.0
    scales["model.layers.2.self_attn.o_proj"] = 19.0
    kept = calibrate_scales.drop_outliers(scales)
    assert kept["model.layers.3.mlp.down_proj"] is None
    assert kept["model.layers.2.self_attn.o_proj"] == 19.0
    assert sum(value is None for value in kept.values()) == 1


def test_calibration_names_match_the_thinker_decoder_only():
    for name in ("model.layers.3.mlp.down_proj", "thinker.model.layers.3.mlp.down_proj"):
        assert calibrate_scales.projection_name(name) == "model.layers.3.mlp.down_proj"
    assert (
        calibrate_scales.projection_name("model.layers.12.self_attn.q_proj")
        == "model.layers.12.self_attn.q_proj"
    )
    for name in (
        "audio_tower.layers.3.self_attn.q_proj",
        "visual.blocks.3.mlp.gate_proj",
        "talker.model.layers.3.mlp.down_proj",
        "model.layers.3.mlp.down_proj.input_quantizer",
        "lm_head",
    ):
        assert calibrate_scales.projection_name(name) is None


def test_fp8_recipe_quantizes_all_seven_projections_except_the_outlier():
    recipe = thinker_quantization(OMNI_7B, _weights(), "fp8")
    expected = {name: value for name, value in SCALES[KEY]["scales"].items() if value is not None}
    assert recipe.input_scales == expected and len(expected) == 7 * 28 - 1
    assert not recipe.covers("model.layers.26.mlp.down_proj") and not recipe.covers("lm_head")
    # q, k and v share the largest weight amax of their layer; the others use their own.
    assert recipe.weight_amax == {
        f"model.layers.{layer}.self_attn.{stem}_proj": 3.0
        for layer in range(28)
        for stem in ("q", "k", "v")
    }
    for stem in PROJECTIONS:
        kind = [value for name, value in recipe.input_scales.items() if name.endswith("." + stem)]
        assert max(kind) < 20 * statistics.median(kind)


def test_missing_or_invalid_scale_entries_are_rejected(monkeypatch):
    name = "model.layers.3.mlp.up_proj"
    for broken, error in (
        (lambda entry: entry.pop(name), "no entry"),
        (lambda entry: entry.update({name: 0.0}), "positive"),
        (lambda entry: entry.update({name: float("nan")}), "positive"),
        (lambda entry: entry.update({name: True}), "positive"),
    ):
        entry = dict(SCALES[KEY]["scales"])
        broken(entry)
        monkeypatch.setattr(quantization, "load_activation_scales", lambda cfg, weights: entry)
        with pytest.raises(ValueError, match=error):
            thinker_quantization(OMNI_7B, _weights(), "fp8")


def test_other_configs_and_other_checkpoints_are_rejected():
    other = {"hidden_size": 2048, "num_hidden_layers": 36, "intermediate_size": 11008}
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        thinker_quantization(other, _weights(), "fp8")
    # Same decoder config, other weights (for example a fine-tune).
    norm = _weights()["model.norm.weight"].copy()
    norm[3] += 1e-2
    assert final_norm_head({"model.norm.weight": norm}) != SCALES[KEY]["final_norm_head"]
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        thinker_quantization(OMNI_7B, _weights(norm), "fp8")
    for unsupported in ("nvfp4", "int8"):
        with pytest.raises(ValueError, match="quantization"):
            thinker_quantization(OMNI_7B, _weights(), unsupported)


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
    ThinkerQuantContext({"model.layers.0.mlp.up_proj": 0.1}).configure(config)
    assert config.build_route == route
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


def _request(tmp_path, **change):
    return BuildRequest(
        **{
            "model_dir": tmp_path,
            "output_path": tmp_path / "x.bundle",
            "family": "qwen2_5_omni",
            "task": "text_continuation",
            "precision": "fp32",
            **change,
        }
    )


@pytest.mark.parametrize(
    "change, error",
    [
        ({"quantization": "int8"}, "quantization"),
        ({"quantization": "nvfp4"}, "quantization"),
        ({"quantization": "fp8", "precision": "fp16"}, "fp32"),
    ],
)
def test_build_rejects_unsupported_quantization_requests(tmp_path, change, error):
    with pytest.raises((ValueError, NotImplementedError), match=error):
        build(_request(tmp_path, **change), object())


@pytest.mark.parametrize(
    "config_extra, tensor",
    [
        (
            {"quantization_config": {"quant_method": "awq"}},
            "thinker.model.layers.0.mlp.up_proj.qweight",
        ),
        ({}, "thinker.model.layers.0.mlp.up_proj.weight_scale"),
    ],
)
def test_build_rejects_an_already_quantized_checkpoint(tmp_path, config_extra, tensor):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_omni", **config_extra})
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor: "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="already quantized"):
        build(_request(tmp_path, quantization="fp8", max_sequence_length=16), object())


def _tiny_thinker():
    from transformers import Qwen2_5OmniTextConfig

    hidden, mlp, layers, vocab, kv, dim = 256, 512, 2, 1024, 2, 64
    cfg = Qwen2_5OmniTextConfig(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=kv,
        vocab_size=vocab,
        rope_theta=1000000.0,
        # The multimodal rotary sections must cover head_dim / 2 = 32 frequencies.
        rope_scaling={"type": "default", "rope_type": "default", "mrope_section": [8, 12, 12]},
    ).to_dict()
    cfg["head_dim"] = dim
    rng = np.random.default_rng(0)

    def matrix(rows, columns):
        return (rng.standard_normal((rows, columns)) / np.sqrt(columns)).astype(np.float32)

    weights = {
        "model.embed_tokens.weight": matrix(vocab, hidden) * 8,
        "model.norm.weight": np.ones(hidden, np.float32),
        "lm_head.weight": matrix(vocab, hidden),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = np.ones(hidden, np.float32)
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hidden, np.float32)
        for stem, rows in (("q", hidden), ("k", kv * dim), ("v", kv * dim)):
            weights[f"{prefix}.self_attn.{stem}_proj.weight"] = matrix(rows, hidden)
            weights[f"{prefix}.self_attn.{stem}_proj.bias"] = rng.standard_normal(rows).astype(
                np.float32
            )
        weights[f"{prefix}.self_attn.o_proj.weight"] = matrix(hidden, hidden)
        weights[f"{prefix}.mlp.gate_proj.weight"] = matrix(mlp, hidden)
        weights[f"{prefix}.mlp.up_proj.weight"] = matrix(mlp, hidden)
        weights[f"{prefix}.mlp.down_proj.weight"] = matrix(hidden, mlp)
    return cfg, weights


def test_tiny_fp8_engine_has_fp8_weights_and_tracks_fp32(monkeypatch):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT build requires CUDA")
    from families.qwen2_5_omni.model import build_engine

    cfg, weights = _tiny_thinker()
    scales = {f"model.layers.{layer}.{stem}": 0.03 for layer in range(2) for stem in PROJECTIONS}
    scales["model.layers.1.mlp.down_proj"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda cfg, weights: scales)
    recipe = thinker_quantization(cfg, weights, "fp8")
    assert len(recipe.input_scales) == 13 and len(recipe.weight_amax) == 6
    length, kv, dim = 16, cfg["num_key_value_heads"], cfg["head_dim"]
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    logits = {}
    for name, quant in (("fp32", None), ("fp8", recipe)):
        stream = io.BytesIO()
        build_engine(weights, cfg, length, stream, quant=quant)
        engine = runtime.deserialize_cuda_engine(stream.getvalue())
        context = engine.create_execution_context()
        info = (
            engine.create_engine_inspector()
            .get_engine_information(trt.LayerInformationFormat.JSON)
            .lower()
        )
        assert ("fp8" in info or "e4m3" in info) == (name == "fp8"), name
        caches = [torch.zeros(1, kv, length, dim, device="cuda") for _ in range(4)]
        presents = [torch.zeros(1, kv, length, dim, device="cuda") for _ in range(4)]
        token = torch.zeros(1, dtype=torch.int32, device="cuda")
        position = torch.zeros(1, dtype=torch.int32, device="cuda")
        output = torch.zeros(1, cfg["vocab_size"], device="cuda")
        tensors = {"input_ids": token, "position": position, "logits": output}
        for layer in range(2):
            tensors[f"cache_k_{layer}"], tensors[f"cache_v_{layer}"] = caches[
                2 * layer : 2 * layer + 2
            ]
            tensors[f"present_k_{layer}"], tensors[f"present_v_{layer}"] = presents[
                2 * layer : 2 * layer + 2
            ]
        for tensor_name, tensor in tensors.items():
            context.set_tensor_address(tensor_name, tensor.data_ptr())
        rows = []
        for step, token_id in enumerate((5, 17, 42, 7)):
            token.fill_(token_id)
            position.fill_(step)
            assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            for cache, present in zip(caches, presents):
                cache.copy_(present)
            rows.append(output.clone())
        torch.cuda.synchronize()
        logits[name] = torch.cat(rows)
    assert torch.isfinite(logits["fp8"]).all()
    cosine = torch.nn.functional.cosine_similarity(logits["fp8"], logits["fp32"], dim=-1)
    assert cosine.min() > 0.98, cosine
