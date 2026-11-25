# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of SeamlessM4T v2: recipe, scales, rejected configs, tiny engines."""

import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from families.seamless_m4t_v2 import calibrate_fp8_scales, quantization
from families.seamless_m4t_v2.model import build
from families.seamless_m4t_v2.quantization import (
    ACTIVATION_SCALES_FILENAME,
    DECODER_PROJECTIONS,
    ENCODER_PROJECTIONS,
    QuantContext,
    final_norm_head,
    fp8_quantization,
    projection_names,
    quantize_fp8_weight,
    scale_key,
)
from tensorrt_model_connect import BuildRequest

SCALES = json.loads((Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text())
KEY = "seamless_m4t_v2-h1024-e24-d24-f8192"
LARGE = {"hidden_size": 1024, "encoder_layers": 24, "decoder_layers": 24, "decoder_ffn_dim": 8192}
NORM = "text_decoder.layer_norm.weight"


def _weights(norm=None):
    """The final decoder norm (by default with the calibrated checkpoint's first values) and
    small attention weights whose amax is 1, 3 and 2 (q, k, v) and 4 and 5 (cross k, v)."""
    if norm is None:
        norm = np.ones(1024, np.float32)
        norm[:8] = SCALES[KEY]["final_norm_head"]
    weights = {NORM: norm}
    for part, layers in (("text_encoder", 24), ("text_decoder", 24)):
        for layer in range(layers):
            prefix = f"{part}.layers.{layer}"
            for stem, amax in (("q", 1.0), ("k", 3.0), ("v", 2.0)):
                weights[f"{prefix}.self_attn.{stem}_proj.weight"] = np.full(
                    (4, 16), -amax, np.float32
                )
            if part == "text_decoder":
                for stem, amax in (("k", 4.0), ("v", 5.0)):
                    weights[f"{prefix}.cross_attention.{stem}_proj.weight"] = np.full(
                        (4, 16), amax, np.float32
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
    packed, scale = quantize_fp8_weight(np.full((2, 16), 0.5, np.float32), amax=2.0)
    assert scale == pytest.approx(2.0 / 448)
    # 0.5 / (2 / 448) = 112 is exact in e4m3.
    assert np.all(_decode_fp8(packed) == 112)


def test_scale_file_covers_every_projection_of_the_published_model():
    assert list(SCALES) == [KEY] == [scale_key(LARGE)]
    head = SCALES[KEY]["final_norm_head"]
    assert len(head) == 8 and all(float(np.float32(value)) == value for value in head)
    entry = SCALES[KEY]["scales"]
    assert set(entry) == set(projection_names(LARGE))
    assert len(entry) == 24 * len(ENCODER_PROJECTIONS) + 24 * len(DECODER_PROJECTIONS) == 384
    # No projection of the calibrated checkpoint is an outlier (all below 20x the median).
    assert all(np.isfinite(value) and value > 0 for value in entry.values())


def test_outlier_rule_drops_only_scales_above_20x_the_median_of_their_kind():
    scales = {
        name: 1.0 for name in projection_names({**LARGE, "encoder_layers": 4, "decoder_layers": 4})
    }
    scales["text_decoder.layers.3.ffn.fc2"] = 21.0
    scales["text_encoder.layers.2.ffn.fc2"] = 19.0
    kept = calibrate_fp8_scales.drop_outliers(scales)
    assert kept["text_decoder.layers.3.ffn.fc2"] is None
    assert kept["text_encoder.layers.2.ffn.fc2"] == 19.0
    assert sum(value is None for value in kept.values()) == 1


def test_calibration_names_match_the_text_encoder_and_decoder_only():
    for name in (
        "text_decoder.layers.3.ffn.fc1",
        "text_encoder.layers.0.self_attn.out_proj",
        "text_decoder.layers.23.cross_attention.k_proj",
    ):
        assert calibrate_fp8_scales.projection_name(name) == name
        assert calibrate_fp8_scales.projection_name("model." + name) == name
    for name in (
        "speech_encoder.encoder.layers.3.ffn.intermediate_dense",
        "t2u_model.model.encoder.layers.0.self_attn.q_proj",
        "text_decoder.layers.3.ffn.fc1.input_quantizer",
        "text_decoder.embed_tokens",
        "lm_head",
    ):
        assert calibrate_fp8_scales.projection_name(name) is None


def test_fp8_recipe_quantizes_every_encoder_and_decoder_projection():
    recipe = fp8_quantization(LARGE, _weights(), "fp8")
    assert recipe.input_scales == SCALES[KEY]["scales"] and len(recipe.input_scales) == 384
    assert not recipe.covers("lm_head")
    # q, k and v (and cross-attention k and v) share the largest weight amax of their group.
    expected = {}
    for part in ("text_encoder", "text_decoder"):
        for layer in range(24):
            for stem in ("q", "k", "v"):
                expected[f"{part}.layers.{layer}.self_attn.{stem}_proj"] = 3.0
            if part == "text_decoder":
                for stem in ("k", "v"):
                    expected[f"{part}.layers.{layer}.cross_attention.{stem}_proj"] = 5.0
    assert recipe.weight_amax == expected
    for stem in DECODER_PROJECTIONS:
        for part in ("text_encoder", "text_decoder"):
            kind = [
                v for n, v in recipe.input_scales.items() if n.startswith(part) and n.endswith(stem)
            ]
            if kind:
                assert max(kind) < 20 * statistics.median(kind)


def test_a_null_scale_keeps_that_projection_fp32(monkeypatch):
    entry = dict(SCALES[KEY]["scales"])
    entry["text_decoder.layers.5.ffn.fc2"] = None
    entry["text_decoder.layers.7.self_attn.k_proj"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda cfg, weights: entry)
    recipe = fp8_quantization(LARGE, _weights(), "fp8")
    assert len(recipe.input_scales) == 382
    assert not recipe.covers("text_decoder.layers.5.ffn.fc2")
    # A group with an FP32 member keeps each member's own weight scale.
    assert "text_decoder.layers.7.self_attn.q_proj" not in recipe.weight_amax


def test_missing_or_invalid_scale_entries_are_rejected(monkeypatch):
    name = "text_decoder.layers.3.ffn.fc1"
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
            fp8_quantization(LARGE, _weights(), "fp8")


def test_other_configs_and_other_checkpoints_are_rejected():
    other = {**LARGE, "encoder_layers": 12, "decoder_layers": 12}
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        fp8_quantization(other, _weights(), "fp8")
    # Same config, other weights (for example a fine-tune).
    norm = _weights()[NORM].copy()
    norm[3] += 1e-2
    assert final_norm_head({NORM: norm}) != SCALES[KEY]["final_norm_head"]
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        fp8_quantization(LARGE, _weights(norm), "fp8")
    for unsupported in ("nvfp4", "int8"):
        with pytest.raises(ValueError, match="quantization"):
            fp8_quantization(LARGE, _weights(), unsupported)


def test_fp8_build_keeps_layer_names_in_the_plan():
    trt = pytest.importorskip("tensorrt")

    class Config:
        profiling_verbosity = None

    config = Config()
    QuantContext({"text_decoder.layers.0.ffn.fc1": 0.1}).configure(config)
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


def _request(tmp_path, **change):
    return BuildRequest(
        **{
            "model_dir": tmp_path,
            "output_path": tmp_path / "x.bundle",
            "family": "seamless_m4t_v2",
            "task": "text_translation",
            "precision": "fp32",
            **change,
        }
    )


@pytest.mark.parametrize(
    "change, error",
    [
        ({"quantization": "int8"}, "fp32 text translation"),
        ({"quantization": "nvfp4"}, "fp32 text translation"),
        ({"quantization": "fp8", "precision": "fp16"}, "fp32 text translation"),
    ],
)
def test_build_rejects_unsupported_quantization_requests(tmp_path, change, error):
    with pytest.raises(NotImplementedError, match=error):
        build(_request(tmp_path, **change), object())


@pytest.mark.parametrize(
    "config_extra, tensor",
    [
        (
            {"quantization_config": {"quant_method": "modelopt"}},
            "text_decoder.layers.0.ffn.fc1.weight",
        ),
        ({}, "text_decoder.layers.0.ffn.fc1.weight_scale"),
        ({}, "text_encoder.layers.0.self_attn.q_proj.input_scale"),
    ],
)
def test_build_rejects_an_already_quantized_checkpoint(tmp_path, config_extra, tensor):
    from transformers import SeamlessM4Tv2Config

    config = SeamlessM4Tv2Config().to_dict()
    config.update(config_extra)
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor: "model.safetensors"}})
    )
    with pytest.raises(ValueError, match="already quantized"):
        build(_request(tmp_path, quantization="fp8", max_sequence_length=16), object())


def _tiny_model():
    hidden, ffn, layers, vocab, heads = 64, 128, 2, 256, 4
    cfg = {
        "hidden_size": hidden,
        "encoder_layers": layers,
        "decoder_layers": layers,
        "encoder_attention_heads": heads,
        "decoder_attention_heads": heads,
        "decoder_ffn_dim": ffn,
        "scale_embedding": True,
        "pad_token_id": 0,
        "vocab_size": vocab,
    }
    rng = np.random.default_rng(0)

    def matrix(rows, columns):
        return (rng.standard_normal((rows, columns)) / np.sqrt(columns)).astype(np.float32)

    def vector(size, value=None):
        if value is not None:
            return np.full(size, value, np.float32)
        return (rng.standard_normal(size) * 0.1).astype(np.float32)

    weights = {"shared.weight": matrix(vocab, hidden) / 4, "lm_head.weight": matrix(vocab, hidden)}
    for part, stems in (
        ("text_encoder", ENCODER_PROJECTIONS),
        ("text_decoder", DECODER_PROJECTIONS),
    ):
        for layer in range(layers):
            prefix = f"{part}.layers.{layer}"
            for stem in stems:
                rows, columns = (
                    (ffn, hidden)
                    if stem == "ffn.fc1"
                    else (hidden, ffn)
                    if stem == "ffn.fc2"
                    else (hidden, hidden)
                )
                weights[f"{prefix}.{stem}.weight"] = matrix(rows, columns)
                weights[f"{prefix}.{stem}.bias"] = vector(rows)
            for norm in ("self_attn_layer_norm", "cross_attention_layer_norm", "ffn_layer_norm"):
                weights[f"{prefix}.{norm}.weight"] = vector(hidden, 1.0)
                weights[f"{prefix}.{norm}.bias"] = vector(hidden, 0.0)
        weights[f"{part}.layer_norm.weight"] = vector(hidden, 1.0)
        weights[f"{part}.layer_norm.bias"] = vector(hidden, 0.0)
    return cfg, weights


def test_tiny_fp8_engines_have_fp8_weights_and_track_fp32(monkeypatch):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT build requires CUDA")
    from families.seamless_m4t_v2.model import build_engine

    cfg, weights = _tiny_model()
    scales = {name: 0.02 for name in projection_names(cfg)}
    scales["text_decoder.layers.1.ffn.fc2"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda cfg, weights: scales)
    recipe = fp8_quantization(cfg, weights, "fp8")
    assert (
        len(recipe.input_scales) == 2 * 6 + 2 * 10 - 1 and len(recipe.weight_amax) == 2 * 3 + 2 * 5
    )
    length, width, vocab = 16, cfg["hidden_size"], cfg["vocab_size"]
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    source = [3, 17, 42, 9, 2] + [0] * (length - 5)
    target = [2, 5, 77, 13] + [0] * (length - 4)

    def positions(count):
        return [index + 1 if index < count else 0 for index in range(length)]

    def mask(count):
        return [1] * count + [0] * (length - count)

    logits = {}
    for name, quant in (("fp32", None), ("fp8", recipe)):
        engines = [
            runtime.deserialize_cuda_engine(
                build_engine(weights, cfg, length, decoder=d, quant=quant)
            )
            for d in (False, True)
        ]
        for engine in engines:
            info = (
                engine.create_engine_inspector()
                .get_engine_information(trt.LayerInformationFormat.JSON)
                .lower()
            )
            assert ("fp8" in info or "e4m3" in info) == (name == "fp8"), name

        def ints(values):
            return torch.tensor(values, dtype=torch.int32, device="cuda")

        memory = torch.zeros(length, width, device="cuda")
        encoder_inputs = {
            "input_ids": ints(source),
            "attention_mask": ints(mask(5)),
            "position_ids": ints(positions(5)),
            "encoder_hidden": memory,
        }
        context = engines[0].create_execution_context()
        for tensor_name, tensor in encoder_inputs.items():
            context.set_tensor_address(tensor_name, tensor.data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        last = ints([0])
        output = torch.zeros(1, vocab, device="cuda")
        decoder_inputs = {
            "input_ids": ints(target),
            "attention_mask": ints(mask(4)),
            "position_ids": ints(positions(4)),
            "encoder_hidden": memory,
            "encoder_mask": encoder_inputs["attention_mask"],
            "last_index": last,
            "logits": output,
        }
        context = engines[1].create_execution_context()
        for tensor_name, tensor in decoder_inputs.items():
            context.set_tensor_address(tensor_name, tensor.data_ptr())
        rows = []
        for row in range(4):
            last.fill_(row)
            assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            rows.append(output.clone())
        torch.cuda.synchronize()
        logits[name] = torch.cat(rows)
    assert torch.isfinite(logits["fp8"]).all()
    cosine = torch.nn.functional.cosine_similarity(logits["fp8"], logits["fp32"], dim=-1)
    assert cosine.min() > 0.98, cosine
