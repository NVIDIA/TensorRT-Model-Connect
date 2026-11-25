# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 self-quantization of the LLaVA decoder: recipe, scales, rejected configs, tiny engines."""

import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from families.llava import calibrate_scales, quantization
from families.llava.config import ModelConfig
from families.llava.model import build
from families.llava.quantization import (
    ACTIVATION_SCALES_FILENAME,
    PROJECTIONS,
    LlavaQuantContext,
    decoder_quantization,
    final_norm_head,
    quantize_fp8_weight,
    scale_key,
)
from tensorrt_model_connect import BuildRequest

SCALES = json.loads((Path(quantization.__file__).parent / ACTIVATION_SCALES_FILENAME).read_text())
KEY = "llava-h4096-l32-i11008"
STEMS = tuple(stem for _, stem in PROJECTIONS)
LLAVA_7B = ModelConfig(
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    vocab_size=32064,
)


def _weights(norm=None):
    """A final norm weight; by default with the calibrated checkpoint's first values."""
    if norm is None:
        norm = np.ones(4096, np.float16)
        norm[:8] = SCALES[KEY]["final_norm_head"]
    return {"final_norm": norm}


def test_fp8_weight_scale_is_amax_over_448_and_round_trips():
    import ml_dtypes

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
    assert list(SCALES) == [KEY] == [scale_key(LLAVA_7B)]
    head = SCALES[KEY]["final_norm_head"]
    assert len(head) == 8 and all(float(np.float32(value)) == value for value in head)
    entry = SCALES[KEY]["scales"]
    assert set(entry) == {f"layer.{layer}.{stem}" for layer in range(32) for stem in STEMS}
    values = [value for value in entry.values() if value is not None]
    assert all(np.isfinite(value) and value > 0 for value in values)
    # The down projection of layer 1 sees the massive activations (27.5x the median).
    assert [name for name, value in entry.items() if value is None] == ["layer.1.w_down"]


def test_outlier_rule_drops_only_scales_above_20x_the_median_of_their_kind():
    scales = {f"layer.{layer}.{stem}": 1.0 for layer in range(4) for stem in STEMS}
    scales["layer.3.w_down"] = 21.0
    scales["layer.2.w_o"] = 19.0
    kept = calibrate_scales.drop_outliers(scales)
    assert kept["layer.3.w_down"] is None
    assert kept["layer.2.w_o"] == 19.0
    assert sum(value is None for value in kept.values()) == 1


def test_calibration_names_match_both_module_layouts_and_skip_the_vision_tower():
    for name in (
        "language_model.model.layers.3.mlp.down_proj",
        "model.language_model.layers.3.mlp.down_proj",
    ):
        assert calibrate_scales.projection_name(name) == "layer.3.w_down"
    assert (
        calibrate_scales.projection_name("model.language_model.layers.12.self_attn.q_proj")
        == "layer.12.w_q"
    )
    for name in (
        "model.vision_tower.vision_model.encoder.layers.3.mlp.fc1",
        "model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj",
        "model.multi_modal_projector.linear_1",
        "model.language_model.layers.3.mlp.down_proj.input_quantizer",
        "lm_head",
    ):
        assert calibrate_scales.projection_name(name) is None


def test_fp8_recipe_quantizes_all_seven_projections_except_the_outlier():
    recipe = decoder_quantization(LLAVA_7B, _weights(), "fp8")
    expected = {name: value for name, value in SCALES[KEY]["scales"].items() if value is not None}
    assert recipe.input_scales == expected and len(expected) == 7 * 32 - 1
    assert "layer.1.w_down" not in recipe.input_scales and "w_out" not in recipe.input_scales
    for stem in STEMS:
        kind = [value for name, value in recipe.input_scales.items() if name.endswith("." + stem)]
        assert max(kind) < 20 * statistics.median(kind)


def test_missing_or_invalid_scale_entries_are_rejected(monkeypatch):
    for broken, error in (
        (lambda entry: entry.pop("layer.3.w_up"), "no entry"),
        (lambda entry: entry.update({"layer.3.w_up": 0.0}), "positive"),
        (lambda entry: entry.update({"layer.3.w_up": float("nan")}), "positive"),
        (lambda entry: entry.update({"layer.3.w_up": True}), "positive"),
    ):
        entry = dict(SCALES[KEY]["scales"])
        broken(entry)
        monkeypatch.setattr(quantization, "load_activation_scales", lambda config, weights: entry)
        with pytest.raises(ValueError, match=error):
            decoder_quantization(LLAVA_7B, _weights(), "fp8")


def test_other_configs_and_other_checkpoints_are_rejected():
    other = ModelConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        vocab_size=32064,
    )
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(other, _weights(), "fp8")
    # Same decoder config, other weights (for example a fine-tune).
    norm = _weights()["final_norm"].copy()
    norm[3] += np.float16(1e-2)
    assert final_norm_head({"final_norm": norm}) != SCALES[KEY]["final_norm_head"]
    with pytest.raises(NotImplementedError, match="calibrate_scales"):
        decoder_quantization(LLAVA_7B, _weights(norm), "fp8")
    for unsupported in ("nvfp4", "int8"):
        with pytest.raises(ValueError, match="quantization"):
            decoder_quantization(LLAVA_7B, _weights(), unsupported)


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
    LlavaQuantContext({}).configure(config)
    assert config.build_route == route
    assert config.profiling_verbosity == trt.ProfilingVerbosity.DETAILED


@pytest.mark.parametrize(
    "change, error",
    [
        ({"quantization": "int8"}, "quantization"),
        ({"quantization": "nvfp4"}, "quantization"),
        ({"quantization": "fp8", "precision": "bf16"}, "FP16"),
    ],
)
def test_build_rejects_unsupported_quantization_requests(tmp_path, change, error):
    request = BuildRequest(
        **{
            "model_dir": tmp_path,
            "output_path": tmp_path / "x.bundle",
            "family": "llava",
            "task": "vision_language_generation",
            "precision": "fp16",
            **change,
        }
    )
    with pytest.raises(ValueError, match=error):
        build(request, object())


def test_build_rejects_an_already_quantized_checkpoint(tmp_path):
    from safetensors.numpy import save_file

    config = {
        "model_type": "llava",
        "text_config": {"model_type": "llama"},
        "vision_config": {"image_size": 336, "patch_size": 14},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file(
        {"language_model.model.layers.0.mlp.up_proj.weight_scale": np.ones(1, np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "x.bundle",
        family="llava",
        task="vision_language_generation",
        precision="fp16",
        quantization="fp8",
        max_sequence_length=1024,
    )
    with pytest.raises(ValueError, match="already quantized"):
        build(request, object())


def _tiny_decoder():
    rng = np.random.default_rng(0)
    hidden, mlp, layers, vocab = 64, 128, 2, 128
    config = ModelConfig(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=vocab,
        raw={},
    )

    def matrix(rows, columns):
        return (rng.standard_normal((rows, columns)) / np.sqrt(rows)).astype(np.float16)

    weights = {
        "embedding": matrix(vocab, hidden) * 8,
        "final_norm": np.ones(hidden, np.float16),
        "w_out": matrix(hidden, vocab),
        "_attention_size": hidden,
        "_mlp_size": mlp,
        "_explicit_attention": True,
    }
    for layer in range(layers):
        prefix = f"layer.{layer}"
        weights[f"{prefix}.input_norm"] = np.ones(hidden, np.float16)
        weights[f"{prefix}.post_attn_norm"] = np.ones(hidden, np.float16)
        for stem in ("w_q", "w_k", "w_v", "w_o"):
            weights[f"{prefix}.{stem}"] = matrix(hidden, hidden)
        weights[f"{prefix}.w_gate"] = matrix(hidden, mlp)
        weights[f"{prefix}.w_up"] = matrix(hidden, mlp)
        weights[f"{prefix}.w_down"] = matrix(mlp, hidden)
    return config, weights


def test_tiny_fp8_prefill_and_decode_engines_have_fp8_weights_and_track_fp16(monkeypatch):
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT build requires CUDA")
    from families.llava.default_decoder import build_standard_decoder_engine
    from families.llava.default_dual_profile_decoder import build_dual_profile_decoder_engine

    config, weights = _tiny_decoder()
    scales = {f"layer.{layer}.{stem}": 0.02 for layer in range(2) for stem in STEMS}
    scales["layer.1.w_down"] = None
    monkeypatch.setattr(quantization, "load_activation_scales", lambda config, weights: scales)
    recipe = decoder_quantization(config, weights, "fp8")
    assert len(recipe.input_scales) == 13
    capacity, hidden = 16, config.hidden_size
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    def run(plan, inputs):
        engine = runtime.deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        outputs = {}
        for name, tensor in inputs.items():
            context.set_input_shape(name, tuple(tensor.shape))
            context.set_tensor_address(name, tensor.data_ptr())
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                dtype = torch.float32 if name == "logits" else torch.float16
                outputs[name] = torch.empty(
                    tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda"
                )
                context.set_tensor_address(name, outputs[name].data_ptr())
        assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        info = (
            engine.create_engine_inspector()
            .get_engine_information(trt.LayerInformationFormat.JSON)
            .lower()
        )
        return outputs["logits"].reshape(-1, config.vocab_size)[-1].clone(), info

    rows = 3
    mask = torch.full((rows, 1 + rows), -1e4, device="cuda")
    mask[:, 1:] = torch.triu(torch.full((rows, rows), -1e4, device="cuda"), diagonal=1)
    caches = {
        f"cache_{kind}_{layer}": torch.zeros(1, hidden, dtype=torch.float16, device="cuda")
        for layer in range(2)
        for kind in ("k", "v")
    }
    prefill_inputs = {
        "token_id": torch.tensor([5, 17, 42], dtype=torch.int32, device="cuda"),
        "position_id": torch.arange(rows, dtype=torch.int32, device="cuda"),
        "attention_mask": mask,
        "input_embed": torch.zeros(rows, hidden, device="cuda"),
        "use_input_embed": torch.zeros(rows, 1, device="cuda"),
        **caches,
    }
    decode_mask = torch.zeros(1, 2, device="cuda")
    decode_mask[0, 0] = -1e4
    decode_inputs = {
        "token_id": torch.tensor([42], dtype=torch.int32, device="cuda"),
        "position_id": torch.tensor([0], dtype=torch.int32, device="cuda"),
        "attention_mask": decode_mask,
        "input_embed": torch.zeros(1, hidden, device="cuda"),
        "use_input_embed": torch.zeros(1, device="cuda"),
        **caches,
    }
    logits = {}
    for name, quant in (("fp16", None), ("fp8", recipe)):
        config.raw.pop("_decoder_engine_role", None)
        prefill = build_dual_profile_decoder_engine(
            config,
            weights,
            capacity,
            precision="fp16",
            embed_input=True,
            quant_ctx=quant,
            opt_prefill_length=4,
            profile_mode="prefill",
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = build_standard_decoder_engine(
            config, weights, capacity, precision="fp16", embed_input=True, quant_ctx=quant
        )
        for role, plan, inputs in (
            ("prefill", prefill, prefill_inputs),
            ("decode", decode, decode_inputs),
        ):
            logits[name, role], info = run(plan, inputs)
            assert ("fp8" in info or "e4m3" in info) == (name == "fp8"), (name, role)
    for role in ("prefill", "decode"):
        assert torch.isfinite(logits["fp8", role]).all()
        cosine = torch.nn.functional.cosine_similarity(
            logits["fp8", role], logits["fp16", role], dim=0
        )
        assert cosine > 0.98, (role, float(cosine))
