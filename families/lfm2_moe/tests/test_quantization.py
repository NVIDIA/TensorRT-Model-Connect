# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LFM2 MoE FP8 expert quantization: weights, scales, recipe, build contract and a tiny engine."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")

from .. import calibrate_fp8_scales, model, quantization  # noqa: E402
from ..quantization import (  # noqa: E402
    FP8_E4M3_MAX,
    calibrate_lfm2_moe_fp8,
    load_activation_scales,
    moe_layers,
    outlier_scales,
    quantize_fp8_stack,
    scale_key,
    scale_name,
)

SCALES = Path(quantization.__file__).parent / quantization.ACTIVATION_SCALES_FILENAME
MANIFEST = Path(__file__).parent / "manifests" / "lfm2.5-24b-a2b-fp8.json"


def _tiny_cfg() -> dict:
    """A small LFM2 MoE config: 1 dense layer, then conv / attention MoE layers."""
    from transformers import Lfm2MoeConfig

    cfg = Lfm2MoeConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        num_dense_layers=1,
        layer_types=["conv", "full_attention", "conv", "conv"],
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    ).to_dict()
    cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
    return cfg


def _tiny_weights(cfg: dict, seed: int = 0) -> dict:
    """Random fp32 weights in the layout of ``model.load_weights``."""
    rng = np.random.default_rng(seed)
    width, middle, experts = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["num_experts"]
    dense, kv = cfg["intermediate_size"], cfg["num_key_value_heads"] * cfg["head_dim"]

    def mat(*shape):
        return (rng.standard_normal(shape) / np.sqrt(shape[-1])).astype(np.float32)

    weights = {
        "model.embed_tokens.weight": mat(cfg["vocab_size"], width),
        "model.embedding_norm.weight": np.ones(width, np.float32),
    }
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    for layer, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{layer}"
        weights[p + ".operator_norm.weight"] = np.ones(width, np.float32)
        weights[p + ".ffn_norm.weight"] = np.ones(width, np.float32)
        if kind == "conv":
            weights[p + ".conv.in_proj.weight"] = mat(3 * width, width)
            weights[p + ".conv.out_proj.weight"] = mat(width, width)
            weights[p + ".conv.conv.weight"] = mat(width, 1, cfg["conv_L_cache"])
        else:
            weights[p + ".self_attn.q_proj.weight"] = mat(width, width)
            weights[p + ".self_attn.k_proj.weight"] = mat(kv, width)
            weights[p + ".self_attn.v_proj.weight"] = mat(kv, width)
            weights[p + ".self_attn.out_proj.weight"] = mat(width, width)
            weights[p + ".self_attn.q_layernorm.weight"] = np.ones(cfg["head_dim"], np.float32)
            weights[p + ".self_attn.k_layernorm.weight"] = np.ones(cfg["head_dim"], np.float32)
        if layer < cfg["num_dense_layers"]:
            weights[p + ".feed_forward.w1.weight"] = mat(dense, width)
            weights[p + ".feed_forward.w3.weight"] = mat(dense, width)
            weights[p + ".feed_forward.w2.weight"] = mat(width, dense)
        else:
            weights[p + ".feed_forward.gate.weight"] = mat(experts, width)
            weights[p + ".feed_forward.expert_bias"] = np.zeros(experts, np.float32)
            weights[p + ".feed_forward.experts.gate_up"] = mat(experts, 2 * middle, width)
            weights[p + ".feed_forward.experts.down"] = mat(experts, width, middle)
    return weights


def _tiny_scales(cfg: dict, weights: dict, tmp_path: Path, down_null=()) -> Path:
    scales = {}
    for layer in moe_layers(cfg):
        # Above the largest expert inputs of the random tiny model, so nothing is clipped.
        scales[scale_name(layer, "gate_up")] = 8.0 / FP8_E4M3_MAX
        scales[scale_name(layer, "down")] = None if layer in down_null else 32.0 / FP8_E4M3_MAX
    path = tmp_path / "scales.json"
    path.write_text(json.dumps({scale_key(cfg, weights["model.embedding_norm.weight"]): scales}))
    return path


def test_fp8_stack_scale_and_grid():
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((3, 32, 64)).astype(np.float32)
    grid, scale = quantize_fp8_stack(stack)
    assert scale == pytest.approx(np.abs(stack).max() / FP8_E4M3_MAX)
    assert grid.dtype == np.float16 and grid.shape == stack.shape
    assert np.abs(grid).max() == FP8_E4M3_MAX
    # Every value is an exact e4m3 value.
    exact = torch.from_numpy(grid.astype(np.float32)).to(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_array_equal(exact, grid.astype(np.float32))
    # Relative error of e4m3 rounding (3 mantissa bits) for normal values.
    big = np.abs(stack) > 2.0**-6 * scale
    error = np.abs(grid.astype(np.float32) * scale - stack)[big] / np.abs(stack)[big]
    assert error.max() <= 2.0**-4 + 1e-6


def test_fp8_stack_rejects_zero_weights():
    with pytest.raises(ValueError, match="amax"):
        quantize_fp8_stack(np.zeros((2, 4, 4), np.float32))


def test_outlier_rule_only_drops_large_down_inputs():
    scales = {scale_name(layer, "down"): 1.0 for layer in range(2, 12)}
    scales[scale_name(5, "down")] = 21.0
    scales[scale_name(6, "down")] = 19.0
    scales.update({scale_name(layer, "gate_up"): 100.0 for layer in range(2, 12)})
    result = outlier_scales(scales)
    assert [name for name, value in result.items() if value is None] == [scale_name(5, "down")]


def test_committed_scales_cover_the_manifest_model():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["quantization"] == "fp8" and manifest["precision"] == "fp32"
    (case,) = manifest["testcases"]
    assert case["premerge"] and 0 < case["near_tie_logit_tolerance"] < 3
    table = json.loads(SCALES.read_text())
    keys = [key for key in table if key.startswith("lfm2_moe-h2048-l40-e64-i1536-n")]
    assert len(keys) == 1
    scales = table[keys[0]]
    cfg = {"num_dense_layers": 2, "num_hidden_layers": 40}
    expected = {scale_name(layer, p) for layer in moe_layers(cfg) for p in ("gate_up", "down")}
    assert set(scales) == expected
    assert all(scales[scale_name(layer, "gate_up")] > 0 for layer in moe_layers(cfg))
    fp32_down = sorted(int(name.split(".")[1]) for name, value in scales.items() if value is None)
    assert fp32_down == [9, 21, 34, 37]
    assert outlier_scales({k: v for k, v in scales.items() if v is not None}) == {
        k: v for k, v in scales.items() if v is not None
    }


def test_unknown_checkpoint_and_incomplete_table_raise(tmp_path):
    cfg = _tiny_cfg()
    weights = _tiny_weights(cfg)
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales"):
        load_activation_scales(cfg, weights["model.embedding_norm.weight"])
    path = _tiny_scales(cfg, weights, tmp_path)
    table = json.loads(path.read_text())
    key = next(iter(table))
    del table[key][scale_name(2, "down")]
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="no layer.2.experts.down"):
        load_activation_scales(cfg, weights["model.embedding_norm.weight"], path)
    table[key][scale_name(2, "down")] = 0.1
    table[key][scale_name(2, "gate_up")] = None
    path.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="no scale for layer.2.experts.gate_up"):
        load_activation_scales(cfg, weights["model.embedding_norm.weight"], path)


def test_recipe_quantizes_only_the_routed_experts(tmp_path):
    cfg = _tiny_cfg()
    weights = _tiny_weights(cfg)
    before = set(weights)
    ctx = calibrate_lfm2_moe_fp8(cfg, weights, _tiny_scales(cfg, weights, tmp_path, down_null=(3,)))
    assert sorted(ctx.experts) == [1, 2, 3]
    assert ctx.experts[3]["down"] is None and ctx.experts[3]["gate_up"] is not None
    # Quantized stacks leave the fp32 dict; the fp32 down of layer 3 and everything else stay.
    removed = before - set(weights)
    assert removed == {
        "model.layers.1.feed_forward.experts.gate_up",
        "model.layers.1.feed_forward.experts.down",
        "model.layers.2.feed_forward.experts.gate_up",
        "model.layers.2.feed_forward.experts.down",
        "model.layers.3.feed_forward.experts.gate_up",
    }
    assert ctx.experts[1]["gate_up"].input_scale == pytest.approx(8.0 / FP8_E4M3_MAX)


def test_already_quantized_checkpoint_is_rejected(tmp_path):
    cfg = _tiny_cfg()
    weights = _tiny_weights(cfg)
    weights["model.layers.2.self_attn.q_proj.weight_scale"] = np.ones(1, np.float32)
    with pytest.raises(ValueError, match="already quantized"):
        calibrate_lfm2_moe_fp8(cfg, weights, _tiny_scales(cfg, weights, tmp_path))


@pytest.mark.parametrize("quantization", ["nvfp4", "int4_awq", "fp8_kv"])
def test_build_rejects_other_quantizations(quantization):
    request = SimpleNamespace(
        dynamic_kv_cache=False,
        task="text_continuation",
        backend="trt",
        precision="fp32",
        tensor_parallel_size=1,
        context_parallel_size=1,
        max_batch_size=1,
        quantization=quantization,
        fp32_layers=(),
        image_height=None,
        image_width=None,
        video_num_frames=None,
        mtp_seq_len=None,
    )
    with pytest.raises(NotImplementedError, match="quantization none or fp8"):
        model.build(request, writer=None)


def test_calibration_tool_samples(monkeypatch):
    assert [calibrate_fp8_scales.token_limit(i) for i in range(8)] == [512] * 7 + [1024]

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize):
            assert not tokenize
            return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    text = calibrate_fp8_scales.chat_text(Tokenizer(), "body", "short")
    assert text == "user:Summarize the following news article.\n\nbody|assistant:short"


class _Runner:
    """Drive a tiny decode engine one token at a time, like the native runtime."""

    def __init__(self, plan: bytes, cfg: dict, length: int):
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        kv, dim, width = cfg["num_key_value_heads"], cfg["head_dim"], cfg["hidden_size"]
        self.pairs, binds = [], {}
        for layer, kind in enumerate(cfg["layer_types"]):
            names = (
                [("conv", (1, width, cfg["conv_L_cache"] - 1))]
                if kind == "conv"
                else [
                    ("k", (1, kv, length, dim)),
                    ("v", (1, kv, length, dim)),
                ]
            )
            for name, shape in names:
                cache, present = (
                    torch.zeros(shape, device="cuda"),
                    torch.zeros(shape, device="cuda"),
                )
                binds[f"cache_{name}_{layer}"], binds[f"present_{name}_{layer}"] = cache, present
                self.pairs.append((cache, present))
        self.token = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.position = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.logits = torch.zeros(1, cfg["vocab_size"], device="cuda")
        binds.update(input_ids=self.token, position=self.position, logits=self.logits)
        for name, tensor in binds.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

    def run(self, tokens):
        stream = torch.cuda.current_stream()
        rows = []
        for position, token in enumerate(tokens):
            self.token.fill_(token)
            self.position.fill_(position)
            assert self.context.execute_async_v3(stream.cuda_stream)
            for cache, present in self.pairs:
                cache.copy_(present)
            rows.append(self.logits.clone())
        stream.synchronize()
        return torch.cat(rows)


def _plan(weights, cfg, quant) -> bytes:
    stream = io.BytesIO()
    model.build_engine(weights, cfg, 16, stream, quant=quant)
    return stream.getvalue()


def test_tiny_fp8_engine_matches_the_fp32_engine(tmp_path):
    if not torch.cuda.is_available():
        pytest.fail("the FP8 engine test needs a CUDA GPU")
    cfg = _tiny_cfg()
    fp32_plan = _plan(_tiny_weights(cfg), cfg, None)
    weights = _tiny_weights(cfg)
    rng = np.random.default_rng(3)
    tokens = rng.integers(0, cfg["vocab_size"], 12).tolist()
    reference = _Runner(fp32_plan, cfg, 16).run(tokens)
    ctx = calibrate_lfm2_moe_fp8(cfg, weights, _tiny_scales(cfg, weights, tmp_path, down_null=(3,)))
    fp8_plan = _plan(weights, cfg, ctx)
    logits = _Runner(fp8_plan, cfg, 16).run(tokens)
    assert torch.isfinite(logits).all()
    cosine = torch.nn.functional.cosine_similarity(logits, reference, dim=-1)
    assert float(cosine.min()) > 0.98
    # The experts are stored as FP8: the plan is much smaller than the fp32 one.
    assert len(fp8_plan) < 0.6 * len(fp32_plan)
    inspector = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(fp8_plan)
    info = inspector.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    assert "fp8" in info.lower() or "e4m3" in info.lower()
