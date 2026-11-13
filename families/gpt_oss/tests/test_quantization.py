# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS NVFP4: MXFP4 repack, expert reading, scales file, tiny engine build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from .. import calibrate_nvfp4_scales, graph_ops, model  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..quantization import (  # noqa: E402
    ACTIVATION_SCALES_FILENAME,
    GptOssQuantContext,
    calibrate_gpt_oss,
    dequantize_mxfp4,
    dequantize_nvfp4,
    load_activation_scales,
    repack_mxfp4_to_nvfp4,
    scale_key,
)

VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 32, 256, 2, 4, 2, 64
EXPERTS, TOP_K, INTER = 4, 2, 128
ATTENTION, KV = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM


def _mxfp4(gen, experts: int, out_features: int, in_features: int, exp_low=118, exp_high=128):
    blocks = torch.randint(
        0, 256, (experts, out_features, in_features // 32, 16), generator=gen, dtype=torch.uint8
    )
    scales = torch.randint(
        exp_low, exp_high, (experts, out_features, in_features // 32), generator=gen
    ).to(torch.uint8)
    return blocks, scales


def _config_json() -> dict:
    return {
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": INTER,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 8,
    }


def _write_checkpoint(path: Path, *, mxfp4: bool = True) -> dict[str, torch.Tensor]:
    """Write a tiny GPT-OSS checkpoint in the official layout (MXFP4 experts)."""
    gen = torch.Generator().manual_seed(0)
    bf16 = lambda *shape: (0.05 * torch.randn(*shape, generator=gen)).bfloat16()  # noqa: E731
    tensors = {
        "model.embed_tokens.weight": bf16(VOCAB, HIDDEN),
        "model.norm.weight": torch.ones(HIDDEN).bfloat16(),
        "lm_head.weight": bf16(VOCAB, HIDDEN),
    }
    for layer in range(LAYERS):
        base = f"model.layers.{layer}"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(HIDDEN).bfloat16()
        for stem, (out_f, in_f) in {
            "q_proj": (ATTENTION, HIDDEN),
            "k_proj": (KV, HIDDEN),
            "v_proj": (KV, HIDDEN),
            "o_proj": (HIDDEN, ATTENTION),
        }.items():
            tensors[f"{base}.self_attn.{stem}.weight"] = bf16(out_f, in_f)
            tensors[f"{base}.self_attn.{stem}.bias"] = bf16(out_f)
        tensors[f"{base}.self_attn.sinks"] = bf16(HEADS)
        tensors[f"{base}.mlp.router.weight"] = bf16(EXPERTS, HIDDEN)
        tensors[f"{base}.mlp.router.bias"] = bf16(EXPERTS)
        tensors[f"{base}.mlp.experts.gate_up_proj_bias"] = bf16(EXPERTS, 2 * INTER)
        tensors[f"{base}.mlp.experts.down_proj_bias"] = bf16(EXPERTS, HIDDEN)
        if mxfp4:
            for name, (out_f, in_f) in {
                "gate_up_proj": (2 * INTER, HIDDEN),
                "down_proj": (HIDDEN, INTER),
            }.items():
                blocks, scales = _mxfp4(gen, EXPERTS, out_f, in_f)
                tensors[f"{base}.mlp.experts.{name}_blocks"] = blocks
                tensors[f"{base}.mlp.experts.{name}_scales"] = scales
        else:
            tensors[f"{base}.mlp.experts.gate_up_proj"] = bf16(EXPERTS, HIDDEN, 2 * INTER)
            tensors[f"{base}.mlp.experts.down_proj"] = bf16(EXPERTS, INTER, HIDDEN)
    path.mkdir(parents=True, exist_ok=True)
    safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config_json()))
    return tensors


@pytest.fixture
def tiny_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the scale loader at a table for the tiny model shape."""
    config = ModelConfig.from_json(json.dumps(_config_json()))
    table = {f"layer.{i}.experts.{p}": 0.01 for i in range(LAYERS) for p in ("gate_up", "down")}
    monkeypatch.setattr(
        "families.gpt_oss.quantization.load_activation_scales", lambda _config: table
    )
    assert scale_key(config) == "gpt_oss-h256-l2-i128"


def test_repack_keeps_every_weight_value() -> None:
    gen = torch.Generator().manual_seed(1)
    blocks, scales = _mxfp4(gen, 3, 8, 64, exp_low=110, exp_high=127)
    packed, block_scale, global_scale, flushed = repack_mxfp4_to_nvfp4(blocks, scales)

    assert flushed == 0
    assert packed.shape == (3, 8, 32) and packed.dtype == np.uint8
    assert block_scale.shape == (3, 8, 4) and block_scale.dtype == np.uint8
    from ..quantization import _NVFP4Experts

    restored = dequantize_nvfp4(_NVFP4Experts(packed, block_scale, global_scale, 1.0))
    assert torch.equal(restored, dequantize_mxfp4(blocks, scales))  # lossless
    assert np.array_equal(packed, blocks.reshape(3, 8, 32).numpy())  # E2M1 bytes untouched


def test_repack_flushes_scales_outside_the_e4m3_window() -> None:
    gen = torch.Generator().manual_seed(2)
    blocks, scales = _mxfp4(gen, 1, 4, 64, exp_low=127, exp_high=128)
    scales[0, 0, 0] = 127 - 20  # 2^-20 against 2^0: more than 17 powers of two apart
    _, block_scale, _, flushed = repack_mxfp4_to_nvfp4(blocks, scales)
    assert flushed == 1
    assert not block_scale[0, 0, :2].any()  # both 16-value halves of the block are zero


def test_repack_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="uint8"):
        repack_mxfp4_to_nvfp4(torch.zeros(1, 2, 2, 16), torch.zeros(1, 2, 2))


def test_read_experts_deinterleaves_gate_up_bias(tmp_path: Path, tiny_scales) -> None:
    tensors = _write_checkpoint(tmp_path)
    ctx = calibrate_gpt_oss(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "nvfp4")

    assert set(ctx.experts) == {f"layer.{i}" for i in range(LAYERS)}
    layer = ctx.experts["layer.1"]
    assert layer.gate_up.packed.shape == (EXPERTS, 2 * INTER, HIDDEN // 2)
    assert layer.down.packed.shape == (EXPERTS, HIDDEN, INTER // 2)
    assert layer.gate_up.block_scale.shape == (EXPERTS, 2 * INTER, HIDDEN // 16)
    assert layer.flushed_blocks == 0
    bias = tensors["model.layers.1.mlp.experts.gate_up_proj_bias"].half().numpy()
    assert np.array_equal(layer.gate_bias, bias[:, ::2])
    assert np.array_equal(layer.up_bias, bias[:, 1::2])
    assert layer.gate_up.input_scale == pytest.approx(0.01)
    assert ctx.should_quantize("layer.0.experts") and not ctx.should_quantize("layer.9.experts")
    assert not ctx.should_quantize("layer.0.w_q")


def test_calibrate_rejects_dequantized_checkpoint(tmp_path: Path, tiny_scales) -> None:
    _write_checkpoint(tmp_path, mxfp4=False)
    with pytest.raises(ValueError, match="not an MXFP4 tensor"):
        calibrate_gpt_oss(tmp_path, ModelConfig.from_dir(tmp_path), graph_ops, "nvfp4")


def test_calibrate_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        calibrate_gpt_oss(tmp_path, None, graph_ops, "fp8")


def test_scale_file_covers_gpt_oss_20b_and_unknown_shape_raises() -> None:
    path = Path(__file__).resolve().parents[1] / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8"))
    entry = table["gpt_oss-h2880-l24-i2880"]
    assert len(entry) == 2 * 24 and all(value > 0 for value in entry.values())
    config = ModelConfig.from_json(json.dumps({**_config_json(), "hidden_size": 96}))
    with pytest.raises(NotImplementedError, match="calibrate_nvfp4_scales"):
        load_activation_scales(config)


def test_calibration_script_cli_arguments() -> None:
    assert callable(calibrate_nvfp4_scales.main) and callable(calibrate_nvfp4_scales.record_amax)


def test_load_weights_skips_the_experts(tmp_path: Path, tiny_scales) -> None:
    _write_checkpoint(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    ctx = calibrate_gpt_oss(tmp_path, config, graph_ops, "nvfp4")
    weights = model._GptOssModel().load_weights(
        str(tmp_path), config, precision="fp16", quant_ctx=ctx
    )

    assert weights["layer.0.w_q"].shape == (HIDDEN, ATTENTION)
    assert weights["layer.0.w_q"].dtype == np.float16
    assert weights["layer.0.router"].shape == (HIDDEN, EXPERTS)
    assert weights["layer.0.sinks"].shape == (HEADS,)
    assert weights["_attention_size"] == ATTENTION
    assert weights["_num_experts"] == EXPERTS and weights["_moe_intermediate_size"] == INTER
    assert not any(".expert." in key for key in weights)


@pytest.mark.gpu
@pytest.mark.trt
def test_engine_builds_with_packed_experts(tmp_path: Path, tiny_scales) -> None:
    """The quantized engine builds and stores the experts in 4 bits, not 16."""
    _write_checkpoint(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    ctx = calibrate_gpt_oss(tmp_path, config, graph_ops, "nvfp4")
    assert isinstance(ctx, GptOssQuantContext)
    instance = model._GptOssModel()
    weights = instance.load_weights(str(tmp_path), config, precision="fp16", quant_ctx=ctx)
    plan = instance.build_engine(config, weights, 4, precision="fp16", quant_ctx=ctx)

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
    assert {"token_id", "logits", "present_k_0"} <= names
    # Experts at 16 bits would take 3 matrices x E x I x K x 2 bytes per layer.
    fp16_expert_bytes = LAYERS * 3 * EXPERTS * INTER * HIDDEN * 2
    assert len(bytes(plan)) < 0.8 * fp16_expert_bytes + 1_000_000
