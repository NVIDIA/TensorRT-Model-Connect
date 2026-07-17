#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Wan2.2 block for source-isolated TensorRT parity analysis."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.families.wan2_2_ti2v import trt_ops as op
from tensorrt_model_connect.families.wan2_2_ti2v.dit_builder import (
    _numpy_state,
    _slice_chunks,
    _wan_rope,
)
from tensorrt_model_connect.families.wan2_2_ti2v.model_config import WAN22_TI2V_5B


trt = trt_compat.get_trt()


def _mark(network, tensor, name: str):
    output = network.add_identity(op.cast(network, tensor, trt.float32)).get_output(0)
    output.name = name
    network.mark_output(output)


def build_block(
    checkpoint: Path,
    block_index: int,
    *,
    self_attention_fp32: bool,
    cross_attention_fp32: bool,
    emulate_bf16_gemm: bool,
    debug_qkv: bool,
    source_attention_plugin: Path | None,
    cuda_bf16_plugin: Path | None,
    dit_cuda_plugin: Path | None,
) -> bytes:
    cfg = WAN22_TI2V_5B
    weights = _numpy_state(str(checkpoint))
    op.set_bf16_gemm_emulation(emulate_bf16_gemm)
    op.set_source_attention_plugin(source_attention_plugin is not None)
    op.set_cuda_bf16_barriers(cuda_bf16_plugin is not None)
    op.set_dit_cuda_numerics(dit_cuda_plugin is not None)
    if source_attention_plugin is not None:
        ctypes.CDLL(str(source_attention_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    if cuda_bf16_plugin is not None:
        ctypes.CDLL(str(cuda_bf16_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    if dit_cuda_plugin is not None:
        ctypes.CDLL(str(dit_cuda_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    prefix = f"blocks.{block_index}"
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 160 << 30)

    # Patch embedding returns BF16 into block 0 under the official outer
    # autocast.  Every later block receives the prior FP32 residual.  Preserve
    # that one-time dtype boundary in the isolated probe.
    hidden_dtype = trt.bfloat16 if block_index == 0 else trt.float32
    hidden = network.add_input("hidden", hidden_dtype, (cfg.num_patches, cfg.dim))
    modulation = network.add_input("modulation", trt.float32, (1, 6 * cfg.dim))
    text_hidden = network.add_input("text_hidden", trt.bfloat16, (cfg.text_seq_len, cfg.dim))
    shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = _slice_chunks(
        network, modulation, 6, cfg.dim
    )

    normalized = op.layer_norm(
        network,
        hidden,
        cfg.dim,
        cfg.eps,
        round_bf16=block_index == 0,
    )
    qkv_input = op.adaptive_norm(network, normalized, shift_sa, scale_sa)
    if debug_qkv:
        _mark(network, normalized, "normalized")
        _mark(network, qkv_input, "qkv_input")
    q = op.linear(
        network,
        qkv_input,
        weights[f"{prefix}.attn1.to_q.weight"],
        weights[f"{prefix}.attn1.to_q.bias"],
    )
    k = op.linear(
        network,
        qkv_input,
        weights[f"{prefix}.attn1.to_k.weight"],
        weights[f"{prefix}.attn1.to_k.bias"],
    )
    v = op.linear(
        network,
        qkv_input,
        weights[f"{prefix}.attn1.to_v.weight"],
        weights[f"{prefix}.attn1.to_v.bias"],
    )
    if debug_qkv:
        _mark(network, q, "q_raw")
        _mark(network, k, "k_raw")
        _mark(network, v, "v_raw")
    q = op.rms_norm(network, q, weights[f"{prefix}.attn1.norm_q.weight"], cfg.dim, cfg.eps)
    k = op.rms_norm(network, k, weights[f"{prefix}.attn1.norm_k.weight"], cfg.dim, cfg.eps)
    if debug_qkv:
        _mark(network, q, "q_normalized")
        _mark(network, k, "k_normalized")
    rope_cos, rope_sin = _wan_rope(cfg.latent_frames, cfg.latent_height, cfg.latent_width)
    q = op.rotary(network, q, rope_cos, rope_sin, cfg.num_patches, cfg.num_heads, cfg.head_dim)
    k = op.rotary(network, k, rope_cos, rope_sin, cfg.num_patches, cfg.num_heads, cfg.head_dim)
    if debug_qkv:
        _mark(network, q, "q_rotated")
        _mark(network, k, "k_rotated")
    self_update = op.attention(
        network,
        q,
        k,
        v,
        q_seq=cfg.num_patches,
        kv_seq=cfg.num_patches,
        heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        fp32_accumulation=self_attention_fp32,
    )
    if debug_qkv:
        _mark(network, self_update, "self_context")
    self_update = op.linear(
        network,
        self_update,
        weights[f"{prefix}.attn1.to_out.0.weight"],
        weights[f"{prefix}.attn1.to_out.0.bias"],
    )
    _mark(network, self_update, "self_update")
    after_self = op.add_fp32_residual(network, hidden, self_update, gate_sa)
    _mark(network, after_self, "after_self")

    cross_input = op.affine_layer_norm(
        network,
        after_self,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
        cfg.dim,
        cfg.eps,
    )
    cq = op.linear(
        network,
        cross_input,
        weights[f"{prefix}.attn2.to_q.weight"],
        weights[f"{prefix}.attn2.to_q.bias"],
    )
    ck = op.linear(
        network,
        text_hidden,
        weights[f"{prefix}.attn2.to_k.weight"],
        weights[f"{prefix}.attn2.to_k.bias"],
    )
    cv = op.linear(
        network,
        text_hidden,
        weights[f"{prefix}.attn2.to_v.weight"],
        weights[f"{prefix}.attn2.to_v.bias"],
    )
    cq = op.rms_norm(network, cq, weights[f"{prefix}.attn2.norm_q.weight"], cfg.dim, cfg.eps)
    ck = op.rms_norm(network, ck, weights[f"{prefix}.attn2.norm_k.weight"], cfg.dim, cfg.eps)
    cross_update = op.attention(
        network,
        cq,
        ck,
        cv,
        q_seq=cfg.num_patches,
        kv_seq=cfg.text_seq_len,
        heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        fp32_accumulation=cross_attention_fp32,
    )
    cross_update = op.linear(
        network,
        cross_update,
        weights[f"{prefix}.attn2.to_out.0.weight"],
        weights[f"{prefix}.attn2.to_out.0.bias"],
    )
    _mark(network, cross_update, "cross_update")
    after_cross = op.add_fp32_residual(network, after_self, cross_update)
    _mark(network, after_cross, "after_cross")

    normalized = op.layer_norm(network, after_cross, cfg.dim, cfg.eps)
    ffn_input = op.adaptive_norm(network, normalized, shift_ff, scale_ff)
    ffn_update = op.linear(
        network,
        ffn_input,
        weights[f"{prefix}.ffn.net.0.proj.weight"],
        weights[f"{prefix}.ffn.net.0.proj.bias"],
    )
    ffn_update = op.gelu_tanh(network, ffn_update)
    ffn_update = op.linear(
        network,
        ffn_update,
        weights[f"{prefix}.ffn.net.2.weight"],
        weights[f"{prefix}.ffn.net.2.bias"],
    )
    _mark(network, ffn_update, "ffn_update")
    output = op.add_fp32_residual(network, after_cross, ffn_update, gate_ff)
    _mark(network, output, "hidden_out")

    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the isolated Wan2.2 block")
    return bytes(plan)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--self-attention-fp32", action="store_true")
    parser.add_argument("--cross-attention-fp32", action="store_true")
    parser.add_argument("--emulate-bf16-gemm", action="store_true")
    parser.add_argument("--debug-qkv", action="store_true")
    parser.add_argument("--source-attention-plugin", type=Path)
    parser.add_argument("--cuda-bf16-plugin", type=Path)
    parser.add_argument("--dit-cuda-plugin", type=Path)
    args = parser.parse_args()
    if args.block_index < 0 or args.block_index >= WAN22_TI2V_5B.num_layers:
        parser.error("block-index is outside the model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_block(
        args.checkpoint,
        args.block_index,
        self_attention_fp32=args.self_attention_fp32,
        cross_attention_fp32=args.cross_attention_fp32,
        emulate_bf16_gemm=args.emulate_bf16_gemm,
        debug_qkv=args.debug_qkv,
        source_attention_plugin=args.source_attention_plugin,
        cuda_bf16_plugin=args.cuda_bf16_plugin,
        dit_cuda_plugin=args.dit_cuda_plugin,
    )
    args.output.write_bytes(plan)
    print(f"wrote {args.output} ({len(plan)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
