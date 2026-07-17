#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feed exact upstream Wan2.2 Q/K/V into a TensorRT attention probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import tensorrt as trt
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--context-tokens", type=int, default=37)
    parser.add_argument("--native-plugin", type=Path)
    args = parser.parse_args()
    if args.native_plugin is not None:
        ctypes.CDLL(str(args.native_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.attention import attention as source_attention
    from wan.modules.model import WanModel, rope_apply, sinusoidal_embedding_1d

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn((1, 48, 31, 44, 80), generator=generator, device=device)
    text_short = torch.randn(
        args.context_tokens,
        4096,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float32)
    native = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False).to(device)
    native.freqs = native.freqs.to(device)
    block = native.blocks[args.block_index]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = native.patch_embedding(latent).flatten(2).transpose(1, 2)
        num_patches = hidden.shape[1]
        grid_sizes = torch.tensor([[31, 22, 40]], dtype=torch.long)
        seq_lens = torch.tensor([num_patches], dtype=torch.long)
        expanded_timestep = timestep.expand(1, num_patches)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            time = native.time_embedding(
                sinusoidal_embedding_1d(256, expanded_timestep.flatten())
                .unflatten(0, (1, num_patches))
                .float()
            )
            time_proj = native.time_projection(time).unflatten(2, (6, native.dim))
        padded_text = torch.cat(
            [
                text_short,
                text_short.new_zeros(native.text_len - text_short.shape[0], 4096),
            ]
        ).unsqueeze(0)
        text_hidden = native.text_embedding(padded_text)
        kwargs = {
            "e": time_proj,
            "seq_lens": seq_lens,
            "grid_sizes": grid_sizes,
            "freqs": native.freqs,
            "context": text_hidden,
            "context_lens": None,
        }
        for index in range(args.block_index):
            hidden = native.blocks[index](hidden, **kwargs)
        modulation = (block.modulation.unsqueeze(0) + time_proj).chunk(6, dim=2)
        qkv_input = block.norm1(hidden).float() * (1 + modulation[1].squeeze(2)) + modulation[
            0
        ].squeeze(2)
        q = block.self_attn.norm_q(block.self_attn.q(qkv_input)).view(
            1, num_patches, native.num_heads, native.dim // native.num_heads
        )
        k = block.self_attn.norm_k(block.self_attn.k(qkv_input)).view_as(q)
        v = block.self_attn.v(qkv_input).view_as(q)
        q = rope_apply(q, grid_sizes, native.freqs).transpose(1, 2).contiguous().to(torch.bfloat16)
        k = rope_apply(k, grid_sizes, native.freqs).transpose(1, 2).contiguous().to(torch.bfloat16)
        v = v.transpose(1, 2).contiguous().to(torch.bfloat16)
        reference = (
            source_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                k_lens=seq_lens,
            )
            .transpose(1, 2)
            .contiguous()
            .float()
        )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    context = engine.create_execution_context()
    actual = torch.empty(
        tuple(engine.get_tensor_shape("context")), device=device, dtype=torch.float32
    )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("context", actual)):
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.current_stream(device).cuda_stream
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT attention probe execution failed")
    torch.cuda.synchronize(device)
    reference = reference.double().reshape(-1)
    actual_cpu = actual.double().cpu().reshape(-1)
    delta = actual_cpu - reference.cpu()
    report = {
        "kind": "wan2_2_ti2v_source_qkv_attention_parity",
        "device": torch.cuda.get_device_name(device),
        "block_index": args.block_index,
        "shape": list(actual.shape),
        "metrics": {
            "max_abs_error": float(delta.abs().max()),
            "mean_abs_error": float(delta.abs().mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(reference.cpu(), actual_cpu, dim=0)
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
