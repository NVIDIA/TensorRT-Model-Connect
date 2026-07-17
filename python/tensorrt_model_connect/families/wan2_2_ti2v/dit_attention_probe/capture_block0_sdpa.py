#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the exact upstream Wan2.2 block-0 self-SDPA contract.

This is a qualification-only helper.  It writes raw BF16 buffers in their
physical BSHD order plus the logical BHSD dimensions/strides passed by
PyTorch to cuDNN.  The native probe consumes those files without linking to
PyTorch or ATen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel


def _write_bf16(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{path.name}: expected BF16, got {tensor.dtype}")
    physical = tensor.detach().contiguous().cpu()
    payload = physical.view(torch.uint16).numpy().tobytes()
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "physical_shape_bshd": list(physical.shape),
    }


def _bitwise_report(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    reference_cpu = reference.detach().contiguous().cpu()
    actual_cpu = actual.detach().contiguous().cpu()
    delta = actual_cpu.float().double().reshape(-1) - reference_cpu.float().double().reshape(-1)
    return {
        "bitwise_equal": bool(
            torch.equal(reference_cpu.view(torch.uint16), actual_cpu.view(torch.uint16))
        ),
        "exact_elements": int(
            (reference_cpu.view(torch.uint16) == actual_cpu.view(torch.uint16)).sum().item()
        ),
        "elements": reference_cpu.numel(),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(
            functional.cosine_similarity(
                reference_cpu.float().double().reshape(-1),
                actual_cpu.float().double().reshape(-1),
                dim=0,
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-call", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.attention import attention as source_attention
    from wan.modules.model import WanModel, rope_apply, sinusoidal_embedding_1d

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    first_call = torch.load(args.first_call, map_location="cpu", weights_only=False)
    latent = first_call["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
    timestep = first_call["timestep"].to(device=device, dtype=torch.float32)
    text_short = first_call["context"].to(device=device, dtype=torch.bfloat16)

    native = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False).to(device)
    if native.freqs.device != device:
        native.freqs = native.freqs.to(device)
    block = native.blocks[0]

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
        # Exercise the same text projection as the full call so autocast and
        # allocator state match the official block input path.
        _ = native.text_embedding(padded_text)

        modulation = (block.modulation.unsqueeze(0) + time_proj).chunk(6, dim=2)
        qkv_input = block.norm1(hidden).float() * (1 + modulation[1].squeeze(2)) + modulation[
            0
        ].squeeze(2)
        q = block.self_attn.norm_q(block.self_attn.q(qkv_input)).view(
            1, num_patches, native.num_heads, native.dim // native.num_heads
        )
        k = block.self_attn.norm_k(block.self_attn.k(qkv_input)).view_as(q)
        v = block.self_attn.v(qkv_input).view_as(q)
        q_bshd = rope_apply(q, grid_sizes, native.freqs).to(torch.bfloat16)
        k_bshd = rope_apply(k, grid_sizes, native.freqs).to(torch.bfloat16)
        v_bshd = v.to(torch.bfloat16)

        # This is exactly the fallback branch in wan.modules.attention:
        # transpose BSHD -> BHSD, cast, default SDPA, transpose back.
        q_bhsd = q_bshd.transpose(1, 2).to(torch.bfloat16)
        k_bhsd = k_bshd.transpose(1, 2).to(torch.bfloat16)
        v_bhsd = v_bshd.transpose(1, 2).to(torch.bfloat16)
        default_bhsd = functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            cudnn_bhsd = functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        source_bshd = source_attention(
            q_bshd,
            k_bshd,
            v_bshd,
            k_lens=seq_lens,
        )

    default_bshd = default_bhsd.transpose(1, 2).contiguous()
    cudnn_bshd = cudnn_bhsd.transpose(1, 2).contiguous()
    physical_tensors = {
        "q": q_bshd,
        "k": k_bshd,
        "v": v_bshd,
        "o": cudnn_bshd,
    }
    files = {
        name: _write_bf16(args.output_dir / f"{name}_bshd_bf16.bin", tensor)
        for name, tensor in physical_tensors.items()
    }
    logical_views = {
        "q": q_bhsd,
        "k": k_bhsd,
        "v": v_bhsd,
        "o": cudnn_bhsd,
    }
    for name, tensor in logical_views.items():
        files[name]["logical_shape_bhsd"] = list(tensor.shape)
        files[name]["logical_stride_bhsd"] = list(tensor.stride())
        files[name]["dtype"] = str(tensor.dtype)

    report = {
        "kind": "wan2_2_ti2v_official_block0_cudnn_sdpa_capture",
        "official_source": str(args.official_source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "first_call": str(args.first_call.resolve()),
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "attention_scale": float(1.0 / (native.dim // native.num_heads) ** 0.5),
        "causal": False,
        "dropout_probability": 0.0,
        "invocation": shlex.join([sys.executable, *sys.argv]),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDNN_FRONTEND_LOG_INFO",
                "CUDNN_FRONTEND_LOG_FILE",
                "PYTHONPATH",
            )
        },
        "files": files,
        "comparisons": {
            "default_vs_forced_cudnn": _bitwise_report(cudnn_bshd, default_bshd),
            "wan_source_vs_forced_cudnn": _bitwise_report(cudnn_bshd, source_bshd),
        },
    }
    (args.output_dir / "capture_command.txt").write_text(report["invocation"] + "\n")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
