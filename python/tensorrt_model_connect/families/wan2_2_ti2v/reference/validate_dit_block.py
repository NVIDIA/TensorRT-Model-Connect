#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare one TensorRT Wan2.2 block from the exact upstream block input."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

from tensorrt_model_connect.trt_compat import trt
import torch


BASE_OUTPUT_NAMES = (
    "self_update",
    "after_self",
    "cross_update",
    "after_cross",
    "ffn_update",
    "hidden_out",
)


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | int]:
    reference = reference.detach().float().cpu().reshape(-1).double()
    actual = actual.detach().float().cpu().reshape(-1).double()
    delta = actual - reference
    exact_elements = int(torch.count_nonzero(actual == reference))
    elements = reference.numel()
    return {
        "exact_elements": exact_elements,
        "elements": elements,
        "exact_fraction": exact_elements / elements,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(reference, actual, dim=0)),
    }


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
    parser.add_argument("--native-plugin", type=Path, action="append", default=[])
    args = parser.parse_args()
    for native_plugin in args.native_plugin:
        ctypes.CDLL(str(native_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
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
    if native.freqs.device != device:
        native.freqs = native.freqs.to(device)
    captures: dict[str, torch.Tensor] = {}
    block = native.blocks[args.block_index]
    hooks = [
        block.self_attn.register_forward_hook(
            lambda _m, _i, value: captures.__setitem__("self_update", value.detach())
        ),
        block.cross_attn.register_forward_hook(
            lambda _m, _i, value: captures.__setitem__("cross_update", value.detach())
        ),
        block.ffn.register_forward_hook(
            lambda _m, _i, value: captures.__setitem__("ffn_update", value.detach())
        ),
    ]
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
        hidden_input = hidden.detach()
        modulation = (
            (block.modulation.unsqueeze(0) + time_proj[:, :1]).reshape(1, 6 * native.dim).detach()
        )
        modulation_chunks = (block.modulation.unsqueeze(0) + time_proj[:, :1]).chunk(6, dim=2)
        normalized = block.norm1(hidden_input)
        qkv_input = normalized.float() * (1 + modulation_chunks[1].squeeze(2)) + modulation_chunks[
            0
        ].squeeze(2)
        q_raw = block.self_attn.q(qkv_input)
        k_raw = block.self_attn.k(qkv_input)
        v_raw = block.self_attn.v(qkv_input)
        q_normalized = block.self_attn.norm_q(q_raw)
        k_normalized = block.self_attn.norm_k(k_raw)
        q_heads = q_normalized.view(
            1, num_patches, native.num_heads, native.dim // native.num_heads
        )
        k_heads = k_normalized.view_as(q_heads)
        v_heads = v_raw.view_as(q_heads)
        q_rotated = rope_apply(q_heads, grid_sizes, native.freqs)
        k_rotated = rope_apply(k_heads, grid_sizes, native.freqs)
        self_context = source_attention(
            q_rotated,
            k_rotated,
            v_heads,
            k_lens=seq_lens,
        ).flatten(2)
        captures.update(
            {
                "normalized": normalized.detach(),
                "qkv_input": qkv_input.detach(),
                "q_raw": q_raw.detach(),
                "k_raw": k_raw.detach(),
                "v_raw": v_raw.detach(),
                "q_normalized": q_normalized.detach(),
                "k_normalized": k_normalized.detach(),
                "q_rotated": q_rotated.to(torch.bfloat16).flatten(2).detach(),
                "k_rotated": k_rotated.to(torch.bfloat16).flatten(2).detach(),
                "self_context": self_context.detach(),
            }
        )
        after_self = hidden_input + captures.get("self_update", 0)  # placeholder until hook runs
        hidden_out = block(hidden_input, **kwargs)
        # Reconstruct the two intermediate residual boundaries from captured updates.
        chunks = modulation.reshape(1, 1, 6, native.dim).chunk(6, dim=2)
        after_self = hidden_input.float() + captures["self_update"].float() * chunks[2].squeeze(2)
        after_cross = after_self + captures["cross_update"].float()
        captures["after_self"] = after_self
        captures["after_cross"] = after_cross
        captures["hidden_out"] = hidden_out.detach()
    for hook in hooks:
        hook.remove()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    context = engine.create_execution_context()
    output_names = tuple(
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
    )
    if not set(BASE_OUTPUT_NAMES).issubset(output_names):
        raise RuntimeError(f"Block engine is missing required outputs: {output_names}")
    hidden_dtype = (
        torch.bfloat16 if engine.get_tensor_dtype("hidden") == trt.bfloat16 else torch.float32
    )
    trt_inputs = {
        "hidden": hidden_input.reshape(num_patches, native.dim).to(hidden_dtype).contiguous(),
        "modulation": modulation.float().contiguous(),
        "text_hidden": text_hidden.reshape(native.text_len, native.dim)
        .to(torch.bfloat16)
        .contiguous(),
    }
    trt_outputs = {
        name: torch.empty(tuple(engine.get_tensor_shape(name)), device=device, dtype=torch.float32)
        for name in output_names
    }
    for name, tensor in {**trt_inputs, **trt_outputs}.items():
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.current_stream(device).cuda_stream
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT block execution failed")
    torch.cuda.synchronize(device)

    report = {
        "kind": "wan2_2_ti2v_isolated_block_parity",
        "device": torch.cuda.get_device_name(device),
        "block_index": args.block_index,
        "num_patches": num_patches,
        "metrics": {name: _metrics(captures[name], trt_outputs[name]) for name in output_names},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
