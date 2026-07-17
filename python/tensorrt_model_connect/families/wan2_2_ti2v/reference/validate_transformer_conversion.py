#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the upstream Wan2.2 DiT with this family's canonical mapping.

This is deliberately a component test, not a pipeline similarity metric.  Both
implementations receive identical tensors and the report records full-tensor
error, cosine similarity, and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import WanTransformer3DModel

from tensorrt_model_connect.families.wan2_2_ti2v.checkpoint_mapper import (
    convert_transformer_state_dict,
    load_native_transformer_state_dict,
)
from tensorrt_model_connect.families.wan2_2_ti2v.model_config import WAN22_TI2V_5B


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.detach().float().cpu()
    got = candidate.detach().float().cpu()
    delta = got - ref
    ref_flat = ref.flatten().double()
    got_flat = got.flatten().double()
    cosine = torch.nn.functional.cosine_similarity(ref_flat, got_flat, dim=0)
    return {
        "shape": list(ref.shape),
        "reference_sha256_fp32": _tensor_hash(ref),
        "candidate_sha256_fp32": _tensor_hash(got),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(cosine),
    }


def _capture(store: dict[str, torch.Tensor], name: str):
    def hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, torch.Tensor):
            store[name] = output.detach().float().cpu()

    return hook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--latent-frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.model import WanModel  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    native_model = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
    native_loaded = time.perf_counter()

    canonical_state = convert_transformer_state_dict(
        load_native_transformer_state_dict(args.checkpoint)
    )
    cfg = WAN22_TI2V_5B
    with init_empty_weights():
        canonical_model = WanTransformer3DModel(
            patch_size=cfg.patch_size,
            num_attention_heads=cfg.num_heads,
            attention_head_dim=cfg.head_dim,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            text_dim=cfg.text_dim,
            freq_dim=cfg.freq_dim,
            ffn_dim=cfg.ffn_dim,
            num_layers=cfg.num_layers,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            eps=cfg.eps,
        )
    load_result = canonical_model.load_state_dict(canonical_state, strict=True, assign=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(str(load_result))
    del canonical_state
    canonical_model.eval().requires_grad_(False)
    canonical_loaded = time.perf_counter()

    native_model.to(device)
    canonical_model.to(device)
    moved = time.perf_counter()

    native_trace: dict[str, torch.Tensor] = {}
    canonical_trace: dict[str, torch.Tensor] = {}
    if args.trace:
        native_model.patch_embedding.register_forward_hook(
            _capture(native_trace, "patch_embedding")
        )
        canonical_model.patch_embedding.register_forward_hook(
            _capture(canonical_trace, "patch_embedding")
        )
        native_model.time_embedding.register_forward_hook(_capture(native_trace, "time_embedding"))
        canonical_model.condition_embedder.time_embedder.register_forward_hook(
            _capture(canonical_trace, "time_embedding")
        )
        native_model.time_projection.register_forward_hook(
            _capture(native_trace, "time_projection")
        )
        canonical_model.condition_embedder.time_proj.register_forward_hook(
            _capture(canonical_trace, "time_projection")
        )
        native_model.text_embedding.register_forward_hook(_capture(native_trace, "text_embedding"))
        canonical_model.condition_embedder.text_embedder.register_forward_hook(
            _capture(canonical_trace, "text_embedding")
        )
        for index in (0, 1, 2, 5, 10, 20, 29):
            native_model.blocks[index].register_forward_hook(
                _capture(native_trace, f"block_{index}")
            )
            canonical_model.blocks[index].register_forward_hook(
                _capture(canonical_trace, f"block_{index}")
            )
        native_model.blocks[0].self_attn.register_forward_hook(
            _capture(native_trace, "block_0_self_attention")
        )
        canonical_model.blocks[0].attn1.register_forward_hook(
            _capture(canonical_trace, "block_0_self_attention")
        )
        native_model.blocks[0].cross_attn.register_forward_hook(
            _capture(native_trace, "block_0_cross_attention")
        )
        canonical_model.blocks[0].attn2.register_forward_hook(
            _capture(canonical_trace, "block_0_cross_attention")
        )
        native_model.blocks[0].ffn.register_forward_hook(_capture(native_trace, "block_0_ffn"))
        canonical_model.blocks[0].ffn.register_forward_hook(
            _capture(canonical_trace, "block_0_ffn")
        )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn(
        cfg.in_channels,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    num_patches = (
        args.latent_frames
        * (args.latent_height // cfg.patch_size[1])
        * (args.latent_width // cfg.patch_size[2])
    )
    timestep = torch.full((1, num_patches), 500.0, device=device, dtype=torch.float32)
    context_short = torch.randn(37, cfg.text_dim, generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.bfloat16
    )
    context_padded = torch.zeros(
        1, cfg.text_seq_len, cfg.text_dim, device=device, dtype=torch.bfloat16
    )
    context_padded[0, : context_short.shape[0]] = context_short

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = native_model([latent], timestep, [context_short], seq_len=num_patches)[0]
        torch.cuda.synchronize(device)
        native_done = time.perf_counter()
        candidate = canonical_model(
            latent.unsqueeze(0),
            timestep,
            context_padded,
            return_dict=False,
        )[0][0].float()
        torch.cuda.synchronize(device)
        canonical_done = time.perf_counter()

    report = {
        "kind": "wan2_2_ti2v_transformer_conversion_parity",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "latent_shape": list(latent.shape),
        "num_patches": num_patches,
        "context_tokens": int(context_short.shape[0]),
        "timing_seconds": {
            "native_load": native_loaded - started,
            "canonical_load": canonical_loaded - native_loaded,
            "move_to_gpu": moved - canonical_loaded,
            "native_forward": native_done - moved,
            "canonical_forward": canonical_done - native_done,
        },
        "metrics": _metrics(reference, candidate),
        "trace_metrics": {
            name: _metrics(native_trace[name], canonical_trace[name])
            for name in native_trace.keys() & canonical_trace.keys()
            if native_trace[name].shape == canonical_trace[name].shape
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
