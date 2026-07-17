#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a fixed-shape TensorRT Wan2.2 VAE plan with PyTorch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorrt as trt
import torch
from diffusers import AutoencoderKLWan

from tensorrt_model_connect.families.wan2_2_ti2v.checkpoint_mapper import (
    VAE22_CONFIG,
    convert_vae_state_dict,
    load_native_vae_state_dict,
)
from tensorrt_model_connect.families.wan2_2_ti2v.vae_builder import load_vae_cuda_plugin


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    delta = actual - reference
    return {
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference.flatten().double(), actual.flatten().double(), dim=0
            )
        ),
    }


def _per_frame_metrics(
    reference: torch.Tensor, actual: torch.Tensor
) -> list[dict[str, float | int]]:
    return [
        {"frame": frame, **_metrics(reference[:, :, frame], actual[:, :, frame])}
        for frame in range(reference.shape[2])
    ]


def _load_tensor(path: Path, *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and "latent" in value:
        value = value["latent"]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value).__name__}")
    if tuple(value.shape) == expected_shape[1:]:
        value = value.unsqueeze(0)
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"Tensor in {path} has shape {tuple(value.shape)}, expected {expected_shape}"
        )
    return value.float().contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--latent",
        type=Path,
        help="Normalized official-pipeline latent; random seed input is used when omitted",
    )
    parser.add_argument(
        "--official-video",
        type=Path,
        help="Optional clamped official decode tensor for an additional source comparison",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    load_vae_cuda_plugin()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize TensorRT VAE engine")
    context = engine.create_execution_context()
    input_shape = tuple(engine.get_tensor_shape("latents"))
    output_shape = tuple(engine.get_tensor_shape("video"))
    if args.latent is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latent = torch.randn(input_shape, generator=generator, device=device)
        latent_source = f"torch.randn(seed={args.seed})"
    else:
        latent = _load_tensor(args.latent, expected_shape=input_shape).to(device)
        latent_source = str(args.latent.resolve())
    output = torch.empty(output_shape, device=device, dtype=torch.float32)
    context.set_tensor_address("latents", latent.data_ptr())
    context.set_tensor_address("video", output.data_ptr())
    stream = torch.cuda.current_stream(device).cuda_stream
    trt_started = time.perf_counter()
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT VAE execution failed")
    torch.cuda.synchronize(device)
    trt_seconds = time.perf_counter() - trt_started
    got = output.float().cpu()
    del output, context, engine, runtime
    torch.cuda.empty_cache()

    reference_started = time.perf_counter()
    vae = AutoencoderKLWan(**VAE22_CONFIG).eval().requires_grad_(False)
    vae.load_state_dict(
        convert_vae_state_dict(load_native_vae_state_dict(args.checkpoint)),
        strict=True,
    )
    vae.to(device)
    mean = latent.new_tensor(VAE22_CONFIG["latents_mean"]).view(1, 48, 1, 1, 1)
    std = latent.new_tensor(VAE22_CONFIG["latents_std"]).view(1, 48, 1, 1, 1)
    with torch.inference_mode():
        reference = vae.decode(latent * std + mean, return_dict=False)[0]
    torch.cuda.synchronize(device)
    reference_seconds = time.perf_counter() - reference_started

    ref = reference.float().cpu()
    report = {
        "kind": "wan2_2_ti2v_vae_tensorrt_parity",
        "device": torch.cuda.get_device_name(device),
        "latent_source": latent_source,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "timing_seconds": {
            "tensorrt_decode": trt_seconds,
            "converted_diffusers_decode_and_load": reference_seconds,
        },
        "metrics": _metrics(ref, got),
        "per_frame_metrics": _per_frame_metrics(ref, got),
    }
    if args.official_video is not None:
        official = _load_tensor(args.official_video, expected_shape=output_shape)
        report["official_video_source"] = str(args.official_video.resolve())
        report["tensorrt_vs_official_clamped_metrics"] = _metrics(official, got.clamp(-1.0, 1.0))
        report["converted_diffusers_vs_official_clamped_metrics"] = _metrics(
            official, ref.clamp(-1.0, 1.0)
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
