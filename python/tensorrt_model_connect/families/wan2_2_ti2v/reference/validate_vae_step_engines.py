#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate recurrent Wan2.2 VAE TensorRT engines against Diffusers.

The comparison exercises the same source contract as
``AutoencoderKLWan._decode``: one initializer call emits one video frame and
each recurrent call emits four.  The raw FP32 decoded video and all 32 final
causal-convolution cache tensors are compared without image quantization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import tensorrt as trt
import torch
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

from tensorrt_model_connect.families.wan2_2_ti2v.checkpoint_mapper import (
    VAE22_CONFIG,
    convert_vae_state_dict,
    load_native_vae_state_dict,
)
from tensorrt_model_connect.families.wan2_2_ti2v.vae_builder import (
    load_vae_cuda_plugin,
)
from tensorrt_model_connect.families.wan2_2_ti2v.vae_step_builder import (
    VAE_STEP_CACHE_SPECS,
    Wan22VaeStepProfile,
    vae_step_cache_bytes,
)


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference = reference.float().cpu()
    actual = actual.float().cpu()
    delta = actual - reference
    reference_all_zero = bool(torch.count_nonzero(reference) == 0)
    actual_all_zero = bool(torch.count_nonzero(actual) == 0)
    if reference_all_zero and actual_all_zero:
        cosine_similarity = 1.0
    else:
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(
                reference.flatten().double(), actual.flatten().double(), dim=0
            )
        )
    return {
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": cosine_similarity,
        "bitwise_exact": bool(torch.equal(reference, actual)),
        "reference_all_zero": reference_all_zero,
        "actual_all_zero": actual_all_zero,
    }


def _engine_device_memory_bytes(engine: trt.ICudaEngine) -> int:
    """Read the TRT 11 diagnostic while remaining usable with TRT 10."""

    if hasattr(engine, "device_memory_size_v2"):
        return int(engine.device_memory_size_v2)
    return int(engine.device_memory_size)


class _EngineRunner:
    """One deserialized TensorRT step engine reused across recurrent calls."""

    def __init__(self, plan_path: Path, *, device: torch.device) -> None:
        self.plan_path = plan_path
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine {plan_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create execution context for {plan_path}")
        self.latencies: list[float] = []

    def run(
        self, *, latent_frame: torch.Tensor, caches: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        expected_latent = tuple(self.engine.get_tensor_shape("latent_frame"))
        if tuple(latent_frame.shape) != expected_latent:
            raise ValueError(
                f"{self.plan_path} expects latent_frame={expected_latent}, "
                f"got {tuple(latent_frame.shape)}"
            )
        self.context.set_tensor_address("latent_frame", latent_frame.data_ptr())
        for index, cache in enumerate(caches):
            name = f"cache_{index}"
            expected = tuple(self.engine.get_tensor_shape(name))
            if tuple(cache.shape) != expected:
                raise ValueError(
                    f"{self.plan_path} expects {name}={expected}, got {tuple(cache.shape)}"
                )
            self.context.set_tensor_address(name, cache.data_ptr())

        video = torch.empty(
            tuple(self.engine.get_tensor_shape("video_frame")),
            device=self.device,
            dtype=torch.float32,
        )
        cache_outputs = [
            torch.empty(
                tuple(self.engine.get_tensor_shape(f"cache_out_{index}")),
                device=self.device,
                dtype=torch.float32,
            )
            for index in range(len(VAE_STEP_CACHE_SPECS))
        ]
        self.context.set_tensor_address("video_frame", video.data_ptr())
        for index, cache in enumerate(cache_outputs):
            self.context.set_tensor_address(f"cache_out_{index}", cache.data_ptr())

        stream = torch.cuda.current_stream(self.device).cuda_stream
        started = time.perf_counter()
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError(f"TensorRT execution failed for {self.plan_path}")
        torch.cuda.synchronize(self.device)
        self.latencies.append(time.perf_counter() - started)
        return video, cache_outputs

    def diagnostics(self) -> dict[str, Any]:
        return {
            "plan": str(self.plan_path.resolve()),
            "plan_bytes": self.plan_path.stat().st_size,
            "device_memory_bytes": _engine_device_memory_bytes(self.engine),
            "num_aux_streams": int(self.engine.num_aux_streams),
            "calls": len(self.latencies),
            "total_latency_seconds": sum(self.latencies),
            "per_call_latency_seconds": self.latencies,
        }

    def close(self) -> None:
        self.context = None
        self.engine = None
        self.runtime = None
        torch.cuda.empty_cache()


def _reference(
    checkpoint: Path,
    *,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], float]:
    started = time.perf_counter()
    vae = AutoencoderKLWan(**VAE22_CONFIG).eval().requires_grad_(False)
    vae.load_state_dict(
        convert_vae_state_dict(load_native_vae_state_dict(checkpoint)),
        strict=True,
    )
    vae.to(latent.device)
    mean = latent.new_tensor(VAE22_CONFIG["latents_mean"]).view(1, 48, 1, 1, 1)
    std = latent.new_tensor(VAE22_CONFIG["latents_std"]).view(1, 48, 1, 1, 1)

    with torch.inference_mode():
        x = vae.post_quant_conv(latent * std + mean)
        vae.clear_cache()
        profile = Wan22VaeStepProfile(latent.shape[-2], latent.shape[-1])
        video_chunks = []
        for frame in range(latent.shape[2]):
            print(
                f"[wan22-vae-parity] Diffusers latent {frame + 1}/{latent.shape[2]}",
                file=sys.stderr,
                flush=True,
            )
            vae._conv_idx = [0]
            patched = vae.decoder(
                x[:, :, frame : frame + 1],
                feat_cache=vae._feat_map,
                feat_idx=vae._conv_idx,
                first_chunk=frame == 0,
            )
            if vae._conv_idx[0] != len(VAE_STEP_CACHE_SPECS):
                raise RuntimeError(
                    f"Diffusers frame {frame} visited {vae._conv_idx[0]} caches, expected 32"
                )
            video_chunks.append(unpatchify(patched, patch_size=2).clamp(-1.0, 1.0).float().cpu())
        final_caches = []
        for spec in VAE_STEP_CACHE_SPECS:
            cache = vae._feat_map[spec.index]
            if not isinstance(cache, torch.Tensor):
                raise TypeError(
                    f"Final Diffusers cache {spec.index} is {type(cache).__name__}, expected tensor"
                )
            if tuple(cache.shape) != spec.shape(profile):
                raise ValueError(
                    f"Final Diffusers cache {spec.index} is {tuple(cache.shape)}, "
                    f"expected {spec.shape(profile)}"
                )
            final_caches.append(cache.float().cpu())
        torch.cuda.synchronize(latent.device)
    return torch.cat(video_chunks, dim=2), final_caches, time.perf_counter() - started


def _cache_metrics(
    reference: list[torch.Tensor], actual: list[torch.Tensor]
) -> list[dict[str, Any]]:
    return [
        {
            "index": spec.index,
            "logical_name": spec.logical_name,
            "shape": list(reference[spec.index].shape),
            **_metrics(reference[spec.index], actual[spec.index]),
        }
        for spec in VAE_STEP_CACHE_SPECS
    ]


def _load_latent(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        if "latent" not in value:
            raise KeyError(f"Latent fixture {path} has no 'latent' entry")
        value = value["latent"]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Latent fixture {path} contains {type(value).__name__}, expected tensor")
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 48:
        raise ValueError(
            f"Latent fixture {path} has shape {tuple(value.shape)}, expected [1,48,T,H,W]"
        )
    if value.shape[2] < 2:
        raise ValueError(f"Latent fixture needs at least two frames, got {value.shape[2]}")
    return value.float().contiguous()


def _worst_metric(metrics: list[dict[str, Any]], *, key: str, minimize: bool) -> dict[str, Any]:
    return dict(
        min(metrics, key=lambda item: item[key])
        if minimize
        else max(metrics, key=lambda item: item[key])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--latent",
        type=Path,
        help="Normalized [1,48,T,H,W] latent fixture; random 2x2x2 input when omitted",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    load_vae_cuda_plugin()
    if args.latent is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latent = torch.randn(
            1, 48, 2, 2, 2, generator=generator, device=device, dtype=torch.float32
        )
        latent_source = f"torch.randn(seed={args.seed})"
    else:
        latent = _load_latent(args.latent).to(device)
        latent_source = str(args.latent.resolve())
    profile = Wan22VaeStepProfile(latent.shape[-2], latent.shape[-1])
    zero_caches = [
        torch.zeros(spec.shape(profile), device=device, dtype=torch.float32)
        for spec in VAE_STEP_CACHE_SPECS
    ]

    initializer = _EngineRunner(args.initializer, device=device)
    first, current_caches = initializer.run(
        latent_frame=latent[:, :, :1].contiguous(), caches=zero_caches
    )
    video_chunks = [first.float().cpu()]
    initializer_diagnostics = initializer.diagnostics()
    initializer.close()
    del zero_caches, first, initializer
    torch.cuda.empty_cache()

    recurrent = _EngineRunner(args.recurrent, device=device)
    for frame in range(1, latent.shape[2]):
        print(
            f"[wan22-vae-parity] TensorRT latent {frame + 1}/{latent.shape[2]}",
            file=sys.stderr,
            flush=True,
        )
        chunk, next_caches = recurrent.run(
            latent_frame=latent[:, :, frame : frame + 1].contiguous(),
            caches=current_caches,
        )
        video_chunks.append(chunk.float().cpu())
        del current_caches, chunk
        current_caches = next_caches
    got_video = torch.cat(video_chunks, dim=2)
    actual_final_caches = [cache.float().cpu() for cache in current_caches]
    recurrent_diagnostics = recurrent.diagnostics()
    recurrent.close()
    del current_caches, recurrent, video_chunks
    torch.cuda.empty_cache()

    reference_video, reference_final_caches, reference_seconds = _reference(
        args.checkpoint, latent=latent
    )
    final_cache_metrics = _cache_metrics(reference_final_caches, actual_final_caches)
    per_frame_metrics = [
        {"frame": frame, **_metrics(reference_video[:, :, frame], got_video[:, :, frame])}
        for frame in range(reference_video.shape[2])
    ]
    tensor_outputs = None
    if args.tensor_output_dir is not None:
        args.tensor_output_dir.mkdir(parents=True, exist_ok=True)
        trt_output = args.tensor_output_dir / "tensorrt_video_fp32.pt"
        reference_output = args.tensor_output_dir / "diffusers_video_fp32.pt"
        torch.save(got_video, trt_output)
        torch.save(reference_video, reference_output)
        tensor_outputs = {
            "tensorrt_video_fp32": str(trt_output.resolve()),
            "diffusers_video_fp32": str(reference_output.resolve()),
        }
    report = {
        "kind": "wan2_2_ti2v_vae_recurrent_step_parity",
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "latent_source": latent_source,
        "latent_shape": list(latent.shape),
        "video_shape": list(reference_video.shape),
        "video_dtype": str(reference_video.dtype),
        "image_quantization_before_comparison": False,
        "cache_count": len(VAE_STEP_CACHE_SPECS),
        "cache_set_bytes": vae_step_cache_bytes(profile),
        "initializer": initializer_diagnostics,
        "recurrent": recurrent_diagnostics,
        "reference_load_and_decode_seconds": reference_seconds,
        "video_metrics": _metrics(reference_video, got_video),
        "initializer_video_metrics": _metrics(reference_video[:, :, :1], got_video[:, :, :1]),
        "recurrent_video_metrics": _metrics(reference_video[:, :, 1:], got_video[:, :, 1:]),
        "per_frame_metrics": per_frame_metrics,
        "final_cache_metrics": final_cache_metrics,
        "cache_summary": {
            "worst_cosine_similarity": _worst_metric(
                final_cache_metrics, key="cosine_similarity", minimize=True
            ),
            "worst_max_abs_error": _worst_metric(
                final_cache_metrics, key="max_abs_error", minimize=False
            ),
        },
        "frame_summary": {
            "worst_cosine_similarity": _worst_metric(
                per_frame_metrics, key="cosine_similarity", minimize=True
            ),
            "worst_max_abs_error": _worst_metric(
                per_frame_metrics, key="max_abs_error", minimize=False
            ),
        },
        "tensor_outputs": tensor_outputs,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
