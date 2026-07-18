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
import ctypes
import hashlib
import json
import math
import sys
import time
import weakref
from pathlib import Path
from typing import Any

import numpy as np
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, str | int]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference = reference.float().cpu()
    actual = actual.float().cpu()
    delta = actual - reference
    reference_all_zero = bool(torch.count_nonzero(reference) == 0)
    actual_all_zero = bool(torch.count_nonzero(actual) == 0)
    reference_l2_norm = float(torch.linalg.vector_norm(reference.flatten().double()))
    actual_l2_norm = float(torch.linalg.vector_norm(actual.flatten().double()))
    delta_l2_norm = float(torch.linalg.vector_norm(delta.flatten().double()))
    relative_l2_error = (
        delta_l2_norm / reference_l2_norm
        if reference_l2_norm > 0.0
        else (0.0 if delta_l2_norm == 0.0 else math.inf)
    )
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
        "reference_l2_norm": reference_l2_norm,
        "actual_l2_norm": actual_l2_norm,
        "delta_l2_norm": delta_l2_norm,
        "relative_l2_error": relative_l2_error,
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


_CACHE_ALIGNMENT = 256
_CUDA_HOST_ALLOC_MAPPED = 0x02
_CUDA_DEV_ATTR_CAN_MAP_HOST_MEMORY = 19
_DEFAULT_MAX_VIDEO_RMSE = 2.0 / 255.0
_DEFAULT_MAX_RELATIVE_L2_ERROR = 0.01


def _align_up(value: int, alignment: int = _CACHE_ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


def _load_cudart() -> ctypes.CDLL:
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError as error:
        raise RuntimeError("Could not load CUDA runtime library libcudart.so") from error
    cudart.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cudart.cudaDeviceGetAttribute.restype = ctypes.c_int
    cudart.cudaHostAlloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    cudart.cudaHostAlloc.restype = ctypes.c_int
    cudart.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    cudart.cudaHostGetDevicePointer.restype = ctypes.c_int
    cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
    cudart.cudaFreeHost.restype = ctypes.c_int
    cudart.cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    cudart.cudaMemsetAsync.restype = ctypes.c_int
    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    return cudart


def _check_cuda(cudart: ctypes.CDLL, status: int, operation: str) -> None:
    if status == 0:
        return
    message = cudart.cudaGetErrorString(status)
    rendered = message.decode() if message is not None else f"CUDA error {status}"
    raise RuntimeError(f"{operation} failed: {rendered}")


def _cuda_device_can_map_host_memory(device: torch.device) -> bool:
    if device.type != "cuda":
        raise ValueError(f"Mapped-host VAE caches require a CUDA device, got {device}")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    cudart = _load_cudart()
    can_map = ctypes.c_int()
    _check_cuda(
        cudart,
        cudart.cudaDeviceGetAttribute(
            ctypes.byref(can_map),
            _CUDA_DEV_ATTR_CAN_MAP_HOST_MEMORY,
            device_index,
        ),
        f"cudaDevAttrCanMapHostMemory query for cuda:{device_index}",
    )
    return can_map.value != 0


class _MappedCacheSlice:
    """One typed view into a CUDA-mapped pinned-host cache bank."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        host_address: int,
        device_address: int,
        owner: _MappedHostCacheBank,
    ) -> None:
        self.shape = shape
        self._host_address = host_address
        self._device_address = device_address
        self._owner = weakref.ref(owner)

    def _require_open(self) -> None:
        owner = self._owner()
        if owner is None or owner.closed:
            raise RuntimeError("CUDA mapped-host cache slice is no longer valid")

    @property
    def device_address(self) -> int:
        self._require_open()
        return self._device_address

    def cpu_tensor(self) -> torch.Tensor:
        self._require_open()
        count = math.prod(self.shape)
        pointer = ctypes.cast(
            ctypes.c_void_p(self._host_address),
            ctypes.POINTER(ctypes.c_float),
        )
        array = np.ctypeslib.as_array(pointer, shape=(count,)).reshape(self.shape)
        return torch.from_numpy(array)


class _MappedHostCacheBank:
    """Thor cache storage matching the native runtime's cudaHostAllocMapped policy."""

    def __init__(self, profile: Wan22VaeStepProfile) -> None:
        self._closed = False
        self._host_base = ctypes.c_void_p()
        self._device_base = ctypes.c_void_p()
        self.slices: list[_MappedCacheSlice] = []
        self._cudart = _load_cudart()

        offsets = []
        cursor = 0
        shapes = [spec.shape(profile) for spec in VAE_STEP_CACHE_SPECS]
        for shape in shapes:
            cursor = _align_up(cursor)
            offsets.append(cursor)
            cursor += math.prod(shape) * ctypes.sizeof(ctypes.c_float)
        self.total_bytes = _align_up(cursor)
        _check_cuda(
            self._cudart,
            self._cudart.cudaHostAlloc(
                ctypes.byref(self._host_base),
                self.total_bytes,
                _CUDA_HOST_ALLOC_MAPPED,
            ),
            "cudaHostAllocMapped",
        )
        try:
            _check_cuda(
                self._cudart,
                self._cudart.cudaHostGetDevicePointer(
                    ctypes.byref(self._device_base),
                    self._host_base,
                    0,
                ),
                "cudaHostGetDevicePointer",
            )
        except BaseException:
            self._cudart.cudaFreeHost(self._host_base)
            self._host_base = ctypes.c_void_p()
            self._device_base = ctypes.c_void_p()
            self._closed = True
            raise
        if self._host_base.value is None or self._device_base.value is None:
            self.close()
            raise RuntimeError("CUDA mapped-host cache allocation returned a null address")
        self.slices = [
            _MappedCacheSlice(
                shape=shape,
                host_address=self._host_base.value + offset,
                device_address=self._device_base.value + offset,
                owner=self,
            )
            for shape, offset in zip(shapes, offsets, strict=True)
        ]

    @property
    def closed(self) -> bool:
        return self._closed

    def zero_async(self, stream: int) -> None:
        if self._closed:
            raise RuntimeError("CUDA mapped-host cache bank is closed")
        _check_cuda(
            self._cudart,
            self._cudart.cudaMemsetAsync(
                self._device_base,
                0,
                self.total_bytes,
                ctypes.c_void_p(stream),
            ),
            "cudaMemsetAsync(mapped-host cache)",
        )

    def close(self) -> None:
        if self._closed:
            return
        host_base = self._host_base
        self._closed = True
        self._host_base = ctypes.c_void_p()
        self._device_base = ctypes.c_void_p()
        if host_base.value is not None:
            _check_cuda(self._cudart, self._cudart.cudaFreeHost(host_base), "cudaFreeHost")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _cache_address(cache: torch.Tensor | _MappedCacheSlice) -> int:
    if isinstance(cache, _MappedCacheSlice):
        return cache.device_address
    return cache.data_ptr()


class _EngineRunner:
    """One deserialized TensorRT step engine reused across recurrent calls."""

    def __init__(self, plan_path: Path, *, device: torch.device) -> None:
        self.plan_path = plan_path
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        plan = plan_path.read_bytes()
        self.plan_sha256 = hashlib.sha256(plan).hexdigest()
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        del plan
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine {plan_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create execution context for {plan_path}")
        self.latencies: list[float] = []

    def run(
        self,
        *,
        latent_frame: torch.Tensor,
        caches: list[torch.Tensor | _MappedCacheSlice],
        cache_outputs: list[torch.Tensor | _MappedCacheSlice] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | _MappedCacheSlice]]:
        if len(caches) != len(VAE_STEP_CACHE_SPECS):
            raise ValueError(
                f"Expected {len(VAE_STEP_CACHE_SPECS)} cache inputs, got {len(caches)}"
            )
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
            self.context.set_tensor_address(name, _cache_address(cache))

        video = torch.empty(
            tuple(self.engine.get_tensor_shape("video_frame")),
            device=self.device,
            dtype=torch.float32,
        )
        if cache_outputs is None:
            cache_outputs = [
                torch.empty(
                    tuple(self.engine.get_tensor_shape(f"cache_out_{index}")),
                    device=self.device,
                    dtype=torch.float32,
                )
                for index in range(len(VAE_STEP_CACHE_SPECS))
            ]
        if len(cache_outputs) != len(VAE_STEP_CACHE_SPECS):
            raise ValueError(
                f"Expected {len(VAE_STEP_CACHE_SPECS)} cache outputs, got {len(cache_outputs)}"
            )
        self.context.set_tensor_address("video_frame", video.data_ptr())
        for index, cache in enumerate(cache_outputs):
            expected = tuple(self.engine.get_tensor_shape(f"cache_out_{index}"))
            if tuple(cache.shape) != expected:
                raise ValueError(
                    f"{self.plan_path} expects cache_out_{index}={expected}, "
                    f"got {tuple(cache.shape)}"
                )
            self.context.set_tensor_address(f"cache_out_{index}", _cache_address(cache))

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
            "plan_sha256": self.plan_sha256,
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


def _qualification(
    report: dict[str, Any],
    min_cosine: float,
    max_video_rmse: float,
    max_relative_l2_error: float = _DEFAULT_MAX_RELATIVE_L2_ERROR,
) -> dict[str, Any]:
    video_comparisons = [
        ("video", report["video_metrics"]),
        ("initializer_video", report["initializer_video_metrics"]),
        ("recurrent_video", report["recurrent_video_metrics"]),
        *((f"frame_{item['frame']}", item) for item in report["per_frame_metrics"]),
    ]
    comparisons = [
        *video_comparisons,
        *((f"final_cache_{item['index']}", item) for item in report["final_cache_metrics"]),
    ]
    non_finite_cosine_comparisons = [
        name for name, metrics in comparisons if not math.isfinite(metrics["cosine_similarity"])
    ]
    non_finite_relative_l2_comparisons = [
        name for name, metrics in comparisons if not math.isfinite(metrics["relative_l2_error"])
    ]
    non_finite_comparisons = sorted(
        set(non_finite_cosine_comparisons) | set(non_finite_relative_l2_comparisons)
    )
    worst_name = None
    worst_cosine = None
    if comparisons and not non_finite_cosine_comparisons:
        worst_name, worst_metrics = min(
            comparisons,
            key=lambda item: item[1]["cosine_similarity"],
        )
        worst_cosine = worst_metrics["cosine_similarity"]
    worst_relative_l2_name = None
    worst_relative_l2_error = None
    if comparisons and not non_finite_relative_l2_comparisons:
        worst_relative_l2_name, worst_relative_l2_metrics = max(
            comparisons,
            key=lambda item: item[1]["relative_l2_error"],
        )
        worst_relative_l2_error = worst_relative_l2_metrics["relative_l2_error"]
    non_finite_video_rmse_comparisons = [
        name for name, metrics in video_comparisons if not math.isfinite(metrics["rmse"])
    ]
    worst_video_rmse_name = None
    worst_video_rmse = None
    if video_comparisons and not non_finite_video_rmse_comparisons:
        worst_video_rmse_name, worst_video_rmse_metrics = max(
            video_comparisons,
            key=lambda item: item[1]["rmse"],
        )
        worst_video_rmse = worst_video_rmse_metrics["rmse"]
    return {
        "min_cosine": min_cosine,
        "max_video_rmse": max_video_rmse,
        "max_relative_l2_error": max_relative_l2_error,
        "comparisons_checked": len(comparisons),
        "non_finite_comparisons": non_finite_comparisons,
        "non_finite_cosine_comparisons": non_finite_cosine_comparisons,
        "non_finite_relative_l2_comparisons": non_finite_relative_l2_comparisons,
        "worst_comparison": worst_name,
        "worst_cosine_similarity": worst_cosine,
        "worst_relative_l2_comparison": worst_relative_l2_name,
        "worst_relative_l2_error": worst_relative_l2_error,
        "video_rmse_comparisons_checked": len(video_comparisons),
        "non_finite_video_rmse_comparisons": non_finite_video_rmse_comparisons,
        "worst_video_rmse_comparison": worst_video_rmse_name,
        "worst_video_rmse": worst_video_rmse,
        "passed": (
            bool(comparisons)
            and not non_finite_comparisons
            and worst_cosine is not None
            and worst_cosine >= min_cosine
            and worst_relative_l2_error is not None
            and worst_relative_l2_error <= max_relative_l2_error
            and bool(video_comparisons)
            and not non_finite_video_rmse_comparisons
            and worst_video_rmse is not None
            and worst_video_rmse <= max_video_rmse
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--recurrent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-cosine", type=float, default=0.998)
    parser.add_argument(
        "--max-video-rmse",
        type=float,
        default=_DEFAULT_MAX_VIDEO_RMSE,
        help=(
            "Maximum raw FP32 video RMSE; the default is one uint8 code level "
            "over the decoded [-1, 1] range (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--max-relative-l2-error",
        type=float,
        default=_DEFAULT_MAX_RELATIVE_L2_ERROR,
        help="Maximum relative L2 error for every video and cache comparison (default: %(default)s)",
    )
    parser.add_argument(
        "--latent",
        type=Path,
        help="Normalized [1,48,T,H,W] latent fixture; random 2x2x2 input when omitted",
    )
    args = parser.parse_args()
    if not math.isfinite(args.min_cosine) or not 0.0 <= args.min_cosine <= 1.0:
        parser.error("--min-cosine must be finite and in [0, 1]")
    if not math.isfinite(args.max_video_rmse) or not 0.0 <= args.max_video_rmse <= 2.0:
        parser.error("--max-video-rmse must be finite and in [0, 2]")
    if (
        not math.isfinite(args.max_relative_l2_error)
        or not 0.0 <= args.max_relative_l2_error <= 1.0
    ):
        parser.error("--max-relative-l2-error must be finite and in [0, 1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    vae_plugin = _file_identity(load_vae_cuda_plugin())
    if args.latent is None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latent = torch.randn(
            1, 48, 2, 2, 2, generator=generator, device=device, dtype=torch.float32
        )
        latent_source = f"torch.randn(seed={args.seed})"
    else:
        latent = _load_latent(args.latent).to(device)
        latent_source = str(args.latent.resolve())
    latent_sha256 = hashlib.sha256(
        latent.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
    ).hexdigest()
    profile = Wan22VaeStepProfile(latent.shape[-2], latent.shape[-1])
    use_mapped_host_caches = bool(torch.cuda.get_device_properties(device).is_integrated)
    if use_mapped_host_caches and not _cuda_device_can_map_host_memory(device):
        device_index = torch.cuda.current_device() if device.index is None else device.index
        raise RuntimeError(
            f"Integrated CUDA device cuda:{device_index} cannot map host memory required "
            "for Wan2.2 recurrent VAE caches"
        )
    mapped_banks: list[_MappedHostCacheBank] | None = None
    if use_mapped_host_caches:
        mapped_banks = [_MappedHostCacheBank(profile), _MappedHostCacheBank(profile)]
        stream = torch.cuda.current_stream(device)
        mapped_banks[0].zero_async(stream.cuda_stream)
        stream.synchronize()
        zero_caches: list[torch.Tensor | _MappedCacheSlice] = mapped_banks[0].slices
        initializer_cache_outputs: list[torch.Tensor | _MappedCacheSlice] | None = mapped_banks[
            1
        ].slices
    else:
        zero_caches = [
            torch.zeros(spec.shape(profile), device=device, dtype=torch.float32)
            for spec in VAE_STEP_CACHE_SPECS
        ]
        initializer_cache_outputs = None

    initializer = _EngineRunner(args.initializer, device=device)
    first, current_caches = initializer.run(
        latent_frame=latent[:, :, :1].contiguous(),
        caches=zero_caches,
        cache_outputs=initializer_cache_outputs,
    )
    video_chunks = [first.float().cpu()]
    initializer_diagnostics = initializer.diagnostics()
    initializer.close()
    del zero_caches, first, initializer
    torch.cuda.empty_cache()

    recurrent = _EngineRunner(args.recurrent, device=device)
    current_bank_index = 1
    output_bank_index = 0
    for frame in range(1, latent.shape[2]):
        print(
            f"[wan22-vae-parity] TensorRT latent {frame + 1}/{latent.shape[2]}",
            file=sys.stderr,
            flush=True,
        )
        chunk, next_caches = recurrent.run(
            latent_frame=latent[:, :, frame : frame + 1].contiguous(),
            caches=current_caches,
            cache_outputs=(
                mapped_banks[output_bank_index].slices if mapped_banks is not None else None
            ),
        )
        video_chunks.append(chunk.float().cpu())
        del current_caches, chunk
        current_caches = next_caches
        if mapped_banks is not None:
            current_bank_index, output_bank_index = output_bank_index, current_bank_index
    got_video = torch.cat(video_chunks, dim=2)
    if mapped_banks is not None:
        actual_final_caches = [
            cache.cpu_tensor() for cache in mapped_banks[current_bank_index].slices
        ]
    else:
        actual_final_caches = [cache.float().cpu() for cache in current_caches]
    recurrent_diagnostics = recurrent.diagnostics()
    recurrent.close()
    del current_caches, recurrent, video_chunks
    torch.cuda.empty_cache()
    if mapped_banks is not None:
        mapped_banks[output_bank_index].close()

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
        "latent_fp32_le_sha256": latent_sha256,
        "latent_shape": list(latent.shape),
        "video_shape": list(reference_video.shape),
        "video_dtype": str(reference_video.dtype),
        "image_quantization_before_comparison": False,
        "cache_count": len(VAE_STEP_CACHE_SPECS),
        "cache_set_bytes": vae_step_cache_bytes(profile),
        "cache_memory_kind": "mapped_host" if mapped_banks is not None else "cuda_device",
        "vae_plugin": vae_plugin,
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
    report["qualification"] = _qualification(
        report,
        min_cosine=args.min_cosine,
        max_video_rmse=args.max_video_rmse,
        max_relative_l2_error=args.max_relative_l2_error,
    )
    if mapped_banks is not None:
        del actual_final_caches
        mapped_banks[current_bank_index].close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["qualification"]["passed"]:
        raise SystemExit(
            "Wan2.2 VAE qualification failed: "
            + json.dumps(report["qualification"], sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
