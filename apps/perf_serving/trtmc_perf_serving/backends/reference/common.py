# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the Python reference adapters."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from ..base import BackendError, Invocation

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("eager", "compile")


@dataclass(frozen=True)
class ReferenceSpec:
    """How to load one reference: checkpoint, precision, and execution mode."""

    operation: str
    model: str
    revision: str | None = None
    precision: str = "fp16"
    mode: str = "eager"
    trust_remote_code: bool = False
    device: str = "cuda"
    # Golden generation: disable TF32 and request deterministic kernels (slower; not for perf runs).
    deterministic: bool = False
    # Adapter-specific options, e.g. {"processor_kwargs": {...}} for remote-code processors.
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.precision]

    def pretrained_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            kwargs["revision"] = self.revision
        return kwargs


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``call`` inside inference mode and return (result, wall ms) with device sync."""
    with torch.inference_mode():
        synchronize()
        started = time.perf_counter()
        result = call()
        synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def maybe_compile(module: torch.nn.Module, spec: ReferenceSpec) -> torch.nn.Module:
    """torch.compile ``module.forward`` (default mode, dynamic shapes) for compile references."""
    if spec.mode == "compile":
        module.forward = torch.compile(module.forward, dynamic=True)
    return module


def reject_compile(spec: ReferenceSpec, adapter: str) -> None:
    if spec.mode != "eager":
        raise BackendError(f"{adapter} reference supports only eager mode")


def required(request: Mapping[str, Any], name: str) -> Any:
    if name not in request:
        raise BackendError(f"{name} is required")
    return request[name]


def tensor_observation(value: Any, artifact_base: Path) -> dict[str, Any]:
    """Summarize a tensor output and keep the full value as ``<artifact_base>.npy``."""
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value)
    artifact = artifact_base.with_suffix(".npy")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact, array)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "artifact": str(artifact),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def load_image(path: str):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    import soundfile

    audio, rate = soundfile.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
    return audio


def invocation(observation: Mapping[str, Any], model_call_ms: float, **extra: Any) -> Invocation:
    return Invocation(observation=dict(observation), model_call_ms=model_call_ms, extra=extra)
