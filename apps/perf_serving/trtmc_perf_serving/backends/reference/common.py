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
    # A family's own native pipeline: a Python file defining ``Adapter(spec, host)`` with
    # ``invoke(request, artifact_base)`` (families/<family>/reference/adapter.py); see ``NativeHost``.
    adapter: str | None = None

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.precision]

    def pretrained_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.revision:
            kwargs["revision"] = self.revision
        return kwargs

    def model_kwargs(self) -> dict[str, Any]:
        """``from_pretrained`` arguments of a Transformers model: ``pretrained_kwargs`` and the precision,
        named ``dtype`` from Transformers 4.56 on and ``torch_dtype`` before (a family environment may pin
        an older release for its remote code)."""
        import transformers

        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        return {**self.pretrained_kwargs(), ("dtype" if (major, minor) >= (4, 56) else "torch_dtype"): self.dtype}


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


# Output evidence is written after the timed call (the TRTMC worker also stops timing before it
# writes its artifacts); input files are decoded before it (TRTMC excludes asset loading by default).
_PENDING = "_pending_array"
_FILES = "_pending_files"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
_DECODED: dict[tuple[str, str], Any] = {}


def tensor_observation(value: Any, artifact_base: Path, inline: bool = False) -> dict[str, Any]:
    """Summarize a tensor output (copied to host memory, as the Task returns it); ``write_artifacts``
    keeps the full value as ``<artifact_base>.npy`` with its sha256 once the timed call returned.
    ``inline`` also returns the values themselves (row-major), as TRTMC does for small outputs such as
    forecasts."""
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            **({"values": array.reshape(-1).astype(float).tolist()} if inline else {}),
            "artifact": str(artifact_base.with_suffix(".npy")), _PENDING: array}


def deferred_files(observation: Mapping[str, Any], files: Mapping[Path, Any]) -> dict[str, Any]:
    """``observation`` with output ``files`` (path -> array) that ``write_artifacts`` writes once the timed
    call returned, as raw little-endian values in the array's dtype (``tofile``)."""
    return {**observation, _FILES: dict(files)}


def write_artifacts(observation: Any) -> Any:
    """Write the pending output arrays and files of an observation (any nesting); arrays get their sha256."""
    if isinstance(observation, list):
        return [write_artifacts(item) for item in observation]
    if not isinstance(observation, dict):
        return observation
    result = {key: write_artifacts(value) for key, value in observation.items() if key not in (_PENDING, _FILES)}
    for path, array in (observation.get(_FILES) or {}).items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.asarray(array).tofile(path)
    if _PENDING in observation:
        array, artifact = observation[_PENDING], Path(observation["artifact"])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.save(artifact, array)
        result["sha256"] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return result


def _read_image(path: str):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile

    return soundfile.read(path, dtype="float32", always_2d=False)


def preload_inputs(request: Mapping[str, Any]) -> None:
    """Read and decode the request's image and audio files, and read its other input files, ahead of the
    timed call."""
    _DECODED.clear()
    for key, value in request.items():
        if not (key.endswith("_path") or key.endswith("_paths")):
            continue
        for path in value if isinstance(value, list) else [value]:
            if not isinstance(path, str) or not Path(path).is_file():
                continue
            suffix = Path(path).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                _DECODED[("image", path)] = _read_image(path)
            elif suffix in AUDIO_SUFFIXES:
                _DECODED[("audio", path)] = _read_audio(path)
            else:
                _DECODED[("bytes", path)] = Path(path).read_bytes()


def release_inputs() -> None:
    _DECODED.clear()


def load_image(path: str):
    cached = _DECODED.get(("image", path))
    return cached if cached is not None else _read_image(path)


def load_bytes(path: str) -> bytes:
    """An input file's contents (read before the timed call when the request names it)."""
    cached = _DECODED.get(("bytes", path))
    return cached if cached is not None else Path(path).read_bytes()


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    cached = _DECODED.get(("audio", path))
    audio, rate = cached if cached is not None else _read_audio(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)
    return audio


def invocation(observation: Mapping[str, Any], model_call_ms: float, **extra: Any) -> Invocation:
    return Invocation(observation=dict(observation), model_call_ms=model_call_ms, extra=extra)


class NativeHost:
    """What this backend hands a family's native pipeline: ``Adapter(spec, host)``. The family file imports
    nothing from this package (a family must not depend on an application); it reaches the backend's
    model-agnostic mechanics through ``host``: the request's fields and its input files, decoded before the timed
    call (``required``, ``load_image``, ``load_audio``, ``load_bytes``), the timed model call (``timed``), output
    tensors and files written after it (``tensor_observation``, ``deferred_files``), the result
    (``invocation``), and the rejection of a request (``raise host.Error(...)``)."""

    Error = BackendError
    required = staticmethod(required)
    timed = staticmethod(timed)
    load_image = staticmethod(load_image)
    load_audio = staticmethod(load_audio)
    load_bytes = staticmethod(load_bytes)
    tensor_observation = staticmethod(tensor_observation)
    deferred_files = staticmethod(deferred_files)
    invocation = staticmethod(invocation)
