# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python reference backend: one adapter per operation, same request schema as the worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..base import BackendUnavailable, Invocation
from .common import MODES, ReferenceSpec

# Operations whose references need upstream checkouts (PersonaPlex, MoGe, LeRobot,
# Fast-FoundationStereo, Boltz, ...) are served only by the TRTMC backend for now.
_ADAPTERS = {
    "generate": ("text", "TextGeneration"),
    "translate": ("text", "Translation"),
    "encode": ("text", "TextEncoder"),
    "embed": ("text", "TextEncoder"),
    "rerank": ("text", "Reranker"),
    "classify": ("media", "Vision"),
    "detect": ("media", "Vision"),
    "segment": ("media", "Vision"),
    "segment_prompted": ("media", "Vision"),
    "extract_features": ("media", "Vision"),
    "transcribe": ("media", "SpeechRecognition"),
    "generate_audio": ("media", "SpeechSynthesis"),
    "generate_image": ("media", "Diffusion"),
    "solve": ("media", "TimeSeries"),
    "regress": ("media", "TimeSeries"),
}


def supported_operations() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


class ReferenceBackend:
    def __init__(self, spec: ReferenceSpec) -> None:
        if spec.operation not in _ADAPTERS:
            raise BackendUnavailable(
                f"no Python reference adapter for {spec.operation!r}; supported: {', '.join(supported_operations())}")
        if spec.mode not in MODES:
            raise BackendUnavailable(f"mode must be one of {MODES}")
        import importlib

        module_name, class_name = _ADAPTERS[spec.operation]
        module = importlib.import_module(f"{__name__}.{module_name}")
        self.operation = spec.operation
        self._spec = spec
        self._numerics: dict[str, Any] = {"deterministic": False}
        if spec.deterministic:  # before loading so kernels are selected under these settings
            from ...platform import apply_deterministic_numerics

            self._numerics = {"deterministic": True, **apply_deterministic_numerics()}
        self._adapter = getattr(module, class_name)(spec)

    def describe(self) -> Mapping[str, Any]:
        spec = self._spec
        return {"backend": "reference", "operation": spec.operation, "model": spec.model,
                "revision": spec.revision, "precision": spec.precision, "mode": spec.mode,
                "adapter": type(self._adapter).__name__, "timing_scope": "task-call-wall",
                "input_preparation_included": True, "numerics": dict(self._numerics)}

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        """Timed like the TRTMC public Task call: the whole request (input preparation, model call, and
        output decoding); the adapter's model-only time stays in ``model_only_ms``."""
        import time

        from .common import synchronize

        synchronize()
        started = time.perf_counter()
        invocation = self._adapter.invoke(request, artifact_base)
        synchronize()
        total_ms = (time.perf_counter() - started) * 1000.0
        return Invocation(invocation.observation, total_ms,
                          {**dict(invocation.extra or {}), "model_only_ms": invocation.model_call_ms})

    def close(self) -> None:
        self._adapter = None
