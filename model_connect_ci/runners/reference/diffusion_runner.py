"""Diffusion reference runner abstraction."""

from __future__ import annotations

from model_connect_ci.runners.reference.text_runner import (
    ReferenceRunRequest,
    ReferenceRunResult,
)


class DiffusionReferenceRunner:
    """Placeholder diffusion reference runner contract."""

    task_strategy = "diffusion_media_generation"

    def run(self, request: ReferenceRunRequest) -> ReferenceRunResult:
        raise NotImplementedError("Diffusion reference execution is provided by tests/e2e_harness")
