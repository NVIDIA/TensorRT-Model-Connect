"""Vision reference runner abstraction."""

from __future__ import annotations

from model_connect_ci.runners.reference.text_runner import (
    ReferenceRunRequest,
    ReferenceRunResult,
)


class VisionReferenceRunner:
    """Placeholder vision reference runner contract."""

    task_strategy = "image_classification"

    def run(self, request: ReferenceRunRequest) -> ReferenceRunResult:
        raise NotImplementedError("Vision reference execution is provided by tests/e2e_harness")
