"""Text model reference runner abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ReferenceRunRequest:
    """Inputs needed to run a source-framework reference."""

    model_id: str
    prompt: str
    revision: str = ""
    seed: int | None = None


@dataclass(frozen=True)
class ReferenceRunResult:
    """Normalized source-framework output."""

    output: object
    metadata: dict[str, object]


class ReferenceRunner(Protocol):
    """Protocol for source-framework reference runners."""

    def run(self, request: ReferenceRunRequest) -> ReferenceRunResult:
        """Run the reference model and return normalized output."""


class TextReferenceRunner:
    """Placeholder text reference runner contract for CI mutation orchestration."""

    task_strategy = "text_generation_causal"

    def run(self, request: ReferenceRunRequest) -> ReferenceRunResult:
        raise NotImplementedError("Text reference execution is provided by tests/e2e_harness")
