"""Converted-runner abstraction for Model Connect bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConvertedRunRequest:
    """Inputs needed to run a converted bundle."""

    model_name: str
    bundle_path: Path
    trtmc_binary: Path
    prompt: str = ""


@dataclass(frozen=True)
class ConvertedRunResult:
    """Normalized converted-model output."""

    output: object
    command: tuple[str, ...]
    metadata: dict[str, object]


class ModelConnectRunner:
    """Runner contract for converted TRTMC bundles."""

    def run(self, request: ConvertedRunRequest) -> ConvertedRunResult:
        raise NotImplementedError("Converted execution is provided by tests/e2e_harness")
