# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core types for the unified diff test framework."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class DiffResult:
    """Result of a single diff test."""

    test_name: str
    model: str
    runtime_strategy: str
    passed: bool
    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    message: str
    metrics: dict = field(default_factory=dict)
    duration_s: float = 0.0
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "model": self.model,
            "runtime_strategy": self.runtime_strategy,
            "passed": self.passed,
            "status": self.status,
            "message": self.message,
            "metrics": self.metrics,
            "duration_s": round(self.duration_s, 2),
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def skip(test_name: str, model: str, runtime_strategy: str,
             reason: str) -> DiffResult:
        return DiffResult(
            test_name=test_name, model=model,
            runtime_strategy=runtime_strategy,
            passed=True, status="SKIP", message=reason)

    @staticmethod
    def error(test_name: str, model: str, runtime_strategy: str,
              message: str, details: str = "") -> DiffResult:
        return DiffResult(
            test_name=test_name, model=model,
            runtime_strategy=runtime_strategy,
            passed=False, status="ERROR", message=message, details=details)


@dataclass
class TestContext:
    """Shared context passed to all diff tests."""

    model: str
    runtime_strategy: str
    bundle_path: str | None = None
    binary_path: str | None = None
    hf_python: str | None = None
    image_path: str | None = None
    max_cache_length: int = 256
    max_new_tokens: int = 20
    atol: float = 1e-3
    layer_atol: float = 0.05
    trust_remote_code: bool = False
    verbose: bool = False
    num_inference_steps: int = 30


@runtime_checkable
class DiffTest(Protocol):
    """Interface for a diff test check."""

    name: str
    description: str
    runtime_strategies: list[str]
    requires_bundle: bool
    requires_gpu: bool

    def run(self, ctx: TestContext) -> DiffResult: ...
