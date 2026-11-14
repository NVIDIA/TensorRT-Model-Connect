# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a benchmark catalog profile/testcase into serving inputs.

Using the same resolution as ``trtmc-bench`` keeps the served base request, the
bundle identity checks, and the load-generator payloads identical to the
one-shot performance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trtmc_benchmark.catalog import ManifestCatalog, resolve_case
from trtmc_benchmark.types import ModelDescriptor

_UNUSED_BUNDLE = Path("/nonexistent/reference-only.bundle")


@dataclass(frozen=True)
class ServingProfile:
    model: ModelDescriptor
    testcase: str
    operation: str
    worker_request: Mapping[str, Any]

    @property
    def base_request(self) -> Mapping[str, Any]:
        return self.worker_request["request"]


def resolve_profile(
    selector: str,
    *,
    manifest_root: Path | None,
    testcase: str | None = None,
    operation: str | None = None,
    selected_task: str | None = None,
    bundle: Path | None = None,
    runtime_root: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ServingProfile:
    model = ManifestCatalog(manifest_root).resolve(selector)
    case = resolve_case(model, bundle or _UNUSED_BUNDLE, case_name=testcase, operation=operation,
                        selected_task=selected_task, overrides=overrides)
    request = case.with_values(runtime_root=runtime_root or Path("/")).worker_request()
    if runtime_root is None:
        request.pop("runtime_root")
    return ServingProfile(model=model, testcase=case.testcase_name, operation=case.operation,
                          worker_request=request)
