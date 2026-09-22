# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command protocol shared by Accuracy and Performance reference execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def command(
    *,
    python: Path,
    generic_runner: Path,
    script: Path | None,
    family: str,
    operation: str,
    manifest: Path,
    selected_task: str | None,
    testcase_name: str | None,
    adapter: str | None,
    adapter_options: Mapping[str, Any],
    timing_contract: Mapping[str, Any],
    padding: str,
    model: str,
    revision: str | None,
    request: Mapping[str, Any],
    precision: str,
    mode: str,
    warmup: int,
    iterations: int,
    case_name: str,
    output: Path,
    trust_remote_code: bool,
    local_files_only: bool,
) -> list[str]:
    if (script is None) == (adapter is None):
        raise ValueError("reference requires exactly one of script or adapter")
    arguments = [
        str(python),
        str(script or generic_runner),
    ]
    if script is not None:
        arguments.extend(("--family", family))
    else:
        arguments.extend(("--adapter", str(adapter)))
    arguments.extend(
        (
            "--operation",
            operation,
            "--manifest",
            str(manifest),
            "--adapter-options-json",
            json.dumps(dict(adapter_options), ensure_ascii=True, separators=(",", ":")),
            "--timing-contract-json",
            json.dumps(dict(timing_contract), ensure_ascii=True, separators=(",", ":")),
            "--padding",
            padding,
            "--model",
            model,
            "--request-json",
            json.dumps(dict(request), ensure_ascii=True, separators=(",", ":")),
            "--precision",
            precision,
            "--mode",
            mode,
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--case-name",
            case_name,
            "--output",
            str(output),
        )
    )
    if script is None and testcase_name is not None:
        arguments.extend(("--testcase-name", testcase_name))
    if selected_task is not None:
        arguments.extend(("--selected-task", selected_task))
    if revision:
        arguments.extend(("--revision", revision))
    if trust_remote_code:
        arguments.append("--trust-remote-code")
    if local_files_only:
        arguments.append("--local-files-only")
    return arguments
