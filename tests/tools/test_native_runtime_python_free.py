# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail closed if native bundle inference grows a Python interpreter bridge."""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY = Path(__file__).resolve().parents[2]
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".toml"}


def _native_runtime_files() -> list[Path]:
    roots = (
        REPOSITORY / "include" / "trtmc",
        REPOSITORY / "src" / "runtime",
        REPOSITORY / "src" / "cabi",
    )
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and (path.suffix in NATIVE_SUFFIXES or path.name == "CMakeLists.txt")
    ]
    return sorted(set(files))


def test_native_runtime_contains_no_python_interpreter_bridge() -> None:
    """The build CLI may use Python; loaded bundle execution may not."""
    forbidden_literals = (
        "hf_python",
        "--hf-python",
        "python interpreter",
        "python component",
        "python talker",
        "python tokenizer bridge",
        "/usr/bin/python",
        "python3",
        "libpython",
    )
    forbidden_calls = re.compile(
        r"\b(?:fork|popen|posix_spawn|execv|execve|execvp|execl|execlp|system)\s*\(|"
        r"\bPy_(?:Initialize|Run|Import|GIL)"
    )
    violations: list[str] = []

    for path in _native_runtime_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for marker in forbidden_literals:
            if marker.lower() in lowered:
                violations.append(f"{path.relative_to(REPOSITORY)}: forbidden {marker!r}")
        for match in forbidden_calls.finditer(text):
            violations.append(
                f"{path.relative_to(REPOSITORY)}: forbidden process/interpreter call "
                f"{match.group(0)!r}"
            )

    public_runtime_helpers = (
        REPOSITORY / "python" / "tensorrt_model_connect" / "pipeline.py",
        REPOSITORY / "examples" / "trtmc_benchmark_worker.cpp",
        REPOSITORY / "examples" / "trtmc_dataset_benchmark.cpp",
        REPOSITORY / "src" / "cli" / "args.cpp",
        REPOSITORY / "src" / "cli" / "args.h",
    )
    for path in public_runtime_helpers:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in ("hf_python", "--hf-python", "python interpreter"):
            if marker in text:
                violations.append(f"{path.relative_to(REPOSITORY)}: forbidden {marker!r}")

    assert not violations, "\n".join(violations)
