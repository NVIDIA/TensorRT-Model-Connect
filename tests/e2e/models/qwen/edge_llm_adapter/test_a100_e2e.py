# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Model Connect E2E for the supported Qwen EdgeLLM profile."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect.optimized_runtime.target import (
    TargetResolutionError,
    _probe_current_target_with_device,
)


_MODEL_ID = "Qwen/Qwen3-0.6B"


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _require_supported_a100() -> None:
    try:
        target, _ = _probe_current_target_with_device()
    except TargetResolutionError as exc:
        pytest.skip(f"the active CUDA target is unavailable: {exc}")
    if target["gpu_name"] != "NVIDIA A100 80GB PCIe":
        pytest.skip("the EdgeLLM profile requires NVIDIA A100 80GB PCIe")


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
def test_public_build_inspect_and_run_delegate_to_edgellm(tmp_path: Path) -> None:
    """Build and execute through unchanged public commands on the supported GPU."""

    _require_supported_a100()
    binary_value = os.environ.get("TRTMC_BINARY", "").strip()
    if not binary_value:
        pytest.skip("TRTMC_BINARY is set by the model CI runtime")
    binary = Path(binary_value).resolve(strict=True)

    bundle = tmp_path / "qwen3-0.6b-edge.trtfb"
    cache = tmp_path / "runtime-cache"
    output = tmp_path / "generated.jsonl"

    _run([str(binary), "build", _MODEL_ID, "-o", str(bundle)], timeout=21_600)
    inspect = _run([str(binary), "inspect", str(bundle)], timeout=60)
    assert "optimized_runtime.json" in inspect.stdout
    assert "optimized_runtime_artifacts/engine.dir/" in inspect.stdout

    _run(
        [
            str(binary),
            "run",
            str(bundle),
            "--prompt",
            "Reply with one short sentence about accelerated computing.",
            "--greedy",
            "--top-p",
            "1",
            "--top-k",
            "1",
            "--no-thinking",
            "--max-new-tokens",
            "32",
            "--runtime-cache",
            str(cache),
            "--output",
            str(output),
        ],
        timeout=600,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["generated"].strip()
    assert any(path.is_dir() for path in cache.rglob("engine.dir"))
