# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public Model Connect E2E for the supported Qwen EdgeLLM profile."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tensorrt_model_connect.runtime_provider.target import (
    TargetResolutionError,
    _probe_current_target_with_device,
)


_MODEL_ID = "Qwen/Qwen3-0.6B"
_IMPLEMENTATION_ID = "qwen3-0.6b-fp16.tensorrt-edge-llm.a100-pcie80-sm80"


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
        pytest.fail(f"the selected A100 proof could not inspect its CUDA target: {exc}")
    if target["gpu_name"] != "NVIDIA A100 80GB PCIe":
        pytest.fail(
            f"the selected A100 proof requires NVIDIA A100 80GB PCIe; found {target['gpu_name']}"
        )


def _read_bundle_header(bundle: Path) -> tuple[int, dict[str, Any]]:
    with bundle.open("rb") as stream:
        assert stream.read(8) == b"TRTFB\x00\x01\x00"
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return header_size, header


def _read_bundle_section(bundle: Path, name: str) -> bytes:
    header_size, header = _read_bundle_header(bundle)
    with bundle.open("rb") as stream:
        section = header["sections"][name]
        stream.seek(16 + header_size + section["offset"])
        return stream.read(section["size"])


def _tree_inventory(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.slow
def test_public_build_inspect_and_run_delegate_to_edgellm(tmp_path: Path) -> None:
    """Build and execute through unchanged public commands on the supported GPU."""

    _require_supported_a100()
    binary_value = os.environ.get("TRTMC_BINARY", "").strip()
    if not binary_value:
        pytest.fail("TRTMC_BINARY is required for the A100 EdgeLLM E2E")
    binary = Path(binary_value).resolve(strict=True)

    bundle = tmp_path / "qwen3-0.6b-edge.trtfb"
    cache = tmp_path / "runtime-cache"
    cold_output = tmp_path / "generated-cold.jsonl"
    warm_output = tmp_path / "generated-warm.jsonl"

    _run(
        [
            str(binary),
            "build",
            _MODEL_ID,
            "-o",
            str(bundle),
            "--precision",
            "fp16",
            "--max-cache-length",
            "4096",
            "--max-batch-size",
            "4",
        ],
        timeout=21_600,
    )
    _header_size, header = _read_bundle_header(bundle)
    assert header["model_type"] == "qwen3"
    assert header["family"] == "qwen"
    descriptor = json.loads(_read_bundle_section(bundle, "optimized_runtime.json"))
    assert descriptor["implementation_id"] == _IMPLEMENTATION_ID
    expected_cache = (
        cache
        / "optimized-runtimes"
        / _IMPLEMENTATION_ID
        / f"{descriptor['profile_id']}-{descriptor['artifact']['tree_sha256']}"
    )
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
            str(cold_output),
        ],
        timeout=600,
    )
    assert expected_cache.is_dir()
    assert (expected_cache / "engine.dir").is_dir()
    cold_inventory = _tree_inventory(expected_cache)
    assert cold_inventory

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
            str(warm_output),
        ],
        timeout=600,
    )
    assert _tree_inventory(expected_cache) == cold_inventory

    cold_rows = [
        json.loads(line) for line in cold_output.read_text(encoding="utf-8").splitlines()
    ]
    warm_rows = [
        json.loads(line) for line in warm_output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(cold_rows) == len(warm_rows) == 1
    assert cold_rows[0]["generated"].strip()
    assert cold_rows[0]["token_ids"]
    assert warm_rows[0]["token_ids"] == cold_rows[0]["token_ids"]
