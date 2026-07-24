# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
MODULE_PATH = TOOLS_DIR / "qualify_native_dynamic_memory_surfaces.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_surfaces", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
surfaces = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = surfaces
SPEC.loader.exec_module(surfaces)

pytestmark = pytest.mark.dynamic_memory


def _receipt(**overrides: object) -> dict:
    receipt = {
        field: index for index, field in enumerate(
            surfaces.RECEIPT_EQUIVALENCE_FIELDS
        )
    }
    receipt.update(
        {
            "runtime_kv_capacity_tokens": 512,
            "peak_device_bytes": 1234,
            "peak_device_bytes_scope": "device_wide",
            "peak_device_sample_count": 2,
            "peak_device_sample_boundaries": [
                "after_runtime_kv_allocation",
                "after_successful_request_completion",
            ],
        }
    )
    receipt.update(overrides)
    return receipt


def test_surface_comparison_accepts_equal_resolution_with_different_peaks() -> None:
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(peak_device_bytes=1000),
        },
        {
            "status": "accepted",
            "surface": "cpp",
            "runtime_memory_receipt": _receipt(peak_device_bytes=2000),
        },
        {
            "status": "accepted",
            "surface": "cabi",
            "runtime_memory_receipt": _receipt(peak_device_bytes=3000),
        },
        {
            "status": "accepted",
            "surface": "python",
            "runtime_memory_receipt": _receipt(peak_device_bytes=4000),
        },
    ]

    comparison, passed = surfaces.compare_surface_receipts(results)

    assert passed
    assert set(comparison) == {"cli", "cpp", "cabi", "python"}
    assert all(item["passed"] for item in comparison.values())


def test_surface_comparison_rejects_R_receipt_or_request_peak_drift() -> None:
    results = [
        {
            "status": "accepted",
            "surface": "cli",
            "runtime_memory_receipt": _receipt(),
        },
        {
            "status": "accepted",
            "surface": "cpp",
            "runtime_memory_receipt": _receipt(
                runtime_kv_capacity_tokens=511
            ),
        },
        {
            "status": "accepted",
            "surface": "python",
            "runtime_memory_receipt": _receipt(
                peak_device_sample_boundaries=[
                    "after_runtime_kv_allocation"
                ]
            ),
        },
    ]

    comparison, passed = surfaces.compare_surface_receipts(results)

    assert not passed
    assert not comparison["cpp"]["resolved_R_is_512"]
    assert (
        "runtime_kv_capacity_tokens"
        in comparison["cpp"]["receipt_mismatches"]
    )
    assert not comparison["python"]["request_complete_peak"]


def test_source_state_gate_is_fail_closed() -> None:
    pre = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    post = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    report = {"passed": True}

    assert surfaces.apply_source_state_gate(report, pre, post)
    assert report["source_state_pre"] is pre
    assert report["source_state_post"] is post
    assert report["source_state_unchanged"] is True
    assert report["passed"] is True

    changed = {"source_state_sha256": "c" * 64, "git_head": "b" * 40}
    report = {"passed": True}
    assert not surfaces.apply_source_state_gate(report, pre, changed)
    assert report["source_state_unchanged"] is False
    assert report["passed"] is False

    report = {"passed": False}
    assert surfaces.apply_source_state_gate(report, pre, post)
    assert report["passed"] is False


def test_source_snapshot_excludes_artifact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str]] = []

    def snapshot(
        repo_root: Path,
        tool_path: Path,
        artifact_dir: Path,
        *,
        label: str,
    ) -> dict:
        assert repo_root == surfaces.REPO_ROOT
        assert tool_path == Path(surfaces.__file__)
        calls.append((artifact_dir, label))
        return {"source_state_sha256": "a" * 64, "git_head": "b" * 40}

    monkeypatch.setattr(
        surfaces.boundary, "source_state_provenance", snapshot
    )
    external = tmp_path / "proof"
    surfaces._source_state_snapshot(external, label="pre")
    artifact = surfaces.REPO_ROOT / "artifacts" / "unit-surface-proof"
    surfaces._source_state_snapshot(artifact, label="post")

    assert calls == [
        (external.resolve(), "pre"),
        (artifact.resolve(), "post"),
    ]
    with pytest.raises(ValueError, match="source snapshots exclude it"):
        surfaces._source_state_snapshot(
            surfaces.REPO_ROOT / "unit-surface-proof",
            label="post",
        )
