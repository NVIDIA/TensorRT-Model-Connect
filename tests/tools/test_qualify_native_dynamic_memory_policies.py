# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
MODULE_PATH = TOOLS_DIR / "qualify_native_dynamic_memory_policies.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_policies", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
policies = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = policies
SPEC.loader.exec_module(policies)

pytestmark = pytest.mark.dynamic_memory


def _case(name: str, token_ids: list[int]) -> dict:
    return {
        "name": name,
        "selected_token_ids": token_ids,
        "step_top1_token_ids": [11, *token_ids],
    }


def test_policy_comparison_requires_exact_tokens_and_full_logits() -> None:
    cases = [_case("auto", [7, 8]), _case("bytes", [7, 8])]
    logits = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    comparisons, passed = policies.compare_policy_outputs(
        cases, {"auto": logits, "bytes": logits.copy()}
    )

    assert passed
    assert comparisons == [
        {
            "reference": "auto",
            "candidate": "bytes",
            "selected_token_ids_equal": True,
            "step_top1_token_ids_equal": True,
            "full_float32_logits_equal": True,
            "passed": True,
        }
    ]


def test_policy_comparison_rejects_one_float32_bit_or_token_change() -> None:
    reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
    changed = reference.copy()
    changed.view(np.uint32)[0, 0] += 1

    comparisons, passed = policies.compare_policy_outputs(
        [_case("auto", [7]), _case("fraction", [8]), _case("bytes", [7])],
        {
            "auto": reference,
            "fraction": reference.copy(),
            "bytes": changed,
        },
    )

    assert not passed
    assert not comparisons[0]["selected_token_ids_equal"]
    assert not comparisons[1]["full_float32_logits_equal"]


def test_source_state_gate_is_fail_closed() -> None:
    pre = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    post = {"source_state_sha256": "a" * 64, "git_head": "b" * 40}
    report = {"passed": True}

    assert policies.apply_source_state_gate(report, pre, post)
    assert report["source_state_pre"] is pre
    assert report["source_state_post"] is post
    assert report["source_state_unchanged"] is True
    assert report["passed"] is True

    changed = {"source_state_sha256": "c" * 64, "git_head": "b" * 40}
    report = {"passed": True}
    assert not policies.apply_source_state_gate(report, pre, changed)
    assert report["source_state_unchanged"] is False
    assert report["passed"] is False

    report = {"passed": False}
    assert policies.apply_source_state_gate(report, pre, post)
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
        assert repo_root == policies.REPO_ROOT
        assert tool_path == Path(policies.__file__)
        calls.append((artifact_dir, label))
        return {"source_state_sha256": "a" * 64, "git_head": "b" * 40}

    monkeypatch.setattr(
        policies.boundary, "source_state_provenance", snapshot
    )
    external = tmp_path / "proof"
    policies._source_state_snapshot(external, label="pre")
    artifact = policies.REPO_ROOT / "artifacts" / "unit-policy-proof"
    policies._source_state_snapshot(artifact, label="post")

    assert calls == [
        (external.resolve(), "pre"),
        (artifact.resolve(), "post"),
    ]
    with pytest.raises(ValueError, match="source snapshots exclude it"):
        policies._source_state_snapshot(
            policies.REPO_ROOT / "unit-policy-proof",
            label="post",
        )
