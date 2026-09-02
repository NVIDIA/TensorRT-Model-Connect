# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools import case_evidence


REVISION_A = "a" * 40
REVISION_B = "b" * 40


def test_stamp_case_rejects_conflicting_source_revision() -> None:
    with pytest.raises(case_evidence.CaseEvidenceError, match="source revision conflict"):
        case_evidence.stamp_case(
            {"id": "model-a::accuracy", "source_revision": REVISION_A},
            REVISION_B,
        )


def test_model_revision_summary_allows_different_models_on_different_revisions() -> None:
    summary = case_evidence.summarize_model_revisions(
        [
            {
                "id": "model-a::accuracy",
                "model": "model-a",
                "task": "accuracy",
                "state": "terminal",
                "source_revision": REVISION_A,
            },
            {
                "id": "model-a::perf",
                "model": "model-a",
                "task": "perf",
                "state": "terminal",
                "source_revision": REVISION_A,
            },
            {
                "id": "model-b::accuracy",
                "model": "model-b",
                "task": "accuracy",
                "state": "terminal",
                "source_revision": REVISION_B,
            },
            {
                "id": "model-b::perf",
                "model": "model-b",
                "task": "perf",
                "state": "terminal",
                "source_revision": REVISION_B,
            },
        ]
    )

    assert summary["consistent"] is True
    assert summary["source_revisions"] == [REVISION_A, REVISION_B]
    assert summary["models"]["model-a"]["source_revision"] == REVISION_A
    assert summary["models"]["model-b"]["source_revision"] == REVISION_B


def test_model_revision_summary_rejects_mixed_revisions_within_one_model() -> None:
    summary = case_evidence.summarize_model_revisions(
        [
            {
                "id": "model-a::accuracy",
                "model": "model-a",
                "task": "accuracy",
                "state": "terminal",
                "source_revision": REVISION_A,
            },
            {
                "id": "model-a::perf",
                "model": "model-a",
                "task": "perf",
                "state": "terminal",
                "source_revision": REVISION_B,
            },
        ]
    )

    assert summary["consistent"] is False
    assert summary["models"]["model-a"] == {
        "status": "mixed",
        "source_revision": None,
        "source_revisions": [REVISION_A, REVISION_B],
        "case_ids": ["model-a::accuracy", "model-a::perf"],
        "tasks": ["accuracy", "perf"],
    }


def test_model_revision_summary_rejects_terminal_case_without_provenance() -> None:
    summary = case_evidence.summarize_model_revisions(
        [
            {
                "id": "legacy",
                "model": "model-a",
                "task": "accuracy",
                "state": "terminal",
            }
        ]
    )

    assert summary["consistent"] is False
    assert summary["models"]["model-a"]["status"] == "missing"
    assert summary["models"]["model-a"]["source_revision"] is None
