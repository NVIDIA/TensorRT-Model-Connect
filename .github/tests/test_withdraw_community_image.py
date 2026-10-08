# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "withdraw_community_image", Path(__file__).parents[1] / "scripts/withdraw_community_image.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_api(monkeypatch, *, wrong_package=False, wrong_root=False):
    rows = [
        {"id": MODULE.ROOT_VERSION_ID + int(wrong_root), "name": MODULE.ROOT_DIGEST},
        *[
            {"id": 100 + index, "name": digest}
            for index, digest in enumerate(sorted(MODULE.DIGESTS - {MODULE.ROOT_DIGEST}))
        ],
        {"id": 999, "name": "sha256:" + "a" * 64},
    ]
    deleted = []

    def api(token, suffix="", *, method="GET"):
        assert token == "test-token"
        if suffix == "":
            return {
                "id": MODULE.PACKAGE_ID + int(wrong_package),
                "name": MODULE.PACKAGE,
                "repository": {"id": MODULE.REPOSITORY_ID},
            }
        if suffix.startswith("/versions?"):
            return [row.copy() for row in rows]
        row = next(row for row in rows if suffix == f"/versions/{row['id']}")
        if method == "DELETE":
            deleted.append(row.copy())
            rows.remove(row)
            return None
        return row.copy()

    monkeypatch.setattr(MODULE, "api", api)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    return rows, deleted


def test_withdraws_only_index_and_children_and_preserves_other_versions(monkeypatch):
    rows, deleted = fixture_api(monkeypatch)
    receipt = MODULE.withdraw("test-token")
    assert deleted[0]["name"] == MODULE.ROOT_DIGEST
    assert {row["name"] for row in deleted} == MODULE.DIGESTS
    assert rows == [{"id": 999, "name": "sha256:" + "a" * 64}]
    assert receipt["authenticated_absence_confirmations"] == 2
    assert MODULE.withdraw("test-token")["deleted"] == []


@pytest.mark.parametrize("mismatch", ["wrong_package", "wrong_root"])
def test_identity_mismatch_prevents_any_deletion(monkeypatch, mismatch):
    _, deleted = fixture_api(monkeypatch, **{mismatch: True})
    with pytest.raises(RuntimeError, match="identity"):
        MODULE.withdraw("test-token")
    assert deleted == []


def test_api_failure_does_not_count_as_confirmed_deletion(monkeypatch):
    def failing_api(*args, **kwargs):
        raise RuntimeError("Package API GET failed with HTTP 403")

    monkeypatch.setattr(MODULE, "api", failing_api)
    with pytest.raises(RuntimeError, match="403"):
        MODULE.withdraw("test-token")
