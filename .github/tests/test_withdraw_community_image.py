# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
import urllib.error
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


def test_delete_diagnostic_retains_only_bounded_redacted_message(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    json.dumps(
                        {
                            "message": "Policy rejection test-token\n" + "x" * 900,
                            "secret": "DO NOT PRINT",
                        }
                    ).encode()
                ),
            )

    monkeypatch.setattr(MODULE.urllib.request, "build_opener", lambda *_: Opener())
    with pytest.raises(RuntimeError) as failure:
        MODULE.api("test-token", "/versions/1356372450", method="DELETE")
    message = str(failure.value)
    assert "HTTP 400: Policy rejection [redacted]" in message
    assert "test-token" not in message and "DO NOT PRINT" not in message
    assert "\n" not in message and len(message) < 600


def whole_api(
    monkeypatch,
    *,
    other=False,
    changed=False,
    error_status=400,
    error_message=None,
    remains=False,
    control_error=False,
    other_deleted=False,
):
    rows = [{"id": MODULE.ROOT_VERSION_ID, "name": MODULE.ROOT_DIGEST}]
    rows += [
        {"id": 100 + index, "name": digest}
        for index, digest in enumerate(sorted(MODULE.DIGESTS - {MODULE.ROOT_DIGEST}))
    ]
    if other:
        rows.append({"id": 999, "name": "sha256:" + "a" * 64})
    calls = []
    deleted = False
    reads = 0

    def api(token, suffix="", *, method="GET", control=False):
        nonlocal deleted, reads
        assert token == "test-token"
        calls.append((method, suffix, control))
        if control:
            if control_error:
                raise MODULE.PackageAPIError("GET", 403)
            return []
        if suffix == "":
            if method == "DELETE":
                deleted = True
                rows.clear()
                return None
            reads += 1
            if deleted and not remains:
                raise MODULE.PackageAPIError("GET", 404)
            return {
                "id": MODULE.PACKAGE_ID + int(changed and reads > 1),
                "name": MODULE.PACKAGE,
                "repository": {"id": MODULE.REPOSITORY_ID},
            }
        if suffix.startswith("/versions?state=deleted"):
            return [{"id": 998, "name": "sha256:" + "b" * 64}] if other_deleted else []
        if suffix.startswith("/versions?"):
            return [row.copy() for row in rows]
        row = next(row for row in rows if suffix == f"/versions/{row['id']}")
        if method == "DELETE":
            raise MODULE.PackageAPIError(
                "DELETE",
                error_status,
                MODULE.LAST_TAGGED_MESSAGE if error_message is None else error_message,
            )
        return row.copy()

    monkeypatch.setattr(MODULE, "api", api)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    return rows, calls


def test_last_tagged_version_with_children_withdraws_whole_package_once(monkeypatch):
    rows, calls = whole_api(monkeypatch)
    receipt = MODULE.withdraw("test-token")
    assert receipt["whole_package_deleted"] is True
    assert receipt["package_delete_http_status"] == 204
    assert receipt["authenticated_absence_confirmations"] == 2
    assert {row["digest"] for row in receipt["deleted"]} == MODULE.DIGESTS
    assert rows == []
    assert calls.count(("DELETE", "", False)) == 1
    position = calls.index(("DELETE", "", False))
    assert calls[position + 1 :] == [("GET", "", False), ("GET", "", True)] * 2


@pytest.mark.parametrize("fault", ["other", "changed", "other_deleted"])
def test_whole_package_fallback_revalidates_scope_before_delete(monkeypatch, fault):
    _, calls = whole_api(monkeypatch, **{fault: True})
    with pytest.raises(RuntimeError):
        MODULE.withdraw("test-token")
    assert ("DELETE", "", False) not in calls


@pytest.mark.parametrize(
    "status,message",
    [
        (403, MODULE.LAST_TAGGED_MESSAGE),
        (400, "Publicly visible package versions with more than 5000 downloads cannot be deleted."),
        (400, MODULE.LAST_TAGGED_MESSAGE + " unexpected detail"),
    ],
)
def test_only_exact_last_tagged_400_permits_fallback(monkeypatch, status, message):
    _, calls = whole_api(monkeypatch, error_status=status, error_message=message)
    with pytest.raises(MODULE.PackageAPIError):
        MODULE.withdraw("test-token")
    assert ("DELETE", "", False) not in calls


@pytest.mark.parametrize("fault", ["remains", "control_error"])
def test_whole_delete_requires_two_absences_and_valid_control(monkeypatch, fault):
    _, calls = whole_api(monkeypatch, **{fault: True})
    with pytest.raises(RuntimeError):
        MODULE.withdraw("test-token")
    assert calls.count(("DELETE", "", False)) == 1


@pytest.mark.parametrize(
    "rows", [[], [{"id": 123, "name": None}], [{"id": 123.0, "name": MODULE.ROOT_DIGEST}]]
)
def test_invalid_whole_inventory_never_deletes_package(monkeypatch, rows):
    _, calls = whole_api(monkeypatch)
    monkeypatch.setattr(MODULE, "inventory", lambda _, **kwargs: rows)
    with pytest.raises(RuntimeError):
        MODULE.withdraw_whole_package("test-token")
    assert ("DELETE", "", False) not in calls
