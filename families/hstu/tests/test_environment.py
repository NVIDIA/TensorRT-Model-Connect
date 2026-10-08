# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The selected oracle checkout must remain pinned, including cached sources."""

import json
from pathlib import Path
import subprocess

import pytest

from families.hstu.tests import environment, reference


def test_reference_declaration_and_provenance_match_the_oracle():
    root = Path(__file__).resolve().parent
    declaration = json.loads((root / "reference-source.json").read_text())
    provenance = json.loads((root / "reference-provenance.json").read_text())
    assert declaration == {
        "repository": "NVIDIA/recsys-examples", "revision": reference.REFERENCE_REVISION,
    }
    assert environment._SOURCE_URL == f"https://github.com/{declaration['repository']}.git"
    assert provenance["repository"] == f"https://github.com/{declaration['repository']}"
    assert provenance["revision"] == declaration["revision"]


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *arguments], check=True,
        text=True, capture_output=True,
    ).stdout.strip()


@pytest.fixture
def local_source(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "--quiet")
    for name in reference.SOURCE_FILES:
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Pinned source fixture\n", encoding="utf-8")
    _git(upstream, "add", ".")
    _git(upstream, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
         "commit", "--quiet", "-m", "test: seed reference fixture")
    revision = _git(upstream, "rev-parse", "HEAD")
    monkeypatch.setattr(environment, "_SOURCE_URL", upstream.as_uri())
    monkeypatch.setattr(environment, "REFERENCE_REVISION", revision)
    monkeypatch.setattr(reference, "REFERENCE_REVISION", revision)
    monkeypatch.delenv("TRTMC_HSTU_REFERENCE_ROOT", raising=False)
    monkeypatch.delenv("TRTMC_REFERENCE_SOURCE_DIR", raising=False)
    monkeypatch.setenv("TRTMC_HSTU_REFERENCE_CACHE", str(tmp_path / "cache"))
    return upstream


def test_prepares_exact_revision_and_reuses_verified_cache(local_source, monkeypatch):
    prepared = environment.reference_source()
    assert prepared != local_source
    assert _git(prepared, "rev-parse", "HEAD") == _git(local_source, "rev-parse", "HEAD")
    # Reuse is local: the fetch source need not remain available.
    monkeypatch.setattr(environment, "_SOURCE_URL", "/unavailable-reference")
    assert environment.reference_source() == prepared


def test_changed_cached_oracle_is_rejected(local_source):
    prepared = environment.reference_source()
    path = prepared / reference.SOURCE_FILES[0]
    path.write_text("# Unexpected source change\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the pinned"):
        environment.reference_source()
    assert path.read_text(encoding="utf-8") == "# Unexpected source change\n"


@pytest.mark.parametrize("variable", ["TRTMC_HSTU_REFERENCE_ROOT", "TRTMC_REFERENCE_SOURCE_DIR"])
def test_explicit_source_is_verified_without_replacement(local_source, monkeypatch, variable):
    monkeypatch.setenv(variable, str(local_source))
    monkeypatch.setattr(environment, "_SOURCE_URL", "/unavailable-reference")
    assert environment.reference_source() == local_source
    assert not (local_source.parent / "cache").exists()
    path = local_source / reference.SOURCE_FILES[0]
    path.write_text("# Changed explicit checkout\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the pinned"):
        environment.reference_source()
    assert not (local_source.parent / "cache").exists()


def test_family_source_override_precedes_shared_ci_source(local_source, monkeypatch):
    monkeypatch.setenv("TRTMC_HSTU_REFERENCE_ROOT", str(local_source))
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", "/unavailable-shared-source")
    monkeypatch.setattr(environment, "_SOURCE_URL", "/unavailable-reference")
    assert environment.reference_source() == local_source
    assert not (local_source.parent / "cache").exists()


def test_fetch_failure_is_an_error_not_a_skipped_reference(local_source, monkeypatch):
    monkeypatch.setattr(environment, "_SOURCE_URL", "/missing-reference-repository")
    with pytest.raises(RuntimeError, match="Cannot prepare pinned HSTU reference"):
        environment.reference_source()
