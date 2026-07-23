# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for declarative, host-local model-reference cache warming."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_ENVIRONMENT = {
    "GIT_AUTHOR_NAME": "Model Reference Test",
    "GIT_AUTHOR_EMAIL": "model-reference-test@example.com",
    "GIT_COMMITTER_NAME": "Model Reference Test",
    "GIT_COMMITTER_EMAIL": "model-reference-test@example.com",
}


def _git(*arguments: str | Path, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *(str(argument) for argument in arguments)],
        cwd=cwd,
        env={**os.environ, **REVISION_ENVIRONMENT},
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _pinned_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "upstream"
    source.mkdir()
    _git("init", "--quiet", source)
    entrypoint = source / "reference.py"
    entrypoint.write_text("# pinned reference\n", encoding="utf-8")
    _git("-C", source, "add", "reference.py")
    _git("-C", source, "commit", "--quiet", "-m", "add pinned reference")
    revision = _git("-C", source, "rev-parse", "HEAD^{commit}")
    remote = tmp_path / "upstream.git"
    _git("clone", "--quiet", "--bare", source, remote)
    return source, remote, revision


def _write_owner(
    repository: Path,
    family: str,
    *,
    remote: str,
    revision: str,
    relative_path: str | None = None,
    entrypoint: str = "reference.py",
    suites: tuple[str, ...] | None = None,
) -> Path:
    owner = repository / "tests/e2e/models" / family / "MODEL.toml"
    owner.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"id = {json.dumps(family)}",
        "",
        "[model_reference_cache]",
        f"repository = {json.dumps(remote)}",
        f"revision = {json.dumps(revision)}",
        "relative_path = "
        + json.dumps(relative_path or f"{family}/reference/Source-{revision[:12]}"),
        f"entrypoint = {json.dumps(entrypoint)}",
    ]
    if suites is not None:
        lines.append("suites = " + json.dumps(list(suites)))
    owner.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return owner


def _warmer_environment(
    cache_root: Path,
    updates: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
        if part
    )
    environment["TRTMC_MODEL_REFERENCE_CACHE_ROOT"] = str(cache_root)
    environment.update(updates or {})
    return environment


def _warmer_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.ci",
        "model-reference-cache",
        "warm",
        "--suite",
        "nightly",
    ]


def _run_warmer(repository: Path, cache_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _warmer_command(),
        cwd=repository,
        env=_warmer_environment(cache_root),
        check=False,
        capture_output=True,
        text=True,
    )


def test_warmer_discovers_suite_contracts_and_reuses_exact_checkout(
    tmp_path: Path,
) -> None:
    _source, remote, revision = _pinned_remote(tmp_path)
    repository = tmp_path / "repository"
    cache_root = tmp_path / "cache"
    active_relative = f"active/reference/Source-{revision[:12]}"
    _write_owner(
        repository,
        "active",
        remote=remote.as_uri(),
        revision=revision,
        relative_path=active_relative,
        suites=("nightly",),
    )
    _write_owner(
        repository,
        "premerge_only",
        remote=remote.as_uri(),
        revision=revision,
        suites=("premerge",),
    )

    first = _run_warmer(repository, cache_root)

    assert first.returncode == 0, first.stderr
    destination = cache_root / active_relative
    assert _git("-C", destination, "rev-parse", "HEAD^{commit}") == revision
    assert _git("-C", destination, "config", "--get", "remote.origin.url") == remote.as_uri()
    assert (destination / "reference.py").is_file()
    assert not (cache_root / f"premerge_only/reference/Source-{revision[:12]}").exists()
    assert f"FETCHED {active_relative} @ {revision[:12]}" in first.stdout

    remote.rename(tmp_path / "offline-upstream.git")
    second = _run_warmer(repository, cache_root)

    assert second.returncode == 0, second.stderr
    assert f"CACHED {active_relative} @ {revision[:12]}" in second.stdout


def test_concurrent_warmers_publish_once_under_the_per_path_lock(
    tmp_path: Path,
) -> None:
    _source, remote, revision = _pinned_remote(tmp_path)
    repository = tmp_path / "repository"
    cache_root = tmp_path / "cache"
    relative_path = f"fixture/reference/Source-{revision[:12]}"
    _write_owner(
        repository,
        "fixture",
        remote=remote.as_uri(),
        revision=revision,
        relative_path=relative_path,
    )
    real_git = shutil.which("git")
    assert real_git is not None
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        f"""#!{sys.executable}
import os
from pathlib import Path
import sys
import time

arguments = sys.argv[1:]
if arguments and arguments[0] == "init":
    with Path(os.environ["FAKE_GIT_INIT_LOG"]).open("a", encoding="utf-8") as stream:
        stream.write("init\\n")
if "fetch" in arguments:
    with Path(os.environ["FAKE_GIT_FETCH_LOG"]).open("a", encoding="utf-8") as stream:
        stream.write("fetch\\n")
    Path(os.environ["FAKE_GIT_FETCH_STARTED"]).touch()
    deadline = time.monotonic() + 10
    while not Path(os.environ["FAKE_GIT_FETCH_RELEASE"]).exists():
        if time.monotonic() >= deadline:
            raise SystemExit("timed out waiting to release fake Git fetch")
        time.sleep(0.01)
os.execv({real_git!r}, [{real_git!r}, *arguments])
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    init_log = tmp_path / "init.log"
    fetch_log = tmp_path / "fetch.log"
    fetch_started = tmp_path / "fetch-started"
    fetch_release = tmp_path / "fetch-release"
    environment = _warmer_environment(
        cache_root,
        {
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GIT_INIT_LOG": str(init_log),
            "FAKE_GIT_FETCH_LOG": str(fetch_log),
            "FAKE_GIT_FETCH_STARTED": str(fetch_started),
            "FAKE_GIT_FETCH_RELEASE": str(fetch_release),
        },
    )
    first = subprocess.Popen(
        _warmer_command(),
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + 5
        while not fetch_started.exists() and first.poll() is None:
            assert time.monotonic() < deadline, "first warmer did not enter Git fetch"
            time.sleep(0.01)
        assert fetch_started.exists()
        second = subprocess.Popen(
            _warmer_command(),
            cwd=repository,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.5)
        assert first.poll() is None
        assert second.poll() is None
        assert init_log.read_text(encoding="utf-8").splitlines() == ["init"]
        assert fetch_log.read_text(encoding="utf-8").splitlines() == ["fetch"]
        fetch_release.touch()
        first_stdout, first_stderr = first.communicate(timeout=10)
        second_stdout, second_stderr = second.communicate(timeout=10)
    finally:
        fetch_release.touch()
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second_stderr
    output = first_stdout + second_stdout
    assert output.count(f"FETCHED {relative_path}") == 1
    assert output.count(f"CACHED {relative_path}") == 1
    assert init_log.read_text(encoding="utf-8").splitlines() == ["init"]
    assert fetch_log.read_text(encoding="utf-8").splitlines() == ["fetch"]


def test_warmer_fails_closed_instead_of_replacing_a_mismatched_checkout(
    tmp_path: Path,
) -> None:
    source, remote, first_revision = _pinned_remote(tmp_path)
    repository = tmp_path / "repository"
    cache_root = tmp_path / "cache"
    relative_path = f"fixture/reference/Source-{first_revision[:12]}"
    owner = _write_owner(
        repository,
        "fixture",
        remote=remote.as_uri(),
        revision=first_revision,
        relative_path=relative_path,
    )
    assert _run_warmer(repository, cache_root).returncode == 0

    (source / "reference.py").write_text("# changed reference\n", encoding="utf-8")
    _git("-C", source, "add", "reference.py")
    _git("-C", source, "commit", "--quiet", "-m", "change pinned reference")
    second_revision = _git("-C", source, "rev-parse", "HEAD^{commit}")
    text = owner.read_text(encoding="utf-8").replace(first_revision, second_revision)
    owner.write_text(text, encoding="utf-8")

    result = _run_warmer(repository, cache_root)

    assert result.returncode == 1
    assert "model reference cache revision mismatch" in result.stderr
    destination = cache_root / relative_path
    assert _git("-C", destination, "rev-parse", "HEAD^{commit}") == first_revision


@pytest.mark.parametrize(
    ("revision", "entrypoint", "error"),
    (
        ("f" * 40, "reference.py", "Command failed"),
        (None, "missing.py", "model reference cache entrypoint is absent"),
    ),
)
def test_warmer_never_publishes_an_incomplete_checkout(
    tmp_path: Path,
    revision: str | None,
    entrypoint: str,
    error: str,
) -> None:
    _source, remote, pinned_revision = _pinned_remote(tmp_path)
    selected_revision = revision or pinned_revision
    repository = tmp_path / "repository"
    cache_root = tmp_path / "cache"
    relative_path = f"fixture/reference/Source-{selected_revision[:12]}"
    _write_owner(
        repository,
        "fixture",
        remote=remote.as_uri(),
        revision=selected_revision,
        relative_path=relative_path,
        entrypoint=entrypoint,
    )

    result = _run_warmer(repository, cache_root)

    assert result.returncode == 1
    assert error in result.stderr
    destination = cache_root / relative_path
    assert not os.path.lexists(destination)
    assert list(destination.parent.glob(f".{destination.name}.*")) == []


def test_warmer_rejects_a_reference_path_outside_its_owner(
    tmp_path: Path,
) -> None:
    _source, remote, revision = _pinned_remote(tmp_path)
    repository = tmp_path / "repository"
    cache_root = tmp_path / "cache"
    _write_owner(
        repository,
        "fixture",
        remote=remote.as_uri(),
        revision=revision,
        relative_path="sibling/reference/source",
    )

    result = _run_warmer(repository, cache_root)

    assert result.returncode == 1
    assert "must be owned by the selected E2E family" in result.stderr
    assert not (cache_root / "sibling").exists()
