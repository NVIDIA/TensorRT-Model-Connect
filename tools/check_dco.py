#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require every introduced commit to carry its author's DCO sign-off."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


_SIGN_OFF = re.compile(
    r"^Signed-off-by:\s*(?P<name>.+?)\s*<(?P<email>[^<>]+)>\s*$",
    re.IGNORECASE,
)


class DcoError(RuntimeError):
    """Raised when the requested Git commit range cannot be inspected."""


def _git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if result.returncode:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise DcoError(stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout


def introduced_commits(repository: Path, base: str, head: str) -> list[str]:
    """Return every commit reachable from head but not from the target base."""

    output = _git(repository, "rev-list", "--reverse", f"{base}..{head}")
    assert isinstance(output, str)
    commits = output.splitlines()
    if not commits:
        raise DcoError(f"no introduced commits found in {base}..{head}")
    return commits


def _normalize_name(value: str) -> str:
    return " ".join(value.split()).casefold()


def _normalize_email(value: str) -> str:
    return value.strip().casefold()


def check_commit(repository: Path, revision: str) -> str | None:
    """Return a human-readable failure when revision lacks its author's sign-off."""

    payload = _git(repository, "show", "-s", "--format=%an%x00%ae%x00%B", revision, binary=True)
    assert isinstance(payload, bytes)
    fields = payload.decode("utf-8", errors="replace").split("\0", maxsplit=2)
    if len(fields) != 3:
        raise DcoError(f"could not read author and message for {revision}")
    author_name, author_email, message = fields
    expected_name = _normalize_name(author_name)
    expected_email = _normalize_email(author_email)
    signoffs: list[tuple[str, str]] = []
    for line in message.splitlines():
        match = _SIGN_OFF.fullmatch(line.strip())
        if match:
            signoffs.append((match.group("name"), match.group("email")))
    if any(
        _normalize_name(name) == expected_name and _normalize_email(email) == expected_email
        for name, email in signoffs
    ):
        return None
    short = revision[:12]
    expected = f"{author_name.strip()} <{author_email.strip()}>"
    if not signoffs:
        return f"{short}: missing Signed-off-by: {expected}"
    rendered = ", ".join(f"{name.strip()} <{email.strip()}>" for name, email in signoffs)
    return f"{short}: Signed-off-by does not match author {expected}; found {rendered}"


def check_range(repository: Path, base: str, head: str) -> tuple[list[str], list[str]]:
    commits = introduced_commits(repository, base, head)
    failures = [failure for revision in commits if (failure := check_commit(repository, revision))]
    return commits, failures


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    repository = arguments.repo_root.resolve()
    try:
        commits, failures = check_range(repository, arguments.base, arguments.head)
    except DcoError as error:
        print(f"DCO check failed: {error}", file=sys.stderr)
        return 2
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        print(
            "Add the matching sign-off with `git commit --signoff` and update the PR branch.",
            file=sys.stderr,
        )
        return 1
    print(f"DCO sign-offs: {len(commits)} introduced commit(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
