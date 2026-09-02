# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep community-ci.txt consumers on the minimal requirements build context.

.dockerignore deliberately excludes ``requirements/`` from the repository-root
context, so an image that needs ``community-ci.txt`` must build from
``requirements/`` itself and copy the file by its bare name. No CI job builds
the development images, so this contract is only checked here.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Dockerfiles that install the pinned community CI profile.
COMMUNITY_PROFILE_DOCKERFILES = (
    "Dockerfile.community-cpu",
    "Dockerfile.dev.x86",
    "Dockerfile.dev.aarch64",
)

SOURCE_BUILD_DOC = REPO_ROOT / "website/docs/getting-started/source-build.md"


def _copy_sources(dockerfile: Path) -> list[str]:
    """Context-relative sources of every COPY that reads the build context."""
    sources: list[str] = []
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        parts = stripped.split()[1:]
        if any(part.startswith("--from=") for part in parts):
            continue  # copies from an earlier stage, not the context
        parts = [part for part in parts if not part.startswith("--")]
        sources.extend(parts[:-1])  # last argument is the destination
    return sources


def test_community_profile_images_copy_by_bare_name() -> None:
    """Intent: requirements/ is excluded from the root context by design.
    Preconditions: each image installs the pinned community CI profile.
    Postconditions: every such COPY names community-ci.txt without a directory.
    """
    offenders = []
    for name in COMMUNITY_PROFILE_DOCKERFILES:
        dockerfile = REPO_ROOT / name
        assert dockerfile.is_file(), f"{name} is declared here but does not exist"
        for source in _copy_sources(dockerfile):
            if Path(source).name == "community-ci.txt" and source != "community-ci.txt":
                offenders.append(f"{name}: COPY {source}")

    assert not offenders, (
        "these Dockerfiles copy community-ci.txt by a root-relative path, which "
        "cannot resolve because .dockerignore excludes requirements/ from the "
        "root context: " + "; ".join(offenders)
    )


def test_requirements_stays_out_of_the_root_context() -> None:
    """Intent: keep the repository-root daemon context minimal.
    Preconditions: .dockerignore excludes everything and re-includes an allowlist.
    Postconditions: requirements/ is never re-included at the root.
    """
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "!requirements/" not in dockerignore
    assert "!requirements/community-ci.txt" not in dockerignore


def test_source_build_doc_uses_the_minimal_requirements_context() -> None:
    """Intent: the documented build must actually succeed.
    Preconditions: source-build.md builds a development image.
    Postconditions: it passes requirements as the build context.
    """
    doc = SOURCE_BUILD_DOC.read_text(encoding="utf-8")
    command = re.search(r"docker build \\\n(?:\s+[^\n]*\\\n)*\s+[^\n]*\n", doc)

    assert command is not None, "no docker build invocation found in source-build.md"
    assert command.group(0).rstrip().endswith("requirements"), (
        "source-build.md must build the development image from the requirements "
        f"context, got: {command.group(0).strip()!r}"
    )
