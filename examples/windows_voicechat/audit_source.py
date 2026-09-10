# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject accidental generated artifacts in this example's Git distribution.

Run with Python's standard library; no dependency installation is needed.
The default audits tracked and non-ignored untracked working-tree files.
Use --staged before a commit to inspect the actual Git index, including files
force-added despite .gitignore. This is an artifact guard, not a license scan:
reviewers must still check the provenance of source changes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


SOURCE_SUFFIXES = {".ps1", ".py", ".h", ".cpp", ".md", ".json", ".js", ".cjs", ".css", ".html"}
SOURCE_NAMES = {".gitignore", "CMakeLists.txt"}
GENERATED_DIRECTORIES = {
    "dependencies", "node_modules", "vendor", "third_party", "third-party",
    "models", "runtime", "logs", "licenses", "nemotron voice lab", "build",
    "dist", "out", ".venv", "venv", "__pycache__", ".pytest_cache",
}
MAX_SOURCE_BYTES = 1024 * 1024


def check_source(name: str, data: bytes, mode: str = "100644", byte_size: int | None = None) -> list[str]:
    """Check one path relative to the example root and its distributed bytes."""
    path = PurePosixPath(name)
    problems = []
    if mode not in {"100644", "100755"}:
        problems.append(f"unsupported Git mode {mode}; only regular source files are allowed")
    if path.is_absolute() or ".." in path.parts:
        problems.append("path escapes the example directory")
    if any(part.lower() in GENERATED_DIRECTORIES or part.lower().startswith("build-")
           for part in path.parts[:-1]):
        problems.append("generated or third-party dependency directory")
    is_requirements = path.name.startswith("requirements") and path.suffix == ".txt"
    if path.suffix not in SOURCE_SUFFIXES and path.name not in SOURCE_NAMES and not is_requirements:
        problems.append("not an approved source file type")
    if max(len(data), byte_size or 0) > MAX_SOURCE_BYTES:
        problems.append(f"larger than the {MAX_SOURCE_BYTES}-byte source limit")
    try:
        source = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        problems.append("not UTF-8 source text")
    else:
        if "\x00" in source:
            problems.append("contains binary NUL bytes")
        if re.search(r"data:[^\s'\"<>]{0,120};base64,[A-Za-z0-9+/]{128}", source):
            problems.append("contains an embedded binary data URL")
    return problems


def git_output(git: str, root: Path, *args: str) -> bytes:
    return subprocess.run([git, "-C", str(root), *args], check=True, capture_output=True).stdout


def audit(example_root: Path, git: str, staged: bool) -> int:
    repository = Path(git_output(git, example_root, "rev-parse", "--show-toplevel").decode().strip())
    prefix = example_root.relative_to(repository).as_posix() + "/"
    errors = []
    counts: Counter[str] = Counter()
    total_bytes = 0
    if staged:
        entries = git_output(git, repository, "ls-files", "--stage", "-z", "--", prefix).split(b"\0")
    else:
        entries = sorted(set(git_output(git, repository, "ls-files", "--cached", "--others",
                                        "--exclude-standard", "-z", "--", prefix).split(b"\0")))
    for entry in entries:
        if not entry:
            continue
        mode = "100644"
        if staged:
            metadata, raw_name = entry.split(b"\t", 1)
            mode, blob, stage = metadata.decode("ascii").split()
            if stage != "0":
                errors.append(f"{raw_name.decode('utf-8')}: unresolved Git merge entry")
                continue
            byte_size = int(git_output(git, repository, "cat-file", "-s", blob))
            # A mistakenly staged checkpoint can be tens of GB. Reject it
            # without loading it into memory or sending it through stdout.
            data = (git_output(git, repository, "cat-file", "blob", blob)
                    if byte_size <= MAX_SOURCE_BYTES and mode in {"100644", "100755"} else b"")
        else:
            raw_name = entry
            local_path = repository / raw_name.decode("utf-8")
            if local_path.is_symlink():
                mode = "120000"
            elif not local_path.exists():
                continue  # A tracked deletion contributes no source bytes.
            if not local_path.resolve().is_relative_to(example_root):
                errors.append(f"{local_path}: resolves outside the example directory")
                continue
            byte_size = local_path.stat().st_size
            with local_path.open("rb") as handle:
                data = handle.read(MAX_SOURCE_BYTES + 1)
        name = raw_name.decode("utf-8")
        if not name.startswith(prefix):
            raise ValueError(f"Git returned a path outside the example: {name}")
        relative_name = name[len(prefix):]
        errors.extend(f"{relative_name}: {problem}"
                      for problem in check_source(relative_name, data, mode, byte_size))
        counts[PurePosixPath(relative_name).suffix or "(no extension)"] += 1
        total_bytes += byte_size
    if not counts:
        errors.append("no source files found; stage the example before using --staged")
    if errors:
        print("Source distribution audit failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    kind = "Git index" if staged else "prospective Git files"
    print(f"Source distribution audit passed: {sum(counts.values())} files, {total_bytes:,} bytes ({kind}).")
    print("Types: " + ", ".join(f"{suffix}={count}" for suffix, count in sorted(counts.items())))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git", default="git", help="Git executable path, if not on PATH")
    parser.add_argument("--staged", action="store_true", help="audit the actual staged/index bytes")
    args = parser.parse_args()
    try:
        return audit(Path(__file__).resolve().parent, args.git, args.staged)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Unable to audit source distribution: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
