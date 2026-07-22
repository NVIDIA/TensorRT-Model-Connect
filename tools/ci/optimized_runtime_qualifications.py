# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select model-owned optimized-runtime qualifications for a source diff.

Boundary: discover and validate generic producer descriptors and emit one
GitHub matrix; model/runtime-specific execution remains in each descriptor's
entrypoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


DESCRIPTOR_GLOB = "tests/e2e/models/*/*/QUALIFICATION.*.toml"
_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_IMAGE = re.compile(r"\S+@sha256:[0-9a-f]{64}")
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.toml")
_MAX_MATRIX_ENTRIES = 256


class QualificationError(ValueError):
    """A qualification descriptor or selection input is invalid."""


@dataclass(frozen=True)
class ProducerQualification:
    path: str
    id: str
    runtime_id: str
    entrypoint: str
    container_image: str
    runner_labels: tuple[str, ...]
    representative: bool
    profile_glob: str
    trigger_globs: tuple[str, ...]
    representative_trigger_globs: tuple[str, ...]
    profile_target: dict[str, object]


def _relative_path(value: object, field: str, *, allow_glob: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise QualificationError(f"{field} must be a non-empty POSIX path")
    if any(character in value for character in "\r\n\t"):
        raise QualificationError(f"{field} must be a single-line POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise QualificationError(f"{field} must be a canonical repository-relative path")
    if not allow_glob and any(character in value for character in "*?["):
        raise QualificationError(f"{field} must not contain glob characters")
    if allow_glob and "[" in value:
        raise QualificationError(f"{field} supports only * and ? glob characters")
    return value


def _matches(path: str, pattern: str) -> bool:
    expression = re.escape(pattern)
    expression = expression.replace(r"\*\*/", "(?:.*/)?")
    expression = expression.replace(r"\*\*", ".*")
    expression = expression.replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.fullmatch(expression, path) is not None


def _is_concrete_model_owned_pattern(pattern: str) -> bool:
    """Return whether a shared trigger reaches below a model-family root."""

    parts = PurePosixPath(pattern).parts
    model_roots = (
        ("python", "tensorrt_model_connect", "families"),
        ("src", "runtime", "models"),
        ("tests", "e2e", "models"),
    )
    for root in model_roots:
        if parts[: len(root)] != root or len(parts) == len(root):
            continue
        descendants = parts[len(root) :]
        if any(character in descendants[0] for character in "*?") or len(descendants) >= 2:
            return True
    return False


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise QualificationError(f"{field} must be a non-empty list")
    if any(
        not isinstance(item, str) or not item or any(character in item for character in "\r\n\t")
        for item in value
    ):
        raise QualificationError(f"{field} entries must be non-empty single-line strings")
    if len(set(value)) != len(value):
        raise QualificationError(f"{field} entries must be unique")
    return tuple(value)


def _identifier(value: object, relative: str, field: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _ID.fullmatch(value) is None:
        raise QualificationError(f"{relative}: {field} is invalid")
    return value


def _target_table(value: object, relative: str, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise QualificationError(f"{relative}: {field} must be a non-empty table")
    if any(
        not isinstance(key, str)
        or not key
        or type(item) not in (str, int, float, bool)
        or (type(item) is float and not math.isfinite(item))
        for key, item in value.items()
    ):
        raise QualificationError(f"{relative}: {field} must contain JSON scalar fields")
    return value


def _common_fields(repository: Path, relative: str, data: dict[str, object]) -> dict[str, object]:
    identifier = _identifier(data["id"], relative, "id")
    runtime_id = _identifier(data["runtime_id"], relative, "runtime_id")
    entrypoint = _relative_path(data["entrypoint"], f"{relative}: entrypoint")
    owned_root = PurePosixPath(relative).parent
    if not PurePosixPath(entrypoint).is_relative_to(owned_root):
        raise QualificationError(f"{relative}: entrypoint must be owned beside the descriptor")
    entrypoint_path = repository / entrypoint
    if not entrypoint_path.is_file() or entrypoint_path.is_symlink():
        raise QualificationError(f"{relative}: entrypoint does not exist: {entrypoint}")
    if not entrypoint_path.stat().st_mode & 0o111:
        raise QualificationError(f"{relative}: entrypoint must be executable")
    image = data["container_image"]
    if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
        raise QualificationError(f"{relative}: container_image must be pinned by sha256 digest")
    return {
        "path": relative,
        "id": identifier,
        "runtime_id": runtime_id,
        "entrypoint": entrypoint,
        "container_image": image,
        "runner_labels": _string_list(data["runner_labels"], f"{relative}: runner_labels"),
    }


def _load_descriptor(repository: Path, path: Path) -> ProducerQualification:
    relative = path.relative_to(repository).as_posix()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise QualificationError(f"cannot read {relative}: {error}") from error
    producer_fields = {
        "schema_version",
        "kind",
        "id",
        "runtime_id",
        "representative",
        "entrypoint",
        "container_image",
        "runner_labels",
        "profile_glob",
        "trigger_globs",
        "representative_trigger_globs",
        "profile_target",
    }
    kind = data.get("kind")
    if data.get("schema_version") != 2 or kind != "producer":
        raise QualificationError(
            f"{relative} must use producer qualification schema version 2"
        )
    if set(data) != producer_fields:
        raise QualificationError(f"{relative}: fields do not match {kind} schema")
    common = _common_fields(repository, relative, data)
    representative = data["representative"]
    if not isinstance(representative, bool):
        raise QualificationError(f"{relative}: representative must be Boolean")
    profile_glob = _relative_path(
        data["profile_glob"], f"{relative}: profile_glob", allow_glob=True
    )
    descriptor_parts = PurePosixPath(relative).parts
    family = descriptor_parts[3]
    adapter = descriptor_parts[4]
    profile_root = PurePosixPath(
        f"python/tensorrt_model_connect/families/{family}/{adapter}/profiles"
    )
    if not PurePosixPath(profile_glob).is_relative_to(profile_root):
        raise QualificationError(f"{relative}: profile_glob must stay in {profile_root}")
    raw_triggers = _string_list(data["trigger_globs"], f"{relative}: trigger_globs")
    trigger_globs = tuple(
        _relative_path(item, f"{relative}: trigger_globs", allow_glob=True) for item in raw_triggers
    )
    trigger_roots = (
        PurePosixPath(f"python/tensorrt_model_connect/families/{family}/{adapter}"),
        PurePosixPath(f"src/runtime/models/{family}/{adapter}"),
        PurePosixPath(f"tests/e2e/models/{family}/{adapter}"),
    )
    if any(
        not any(PurePosixPath(pattern).is_relative_to(root) for root in trigger_roots)
        for pattern in trigger_globs
    ):
        raise QualificationError(f"{relative}: trigger_globs must stay in model-owned roots")
    raw_shared_triggers = data["representative_trigger_globs"]
    if not isinstance(raw_shared_triggers, list) or any(
        not isinstance(item, str) for item in raw_shared_triggers
    ):
        raise QualificationError(f"{relative}: representative_trigger_globs must be a list")
    shared_triggers = tuple(
        _relative_path(item, f"{relative}: representative_trigger_globs", allow_glob=True)
        for item in raw_shared_triggers
    )
    if any(_is_concrete_model_owned_pattern(pattern) for pattern in shared_triggers):
        raise QualificationError(
            f"{relative}: representative_trigger_globs must not claim model-owned roots"
        )
    if representative and not shared_triggers:
        raise QualificationError(
            f"{relative}: representative must declare representative_trigger_globs"
        )
    if not representative and shared_triggers:
        raise QualificationError(
            f"{relative}: only a representative may declare representative_trigger_globs"
        )
    return ProducerQualification(
        **common,
        representative=representative,
        profile_glob=profile_glob,
        trigger_globs=trigger_globs,
        representative_trigger_globs=shared_triggers,
        profile_target=_target_table(data["profile_target"], relative, "profile_target"),
    )


def _profile_data(path: Path, relative: str) -> tuple[bool, dict[str, object]]:
    if path.is_symlink():
        raise QualificationError(f"profile must not be a symlink: {relative}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise QualificationError(f"cannot read profile {relative}: {error}") from error
    target = data.get("target")
    if not isinstance(target, dict):
        raise QualificationError(f"{relative}: target must be a table")
    return data.get("qualification_state") == "qualified", target


def _target_matches(target: dict[str, object], expected: dict[str, object]) -> bool:
    return all(
        type(target.get(key)) is type(value) and target.get(key) == value
        for key, value in expected.items()
    )


def _validate_qualified_profile_ownership(
    repository: Path, producers: Sequence[ProducerQualification]
) -> None:
    """Require every qualified family-adapter profile to have one CI producer."""

    producers_by_adapter: dict[tuple[str, str], list[ProducerQualification]] = {}
    for descriptor in producers:
        parts = PurePosixPath(descriptor.path).parts
        producers_by_adapter.setdefault((parts[3], parts[4]), []).append(descriptor)

    profiles_root = repository / "python/tensorrt_model_connect/families"
    for path in sorted(profiles_root.glob("*/*/profiles/**/*.toml")):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(repository).as_posix()
        qualified, target = _profile_data(path, relative)
        if not qualified:
            continue
        parts = PurePosixPath(relative).parts
        matching = [
            descriptor
            for descriptor in producers_by_adapter.get((parts[3], parts[4]), [])
            if _matches(relative, descriptor.profile_glob)
            and _target_matches(target, descriptor.profile_target)
        ]
        if len(matching) != 1:
            owners = [descriptor.id for descriptor in matching]
            raise QualificationError(
                f"{relative}: qualified profile must be owned by exactly one producer; "
                f"matched {owners}"
            )


def discover_descriptors(repository: Path) -> list[ProducerQualification]:
    repository = repository.resolve()
    descriptors = [
        _load_descriptor(repository, path)
        for path in sorted(repository.glob(DESCRIPTOR_GLOB))
        if path.is_file()
    ]
    identifiers = [descriptor.id for descriptor in descriptors]
    if len(set(identifiers)) != len(identifiers):
        raise QualificationError("qualification descriptor ids must be repository-unique")
    producers = descriptors
    for runtime_id in sorted({descriptor.runtime_id for descriptor in producers}):
        representatives = [
            descriptor
            for descriptor in producers
            if descriptor.runtime_id == runtime_id and descriptor.representative
        ]
        if len(representatives) != 1:
            raise QualificationError(
                f"runtime_id {runtime_id!r} must have exactly one representative descriptor"
            )
    _validate_qualified_profile_ownership(repository, producers)
    return producers


def _profiles(repository: Path, descriptor: ProducerQualification) -> dict[str, bool]:
    profiles: dict[str, bool] = {}
    paths_by_name: dict[str, list[str]] = {}
    for path in sorted(repository.glob(descriptor.profile_glob)):
        if not path.is_file():
            continue
        relative = path.relative_to(repository).as_posix()
        name = path.name
        if _PROFILE_NAME.fullmatch(name) is None:
            raise QualificationError(f"{descriptor.path}: invalid profile basename: {name}")
        paths_by_name.setdefault(name, []).append(relative)
        qualified, target = _profile_data(path, relative)
        profiles[relative] = qualified and _target_matches(target, descriptor.profile_target)
    duplicates = {name: paths for name, paths in paths_by_name.items() if len(paths) > 1}
    if duplicates:
        raise QualificationError(
            f"{descriptor.path}: profile basenames must be unique: {sorted(duplicates)}"
        )
    if not any(profiles.values()):
        raise QualificationError(
            f"{descriptor.path}: profile_glob has no profile for profile_target"
        )
    return profiles


def select_qualifications(
    repository: Path, changed_paths: Sequence[str], *, select_all: bool = False
) -> dict[str, object]:
    repository = repository.resolve()
    changed = sorted(
        {
            _relative_path(path, "changed path")
            for path in changed_paths
            if isinstance(path, str) and path
        }
    )
    descriptors = discover_descriptors(repository)
    producers: list[dict[str, object]] = []
    for descriptor in descriptors:
        profiles = _profiles(repository, descriptor)
        profile_changes = [path for path in changed if _matches(path, descriptor.profile_glob)]
        family_change = any(
            not _matches(path, descriptor.profile_glob)
            and any(_matches(path, pattern) for pattern in descriptor.trigger_globs)
            for path in changed
        )
        representative_change = descriptor.representative and any(
            not _is_concrete_model_owned_pattern(path)
            and any(_matches(path, pattern) for pattern in descriptor.representative_trigger_globs)
            for path in changed
        )
        broad_change = select_all or family_change or representative_change
        selected_names: list[str] = []
        if broad_change or any(path not in profiles for path in profile_changes):
            profile_files = ""
        else:
            selected_names = sorted(
                Path(path).name for path in profile_changes if profiles.get(path, False)
            )
            profile_files = ",".join(selected_names)
        if (
            not broad_change
            and not selected_names
            and not any(path not in profiles for path in profile_changes)
        ):
            continue
        producers.append(
            {
                "id": descriptor.id,
                "runtime_id": descriptor.runtime_id,
                "descriptor": descriptor.path,
                "entrypoint": descriptor.entrypoint,
                "container_image": descriptor.container_image,
                "runner_labels": list(descriptor.runner_labels),
                "profile_files": profile_files,
            }
        )
    selected_count = len(producers)
    if selected_count > _MAX_MATRIX_ENTRIES:
        raise QualificationError(
            f"selected {selected_count} qualifications; GitHub matrix limit is "
            f"{_MAX_MATRIX_ENTRIES}"
        )
    return {"producers": {"include": producers}}


def git_changed_paths(repository: Path, base: str, head: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            base,
            head,
            "--",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--files", nargs="+")
    source.add_argument("--base")
    source.add_argument("--all", action="store_true")
    parser.add_argument("--head", default="HEAD")
    arguments = parser.parse_args(argv)
    try:
        paths = arguments.files
        if paths is None and not arguments.all:
            paths = git_changed_paths(arguments.repository, arguments.base, arguments.head)
        print(
            json.dumps(
                select_qualifications(arguments.repository, paths or (), select_all=arguments.all),
                separators=(",", ":"),
            )
        )
    except (QualificationError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
