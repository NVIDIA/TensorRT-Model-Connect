# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


_CONAN_PY_BUILD_REQUIREMENT = "conan-py-build==0.4.3"
_PYTHON_SOURCE_ROOT = "python"
_FORBIDDEN_ADAPTER_ARCHIVE_PARTS = frozenset(
    {
        ".runtime-build",
        "artifacts",
        "build",
        "dependencies",
        "evidence",
        "qualification",
        "qualifications",
        "results",
        "tests",
    }
)
_GENERATED_ADAPTER_SUFFIXES = (
    ".dll",
    ".dylib",
    ".engine",
    ".onnx",
    ".plan",
    ".safetensors",
    ".so",
)


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    return [_CONAN_PY_BUILD_REQUIREMENT]


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return [_CONAN_PY_BUILD_REQUIREMENT]


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    if _py_only_enabled(config_settings):
        return []
    return [_CONAN_PY_BUILD_REQUIREMENT]


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    conan_build = _conan_build_backend()
    return conan_build.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings,
    )


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    conan_build = _conan_build_backend()
    return conan_build.build_wheel(
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    conan_build = _conan_build_backend()
    filename = conan_build.build_sdist(sdist_directory, config_settings)
    _validate_sdist_adapter_contents(Path(sdist_directory) / filename)
    return filename


def _validate_sdist_adapter_contents(archive_path: Path) -> None:
    """Fail closed if a model adapter source archive carries non-source data."""

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        adapter_prefixes: set[tuple[str, ...]] = set()
        required_runtime_cmake: dict[tuple[str, ...], tuple[str, ...]] = {}
        builder_marker = ("python", "tensorrt_model_connect", "families")
        runtime_marker = ("src", "runtime", "models")
        for member in members:
            parts = tuple(part for part in Path(member.name).parts if part not in {"", "."})
            for index in range(len(parts) - len(builder_marker) - 2):
                if parts[index : index + len(builder_marker)] != builder_marker:
                    continue
                tail = parts[index + len(builder_marker) :]
                if len(tail) == 3 and tail[-1] == "IMPLEMENTATION.toml":
                    builder_prefix = parts[:-1]
                    adapter_prefixes.add(builder_prefix)
                    root_prefix = parts[:index]
                    runtime_prefix = root_prefix + runtime_marker + (tail[0], tail[1])
                    adapter_prefixes.add(runtime_prefix)
                    required_runtime_cmake[builder_prefix] = runtime_prefix + ("CMakeLists.txt",)

        archive_files = {
            tuple(part for part in Path(member.name).parts if part not in {"", "."})
            for member in members
            if member.isfile()
        }
        for builder_prefix, runtime_cmake in sorted(required_runtime_cmake.items()):
            if runtime_cmake not in archive_files:
                builder_name = "/".join(builder_prefix)
                raise RuntimeError(
                    "Source distribution is missing model-adapter Runtime source for "
                    f"{builder_name}: expected {'/'.join(runtime_cmake)}"
                )

        for member in members:
            parts = tuple(part for part in Path(member.name).parts if part not in {"", "."})
            prefix = next(
                (
                    candidate
                    for candidate in adapter_prefixes
                    if len(parts) >= len(candidate) and parts[: len(candidate)] == candidate
                ),
                None,
            )
            if prefix is None:
                continue
            relative = parts[len(prefix) :]
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Source distribution contains a model-adapter link: {member.name}"
                )
            forbidden = sorted(_FORBIDDEN_ADAPTER_ARCHIVE_PARTS.intersection(relative))
            if forbidden:
                raise RuntimeError(
                    "Source distribution contains forbidden model-adapter data: "
                    f"{member.name} ({', '.join(forbidden)})"
                )
            name = member.name.lower()
            if member.isfile() and any(
                name.endswith(suffix) or f"{suffix}." in name
                for suffix in _GENERATED_ADAPTER_SUFFIXES
            ):
                raise RuntimeError(
                    f"Source distribution contains a generated model-adapter payload: {member.name}"
                )


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    if not _py_only_enabled(config_settings):
        return prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    return _write_dist_info(Path(metadata_directory)).name


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    if not _py_only_enabled(config_settings):
        conan_build = _conan_build_backend()
        return conan_build.build_editable(
            wheel_directory,
            config_settings,
            metadata_directory,
        )

    wheel_dir = Path(wheel_directory)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    project = _project_metadata()
    distribution = _wheel_distribution_name(project["name"])
    wheel_name = f"{distribution}-{project['version']}-py3-none-any.whl"
    wheel_path = wheel_dir / wheel_name
    expected_dist_info = f"{distribution}-{project['version']}.dist-info"

    if metadata_directory is None:
        with tempfile.TemporaryDirectory() as tmp:
            dist_info = _write_dist_info(Path(tmp))
            _write_editable_wheel(wheel_path, dist_info, expected_dist_info)
    else:
        dist_info = _find_dist_info(Path(metadata_directory))
        _write_editable_wheel(wheel_path, dist_info, expected_dist_info)

    return wheel_name


def _conan_build_backend() -> Any:
    from conan_py_build import build as conan_build

    return conan_build


def _py_only_enabled(config_settings: dict[str, Any] | None) -> bool:
    if not config_settings:
        return False
    for key in ("py-only", "--py-only", "python-only", "--python-only"):
        if key in config_settings and _truthy(config_settings[key]):
            return True
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_truthy(item) for item in value)
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"", "1", "true", "yes", "on"}


def _read_pyproject() -> dict[str, Any]:
    with open(Path.cwd() / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _project_metadata() -> dict[str, Any]:
    project = _read_pyproject()["project"]
    return {
        "name": project["name"],
        "version": project["version"],
        "description": project.get("description", ""),
        "requires-python": project.get("requires-python"),
        "dependencies": project.get("dependencies", []),
        "optional-dependencies": project.get("optional-dependencies", {}),
    }


def _write_dist_info(parent: Path) -> Path:
    project = _project_metadata()
    dist_info = (
        parent / f"{_wheel_distribution_name(project['name'])}-{project['version']}.dist-info"
    )
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(project), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    return dist_info


def _metadata_text(project: dict[str, Any]) -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
    ]
    if project["description"]:
        lines.append(f"Summary: {project['description']}")
    if project["requires-python"]:
        lines.append(f"Requires-Python: {project['requires-python']}")
    for dependency in project["dependencies"]:
        lines.append(f"Requires-Dist: {dependency}")
    for extra, dependencies in project["optional-dependencies"].items():
        normalized_extra = _extra_name(extra)
        lines.append(f"Provides-Extra: {normalized_extra}")
        for dependency in dependencies:
            lines.append(
                f"Requires-Dist: {_dependency_with_extra_marker(dependency, normalized_extra)}"
            )
    lines.append("")
    return "\n".join(lines)


def _dependency_with_extra_marker(dependency: str, extra: str) -> str:
    if ";" in dependency:
        requirement, marker = dependency.split(";", maxsplit=1)
        return f'{requirement.strip()}; ({marker.strip()}) and extra == "{extra}"'
    return f'{dependency}; extra == "{extra}"'


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: trtmc-pyproject-backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _write_editable_wheel(wheel_path: Path, dist_info: Path, dist_info_name: str) -> None:
    source_path = (Path.cwd() / _PYTHON_SOURCE_ROOT).resolve()
    pth_name = "__editable__.tensorrt_model_connect.pth"
    entries: list[tuple[str, bytes]] = [(pth_name, f"{source_path}\n".encode())]
    for path in sorted(dist_info.iterdir()):
        if path.is_file() and path.name != "RECORD":
            entries.append((f"{dist_info_name}/{path.name}", path.read_bytes()))

    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for archive_name, data in entries:
            wheel.writestr(archive_name, data)
            records.append((archive_name, f"sha256={_sha256(data)}", str(len(data))))
        record_name = f"{dist_info_name}/RECORD"
        records.append((record_name, "", ""))
        wheel.writestr(record_name, _record_text(records))


def _record_text(records: list[tuple[str, str, str]]) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerows(records)
    return out.getvalue().encode()


def _sha256(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _find_dist_info(metadata_directory: Path) -> Path:
    if metadata_directory.name.endswith(".dist-info"):
        return metadata_directory
    matches = sorted(metadata_directory.glob("*.dist-info"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one .dist-info directory under {metadata_directory}, found {len(matches)}"
        )
    return matches[0]


def _wheel_distribution_name(name: str) -> str:
    return re.sub(r"[^\w\d.]+", "_", name, flags=re.ASCII).strip("_")


def _extra_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
