# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
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
    _append_benchmark_catalog_to_sdist(Path(sdist_directory) / filename)
    return filename


def _append_benchmark_catalog_to_sdist(sdist_path: Path) -> None:
    """Add the minimal canonical benchmark catalog to a source archive."""

    catalog_root = Path.cwd() / "tests" / "e2e" / "models"
    descriptors = sorted(catalog_root.glob("*/MODEL.toml"))
    manifests = sorted(catalog_root.glob("*/manifests/*.json"))
    assets = set(catalog_root.glob("*/data/Recording.wav"))
    if not descriptors or not manifests:
        raise RuntimeError(f"benchmark model catalog is empty or unavailable: {catalog_root}")
    for manifest in manifests:
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read benchmark manifest {manifest}: {exc}") from exc
        references = [("fp8_scales", raw.get("fp8_scales"))]
        for index, testcase in enumerate(raw.get("testcases", [])):
            if isinstance(testcase, dict) and "test_image" in testcase:
                references.append((f"testcases[{index}].test_image", testcase["test_image"]))
        for field, declared in references:
            if declared is None:
                continue
            if not isinstance(declared, str) or not declared.strip():
                raise RuntimeError(f"{field} in benchmark manifest {manifest} must be a path")
            family = manifest.parent.parent.resolve()
            declared_path = Path(declared)
            asset = (family / declared_path).resolve()
            source_prefix = Path("tests/e2e/models") / family.name
            if not asset.is_file() and declared_path.is_relative_to(source_prefix):
                asset = (family / declared_path.relative_to(source_prefix)).resolve()
            if not asset.is_relative_to(family) or not asset.is_file():
                raise RuntimeError(
                    f"{field} in benchmark manifest {manifest} is missing or outside {family}: "
                    f"{asset}"
                )
            assets.add(asset)
    assets = sorted(assets)

    with tempfile.NamedTemporaryFile(
        prefix=f".{sdist_path.name}.", suffix=".tmp", dir=sdist_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(sdist_path, "r:gz") as source:
            members = source.getmembers()
            if not members:
                raise RuntimeError(f"source archive is empty: {sdist_path}")
            archive_root = members[0].name.partition("/")[0]
            catalog_names = {
                f"{archive_root}/{path.relative_to(Path.cwd()).as_posix()}"
                for path in (*descriptors, *manifests, *assets)
            }
            existing = {member.name for member in members}
            pending = catalog_names - existing

            with tarfile.open(temporary_path, "w:gz", format=tarfile.PAX_FORMAT) as destination:
                for member in members:
                    stream = source.extractfile(member) if member.isfile() else None
                    destination.addfile(member, stream)
                for path in (*descriptors, *manifests, *assets):
                    archive_name = f"{archive_root}/{path.relative_to(Path.cwd()).as_posix()}"
                    if archive_name in pending:
                        destination.add(path, arcname=archive_name, recursive=False)
        temporary_path.replace(sdist_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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
