# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from email.parser import Parser
from pathlib import Path
from typing import Any, Iterator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


_CONAN_PY_BUILD_REQUIREMENT = "conan-py-build==0.4.3"
_PYTHON_SOURCE_ROOT = "python"
_PACKAGE_TENSORRT_VERSION_ENV = "TRTMC_PACKAGE_TENSORRT_VERSION"
_PACKAGE_VERSION_ENV = "TRTMC_PACKAGE_VERSION"
_TENSORRT_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
_TENSORRT_REQUIREMENT = re.compile(
    r"tensorrt\s*==\s*"
    r"(?P<version>[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)"
    r"\s*;\s*platform_machine\s*==\s*[\"'](?P<arch>aarch64|x86_64)[\"']\s*",
    re.IGNORECASE,
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
    with _variant_conan_build_backend() as conan_build:
        return conan_build.prepare_metadata_for_build_wheel(
            metadata_directory,
            config_settings,
        )


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    with _variant_conan_build_backend(metadata_directory) as conan_build:
        return conan_build.build_wheel(
            wheel_directory,
            config_settings,
            metadata_directory,
        )


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    if os.environ.get(_PACKAGE_TENSORRT_VERSION_ENV, "").strip():
        raise RuntimeError("TensorRT package profiles apply to wheels, not source archives")
    with _variant_conan_build_backend() as conan_build:
        filename = conan_build.build_sdist(sdist_directory, config_settings)
    _append_benchmark_catalog_to_sdist(Path(sdist_directory) / filename)
    return filename


def _append_benchmark_catalog_to_sdist(sdist_path: Path) -> None:
    """Add the minimal canonical benchmark catalog to a source archive."""

    catalog_root = Path.cwd() / "python" / "tensorrt_model_connect" / "models"
    descriptors = sorted(catalog_root.glob("*/MODEL.toml"))
    manifests = sorted(catalog_root.glob("*/tests/manifests/*.json"))
    assets = set(catalog_root.glob("*/tests/data/Recording.wav"))
    if not descriptors or not manifests:
        raise RuntimeError(f"benchmark model catalog is empty or unavailable: {catalog_root}")
    for manifest in manifests:
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read benchmark manifest {manifest}: {exc}") from exc
        references = [("fp8_scales", raw.get("fp8_scales"))]
        for index, testcase in enumerate(raw.get("testcases", [])):
            if not isinstance(testcase, dict):
                continue
            for field in (
                "test_image",
                "prompt_file",
                "test_input_audio",
                "camera_intrinsics_file",
            ):
                if field in testcase:
                    references.append((f"testcases[{index}].{field}", testcase[field]))
        for field, declared in references:
            if declared is None:
                continue
            if not isinstance(declared, str) or not declared.strip():
                raise RuntimeError(f"{field} in benchmark manifest {manifest} must be a path")
            model_tests = manifest.parent.parent.resolve()
            declared_path = Path(declared)
            asset = (model_tests / declared_path).resolve()
            if not asset.is_relative_to(model_tests) or not asset.is_file():
                raise RuntimeError(
                    f"{field} in benchmark manifest {manifest} is missing or outside "
                    f"{model_tests}: "
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
        with _variant_conan_build_backend(metadata_directory) as conan_build:
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
        _validate_prepared_metadata(Path(metadata_directory), project)
        dist_info = _find_dist_info(Path(metadata_directory))
        _write_editable_wheel(wheel_path, dist_info, expected_dist_info)

    return wheel_name


def _conan_build_backend() -> Any:
    from conan_py_build import build as conan_build

    return conan_build


@contextmanager
def _variant_conan_build_backend(
    metadata_directory: str | None = None,
) -> Iterator[Any]:
    """Apply the selected wheel profile to the pinned native build backend."""

    conan_build = _conan_build_backend()
    metadata_reader = getattr(conan_build, "_get_project_metadata", None)
    if metadata_reader is None:
        raise RuntimeError(
            f"{_CONAN_PY_BUILD_REQUIREMENT} no longer exposes its pinned metadata reader"
        )

    resolved = _resolved_project_metadata(Path.cwd())
    if metadata_directory is not None:
        _validate_prepared_metadata(Path(metadata_directory), resolved)

    def variant_metadata(project_dir: Path) -> dict[str, Any]:
        return copy.deepcopy(_resolved_project_metadata(project_dir))

    missing = object()
    previous_package_version = os.environ.get(_PACKAGE_VERSION_ENV, missing)
    os.environ[_PACKAGE_VERSION_ENV] = resolved["version"]
    conan_build._get_project_metadata = variant_metadata
    try:
        yield conan_build
    finally:
        conan_build._get_project_metadata = metadata_reader
        if previous_package_version is missing:
            os.environ.pop(_PACKAGE_VERSION_ENV, None)
        else:
            os.environ[_PACKAGE_VERSION_ENV] = previous_package_version


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


def _project_metadata() -> dict[str, Any]:
    return _resolved_project_metadata(Path.cwd())


def _resolved_project_metadata(project_dir: Path) -> dict[str, Any]:
    """Resolve the dynamic package version and TensorRT dependency profile."""

    with (project_dir / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    project = copy.deepcopy(pyproject["project"])
    package = pyproject["tool"]["tensorrt-model-connect"]["package"]
    base_version = str(package["base-version"])
    target = os.environ.get(_PACKAGE_TENSORRT_VERSION_ENV, "").strip()
    if not target:
        target = str(package["default-tensorrt-version"])
        package_version = base_version
    else:
        if not _TENSORRT_VERSION.fullmatch(target):
            raise RuntimeError(
                f"{_PACKAGE_TENSORRT_VERSION_ENV} must be an exact four-part version"
            )
        major, minor, *_ = target.split(".")
        package_version = f"{base_version}+trt{major}{minor}"

    dynamic = set(project.get("dynamic", []))
    if not {"version", "dependencies"}.issubset(dynamic):
        raise RuntimeError("project version and dependencies must remain backend-resolved")
    project["dynamic"] = sorted(dynamic - {"version", "dependencies"})
    if not project["dynamic"]:
        project.pop("dynamic")
    project["version"] = package_version
    project["dependencies"] = [
        *package.get("dependencies", []),
        f'tensorrt=={target}; platform_machine == "aarch64"',
        f'tensorrt=={target}; platform_machine == "x86_64"',
    ]
    return project


def _validate_prepared_metadata(
    metadata_directory: Path,
    expected: dict[str, Any],
) -> None:
    """Reject metadata prepared for a different wheel profile."""

    dist_info = _find_dist_info(metadata_directory)
    expected_dist_info = (
        f"{_wheel_distribution_name(expected['name'])}-{expected['version']}.dist-info"
    )
    if dist_info.name != expected_dist_info:
        raise RuntimeError(
            f"prepared wheel metadata directory is {dist_info.name}; expected {expected_dist_info}"
        )
    metadata_path = dist_info / "METADATA"
    if not metadata_path.is_file():
        raise RuntimeError(f"prepared wheel metadata is missing: {metadata_path}")
    metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("Name") != expected["name"] or metadata.get("Version") != expected["version"]:
        raise RuntimeError("prepared wheel metadata does not match the selected package variant")
    expected_tensorrt = _tensorrt_dependency_profile(expected["dependencies"])
    observed_tensorrt = _tensorrt_dependency_profile(
        metadata.get_all("Requires-Dist", [])
    )
    if observed_tensorrt != expected_tensorrt:
        raise RuntimeError("prepared wheel metadata has the wrong TensorRT dependency")


def _tensorrt_dependency_profile(dependencies: list[str]) -> dict[str, str]:
    candidates = [
        dependency
        for dependency in dependencies
        if re.match(r"tensorrt(?:\s|[<>=!~@;\[])", dependency, re.IGNORECASE)
    ]
    profile: dict[str, str] = {}
    for dependency in candidates:
        match = _TENSORRT_REQUIREMENT.fullmatch(dependency)
        if match is None or match.group("arch") in profile:
            raise RuntimeError("TensorRT dependencies must be exact per-architecture pins")
        profile[match.group("arch").lower()] = match.group("version")
    if set(profile) != {"aarch64", "x86_64"} or len(set(profile.values())) != 1:
        raise RuntimeError("TensorRT dependencies must pin one version for both architectures")
    return profile


def _write_dist_info(parent: Path) -> Path:
    project = _project_metadata()
    dist_info = (
        parent / f"{_wheel_distribution_name(project['name'])}-{project['version']}.dist-info"
    )
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(project), encoding="utf-8")
    (dist_info / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    _copy_license_files(dist_info, project)
    return dist_info


def _metadata_text(project: dict[str, Any]) -> str:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
    ]
    if project["description"]:
        lines.append(f"Summary: {project['description']}")
    if project.get("license"):
        lines.append(f"License-Expression: {project['license']}")
    for license_file in project.get("license-files", []):
        lines.append(f"License-File: {license_file}")
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


def _copy_license_files(dist_info: Path, project: dict[str, Any]) -> None:
    repository = Path.cwd().resolve()
    destination = dist_info / "licenses"
    for relative in project.get("license-files", []):
        source = (repository / relative).resolve()
        if not source.is_relative_to(repository) or not source.is_file():
            raise FileNotFoundError(f"license file is missing or outside the project: {relative}")
        target = destination / source.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


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
    for path in sorted(dist_info.rglob("*")):
        if path.is_file() and path.name != "RECORD":
            relative = path.relative_to(dist_info).as_posix()
            entries.append((f"{dist_info_name}/{relative}", path.read_bytes()))

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
