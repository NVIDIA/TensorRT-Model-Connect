# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import subprocess
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
_WAN22_BUILDER_COMPANION_GLOB = "libtrtmc_model_wan2_2_ti2v_plugins*.so"
_WAN22_BUILDER_COMPANION_RE = re.compile(
    r"^libtrtmc_model_wan2_2_ti2v_plugins_trt(?P<major>[0-9]+)_"
    r"(?P<minor>[0-9]+)\.so$"
)
_TENSORRT_EXACT_DEPENDENCY_RE = re.compile(
    r"^\s*tensorrt\s*==\s*(?P<major>[0-9]+)\.(?P<minor>[0-9]+)"
    r"(?:\.[A-Za-z0-9][A-Za-z0-9._+-]*)+\s*(?:;.*)?$",
    re.IGNORECASE,
)
_WAN22_RPATH_POLICY_MARKER = "_trtmc_wan22_rpath_policy"


def _is_native_runtime_payload(path: Path) -> bool:
    """Return whether a staged file is part of the relocatable native runtime."""

    name = path.name
    return (
        name == "trtmc"
        or name.startswith("libtrtmc_core.so")
        or name.startswith("libtrtmc_backend_")
        or name.startswith("libtrtmc_model_")
    )


def _run_patchelf(command: list[str], *, action: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("patchelf is required to normalize native wheel RPATHs") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f"failed to {action}: {detail}") from error


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
    return conan_build.build_sdist(sdist_directory, config_settings)


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

    _install_wan22_wheel_rpath_policy(conan_build)
    return conan_build


def _install_wan22_wheel_rpath_policy(conan_build: Any) -> None:
    """Make the flattened native wheel relocatable and seal the Wan companion.

    ``conan-py-build`` treats every file ending in the generic Python
    extension suffix ``.so`` as an extension module and appends ``$ORIGIN`` to
    its existing build-tree RUNPATH.  Model Connect flattens its executable,
    core, backends, and model DSOs into one wheel directory, so replace those
    inherited paths with exactly ``$ORIGIN``.  The Wan companion is an
    executable bundle payload rather than a runtime DSO, so remove its search
    path entirely before distlib writes the wheel and its RECORD.
    """

    original_patch_rpath = getattr(conan_build, "patch_rpath", None)
    if original_patch_rpath is None:
        raise RuntimeError("conan-py-build does not expose its staging RPATH hook")
    if getattr(original_patch_rpath, _WAN22_RPATH_POLICY_MARKER, False):
        return

    def patch_rpath(staging_dir: Path) -> None:
        original_patch_rpath(staging_dir)
        candidates = sorted(Path(staging_dir).rglob(_WAN22_BUILDER_COMPANION_GLOB))
        malformed = [
            path
            for path in candidates
            if not _WAN22_BUILDER_COMPANION_RE.fullmatch(path.name)
            or not path.is_file()
            or path.is_symlink()
        ]
        if malformed:
            raise RuntimeError(
                "wheel staging contains malformed Wan2.2 builder companion paths: "
                f"{[str(path) for path in malformed]}"
            )
        if len(candidates) != 1:
            raise RuntimeError(
                "wheel staging must contain exactly one ABI-tagged Wan2.2 builder "
                f"companion; found {[str(path) for path in candidates]}"
            )

        companion = candidates[0]
        filename_match = _WAN22_BUILDER_COMPANION_RE.fullmatch(companion.name)
        if filename_match is None:  # Covered by the malformed-path gate above.
            raise RuntimeError(f"invalid Wan2.2 builder companion name: {companion.name}")
        companion_abi = (
            int(filename_match.group("major")),
            int(filename_match.group("minor")),
        )
        required_abi = _project_tensorrt_abi()
        if companion_abi != required_abi:
            raise RuntimeError(
                "Wan2.2 wheel companion TensorRT ABI does not match the wheel dependency: "
                f"companion={companion_abi[0]}.{companion_abi[1]}, "
                f"Requires-Dist tensorrt={required_abi[0]}.{required_abi[1]}"
            )

        runtime_payloads = sorted(
            path
            for path in Path(staging_dir).rglob("*")
            if (path.is_file() or path.is_symlink())
            and _is_native_runtime_payload(path)
            and path != companion
        )
        malformed_payloads = [
            path for path in runtime_payloads if not path.is_file() or path.is_symlink()
        ]
        if malformed_payloads:
            raise RuntimeError(
                "wheel staging contains malformed native runtime payloads: "
                f"{[str(path) for path in malformed_payloads]}"
            )
        for payload in runtime_payloads:
            _run_patchelf(
                ["patchelf", "--set-rpath", "$ORIGIN", str(payload)],
                action=f"set the native wheel RUNPATH on {payload}",
            )
        _run_patchelf(
            ["patchelf", "--remove-rpath", str(companion)],
            action="remove the Wan2.2 wheel companion RPATH",
        )

    setattr(patch_rpath, _WAN22_RPATH_POLICY_MARKER, True)
    conan_build.patch_rpath = patch_rpath


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


def _project_tensorrt_abi() -> tuple[int, int]:
    dependencies = _read_pyproject().get("project", {}).get("dependencies", [])
    requirements = [
        str(dependency)
        for dependency in dependencies
        if re.match(r"^\s*tensorrt(?:\s|=|<|>|!|~|;|$)", str(dependency), re.IGNORECASE)
    ]
    if len(requirements) != 1:
        raise RuntimeError(
            "native wheel metadata must contain exactly one TensorRT dependency; "
            f"found {requirements}"
        )
    match = _TENSORRT_EXACT_DEPENDENCY_RE.fullmatch(requirements[0])
    if match is None:
        raise RuntimeError(
            "native wheel TensorRT dependency must be an exact major.minor version pin; "
            f"found {requirements[0]!r}"
        )
    return int(match.group("major")), int(match.group("minor"))


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
