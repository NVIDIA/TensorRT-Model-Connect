# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative Python profile registry and cached materialization helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

PROFILE_PHASES = ("build", "runtime", "reference")
DEFAULT_PROFILE = "base"
PROFILE_ROOT_ENV = "TRTMC_PYTHON_PROFILE_ROOT"
LEGACY_PROFILE_ROOT_ENV = "TRTMC_E2E_PROFILE_ROOT"
PREBUILT_ONLY_ENV = "TRTMC_PYTHON_PROFILE_PREBUILT_ONLY"
DEFAULT_PROFILE_ROOT = "/tmp/trtmc-python-profiles"
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROFILE_LAYOUT_VERSION = "overlay-v4-hermetic-targeted-cuda"
_DEFAULT_PROFILE_BUILD_JOBS = "4"
_PROFILE_INSTALL_TIMEOUT_SECONDS = 7200
_EXACT_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9,._-]+\])?==([^\s;]+)$"
)
_EXACT_VERSION_RE = re.compile(
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:(?:a|b|rc)[0-9]+)?"
    r"(?:\.post[0-9]+)?"
    r"(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?",
    re.IGNORECASE,
)
_PROFILE_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_BUILD_ENVIRONMENT_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
_FORBIDDEN_BUILD_ENVIRONMENT_NAMES = {
    "HOME",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PATH",
    "PYTHONHOME",
    "PYTHONPATH",
}
_FORBIDDEN_BUILD_ENVIRONMENT_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GIT_",
    "GOOGLE_",
    "NVIDIA_",
    "PIP_",
    "SSH_",
    "TRTMC_",
)
_REGISTRY_KEYS = {
    "version",
    "profiles",
    "reference_backend_defaults",
    "runtime_strategy_defaults",
}
_PASSTHROUGH_PROFILE_KEYS = {"kind"}
_VENV_PROFILE_KEYS = {
    "kind",
    "prebuild",
    "build_environment",
    "requirements",
    "system_site_packages",
    "verification_script",
    "verification_script_file",
}


def _absolute_python(path: str) -> str:
    if not path:
        return ""

    candidate = Path(path).absolute()
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", candidate.name):
        environment_root = candidate.parent.parent
        canonical = candidate.parent / "python"
        if (environment_root / "pyvenv.cfg").is_file() and canonical.is_file():
            return str(canonical)
    return str(candidate)


def _normalize_profile_name(profile_name: str | None) -> str:
    return str(profile_name or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE


def _profile_token(profile_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", profile_name.strip()).strip("_").upper()
    if not token:
        raise ValueError(f"Invalid execution profile name: {profile_name!r}")
    return token


def profile_env_var(profile_name: str) -> str:
    """Preferred override env var for a symbolic Python profile."""
    return f"TRTMC_PYTHON_PROFILE_{_profile_token(profile_name)}_PYTHON"


def profile_env_var_candidates(profile_name: str) -> tuple[str, ...]:
    """All supported override env vars, newest first."""
    token = _profile_token(profile_name)
    return (
        f"TRTMC_PYTHON_PROFILE_{token}_PYTHON",
        f"TRTMC_E2E_PROFILE_{token}_PYTHON",
    )


def profile_root() -> Path:
    configured = (
        os.environ.get(PROFILE_ROOT_ENV, "").strip()
        or os.environ.get(LEGACY_PROFILE_ROOT_ENV, "").strip()
        or DEFAULT_PROFILE_ROOT
    )
    return Path(configured)


@lru_cache(maxsize=1)
def load_python_profile_registry() -> dict[str, Any]:
    """Load generic and family-owned profiles into one validated registry."""
    registry_file = _PACKAGE_DIR / "python_profiles.toml"
    with registry_file.open("rb") as f:
        registry = tomllib.load(f)
    if not isinstance(registry, dict):
        raise ValueError("python_profiles.toml must decode to an object")
    registry = dict(registry)
    raw_profiles = registry.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError("python_profiles.toml is missing a [profiles] table")
    profiles = dict(raw_profiles)

    for name, spec in family_python_profile_specs().items():
        if name in profiles:
            raise ValueError(
                f"Execution profile {name!r} is declared in both "
                "python_profiles.toml and family metadata"
            )
        profiles[name] = spec
    registry["profiles"] = profiles
    _validate_python_profile_registry(registry)
    return registry


def _profile_metadata_bool(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid {field_name} bool {value!r}; expected true or false")


def family_python_profile_specs() -> dict[str, dict[str, object]]:
    """Read family-owned profile declarations without importing family code."""
    profiles: dict[str, dict[str, object]] = {}
    for manifest in sorted((_PACKAGE_DIR / "families").glob("*/MODEL.toml")):
        with manifest.open("rb") as stream:
            raw = tomllib.load(stream)
        family_id = raw.get("id") or raw.get("plugin") or manifest.parent.name
        family_profile_names: set[str] = set()
        raw_specs = raw.get("python_profile_specs", [])
        if not isinstance(raw_specs, list):
            raise ValueError(f"python_profile_specs for family {family_id} must be a list")
        for spec in raw_specs:
            if not isinstance(spec, str):
                raise ValueError(
                    f"python_profile_specs for family {family_id} must contain strings"
                )
            parts = [part.strip() for part in spec.split("|")]
            if len(parts) not in {3, 4, 5} or any(not part for part in parts[:3]):
                raise ValueError(
                    f"Invalid python_profile_specs entry {spec!r} for family "
                    f"{family_id}; expected 'name|requirements|verification_script|"
                    "system_site_packages|prebuild'"
                )
            name, requirements, verification_script_file = parts[:3]
            system_site_packages = (
                _profile_metadata_bool(parts[3], "python_profile_specs")
                if len(parts) >= 4
                else True
            )
            prebuild = (
                _profile_metadata_bool(parts[4], "python_profile_specs")
                if len(parts) >= 5
                else True
            )
            if name in profiles:
                raise ValueError(f"Python profile {name!r} is declared by multiple families")
            profile_spec: dict[str, object] = {
                "kind": "venv",
                "requirements": requirements,
                "verification_script_file": verification_script_file,
                "system_site_packages": system_site_packages,
                "prebuild": prebuild,
            }
            profiles[name] = profile_spec
            family_profile_names.add(name)
        raw_build_environment = raw.get("python_profile_build_environment", [])
        if not isinstance(raw_build_environment, list):
            raise ValueError(
                f"python_profile_build_environment for family {family_id} must be a list"
            )
        for entry in raw_build_environment:
            if not isinstance(entry, str):
                raise ValueError(
                    f"python_profile_build_environment for family {family_id} "
                    "must contain strings"
                )
            parts = [part.strip() for part in entry.split("|", 2)]
            if len(parts) != 3 or any(not part for part in parts):
                raise ValueError(
                    f"Invalid python_profile_build_environment entry {entry!r} "
                    f"for family {family_id}; expected 'profile|NAME|value'"
                )
            profile, name, value = parts
            if profile not in family_profile_names:
                raise ValueError(
                    f"python_profile_build_environment selects undeclared profile "
                    f"{profile!r} for family {family_id}"
                )
            if (
                _BUILD_ENVIRONMENT_NAME_RE.fullmatch(name) is None
                or name in _FORBIDDEN_BUILD_ENVIRONMENT_NAMES
                or name.startswith(_FORBIDDEN_BUILD_ENVIRONMENT_PREFIXES)
            ):
                raise ValueError(
                    f"Python profile {profile!r} has unsafe build environment name {name!r}"
                )
            if len(value) > 1024 or "\x00" in value or "\n" in value or "\r" in value:
                raise ValueError(
                    f"Python profile {profile!r} has an unsafe build environment value"
                )
            build_environment = dict(profiles[profile].get("build_environment", {}))
            if name in build_environment:
                raise ValueError(
                    f"Python profile {profile!r} declares build environment {name!r} twice"
                )
            build_environment[name] = value
            profiles[profile]["build_environment"] = build_environment
    return profiles


def _profile_asset_path(path_spec: str, *, field: str, profile_name: str) -> Path:
    path = Path(path_spec)
    if not path_spec or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Execution profile {profile_name!r} has an unsafe {field} path")
    resolved = (_PACKAGE_DIR / path).resolve()
    try:
        resolved.relative_to(_PACKAGE_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            f"Execution profile {profile_name!r} has an unsafe {field} path"
        ) from error
    if not resolved.is_file():
        raise ValueError(
            f"Execution profile {profile_name!r} references missing {field} asset {path_spec!r}"
        )
    return resolved


def _validate_python_profile_registry(registry: Mapping[str, Any]) -> None:
    """Reject ambiguous or non-hermetic profile declarations before building."""
    unknown_registry_keys = set(registry) - _REGISTRY_KEYS
    if unknown_registry_keys:
        raise ValueError(
            "Python profile registry has unknown top-level keys: "
            + ", ".join(sorted(unknown_registry_keys))
        )
    if type(registry.get("version")) is not int or registry["version"] != 1:
        raise ValueError("python_profiles.toml version must be integer 1")

    profiles = registry.get("profiles")
    if not isinstance(profiles, Mapping) or DEFAULT_PROFILE not in profiles:
        raise ValueError("python_profiles.toml must declare profiles.base")
    for name, raw_spec in profiles.items():
        if type(name) is not str or _PROFILE_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Invalid execution profile name: {name!r}")
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Execution profile {name!r} must be a table")
        kind = raw_spec.get("kind")
        allowed_keys = (
            _PASSTHROUGH_PROFILE_KEYS
            if kind == "passthrough"
            else _VENV_PROFILE_KEYS
            if kind == "venv"
            else set()
        )
        if not allowed_keys:
            raise ValueError(f"Execution profile {name!r} has unsupported kind {kind!r}")
        unknown_keys = set(raw_spec) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"Execution profile {name!r} has unknown keys: " + ", ".join(sorted(unknown_keys))
            )
        if kind == "passthrough":
            continue
        for field in ("system_site_packages", "prebuild"):
            if field in raw_spec and type(raw_spec[field]) is not bool:
                raise ValueError(
                    f"Execution profile {name!r} field {field} must be a bool"
                )
        build_environment = raw_spec.get("build_environment", {})
        if not isinstance(build_environment, Mapping) or any(
            not isinstance(name, str)
            or _BUILD_ENVIRONMENT_NAME_RE.fullmatch(name) is None
            or name in _FORBIDDEN_BUILD_ENVIRONMENT_NAMES
            or name.startswith(_FORBIDDEN_BUILD_ENVIRONMENT_PREFIXES)
            or not isinstance(value, str)
            or not value
            or len(value) > 1024
            or "\x00" in value
            or "\n" in value
            or "\r" in value
            for name, value in build_environment.items()
        ):
            raise ValueError(
                f"Execution profile {name!r} build_environment must contain safe strings"
            )
        requirements = raw_spec.get("requirements")
        if type(requirements) is not str:
            raise ValueError(f"Execution profile {name!r} must declare a requirements path")
        requirements_path = _profile_asset_path(
            requirements, field="requirements", profile_name=name
        )
        _exact_pinned_requirements(requirements_path.read_text(encoding="utf-8"))

        inline_verification = raw_spec.get("verification_script")
        file_verification = raw_spec.get("verification_script_file")
        if bool(inline_verification) == bool(file_verification):
            raise ValueError(
                f"Execution profile {name!r} must declare exactly one of "
                "verification_script and verification_script_file"
            )
        if inline_verification is not None and type(inline_verification) is not str:
            raise ValueError(f"Execution profile {name!r} verification_script must be a string")
        if file_verification is not None:
            if type(file_verification) is not str:
                raise ValueError(
                    f"Execution profile {name!r} verification_script_file must be a string"
                )
            _profile_asset_path(
                file_verification,
                field="verification_script_file",
                profile_name=name,
            )

    declared_profiles = set(profiles)
    for section_name in (
        "runtime_strategy_defaults",
        "reference_backend_defaults",
    ):
        section = registry.get(section_name, {})
        if not isinstance(section, Mapping):
            raise ValueError(f"python_profiles.toml [{section_name}] must be an object")
        for selector, defaults in section.items():
            if type(selector) is not str or not selector.strip():
                raise ValueError(f"python_profiles.toml [{section_name}] keys must be strings")
            if not isinstance(defaults, Mapping):
                raise ValueError(
                    f"python_profiles.toml [{section_name}.{selector}] must be an object"
                )
            for phase, profile in defaults.items():
                if phase not in PROFILE_PHASES:
                    raise ValueError(
                        f"python_profiles.toml [{section_name}.{selector}] "
                        f"contains unsupported phase {phase!r}"
                    )
                if type(profile) is not str or not profile.strip():
                    raise ValueError(
                        f"python_profiles.toml [{section_name}.{selector}] "
                        f"phase {phase!r} must select a profile"
                    )
                if profile not in declared_profiles:
                    raise ValueError(
                        f"python_profiles.toml [{section_name}.{selector}] "
                        f"selects undeclared profile {profile!r}"
                    )


def prebuilt_python_profile_names(
    registry: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return non-default profiles prepared before network-disabled execution."""
    selected = (
        registry if registry is not None else load_python_profile_registry()
    )
    profiles = selected.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("Python profile registry is missing a profiles mapping")
    return tuple(
        sorted(
            name
            for name, spec in profiles.items()
            if name != DEFAULT_PROFILE
            and isinstance(spec, Mapping)
            and bool(spec.get("prebuild", True))
        )
    )


def _read_package_text(path_spec: str) -> str:
    path = Path(path_spec)
    if path.is_absolute():
        return path.read_text(encoding="utf-8")
    return (_PACKAGE_DIR / path_spec).read_text(encoding="utf-8")


def _read_requirements_text(path_spec: str) -> str:
    return _read_package_text(path_spec)


def _exact_pinned_requirements(requirements_text: str) -> dict[str, str]:
    """Parse the profile lock contract and reject non-hermetic requirements."""
    pinned: dict[str, str] = {}
    for line_number, raw_line in enumerate(requirements_text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _EXACT_REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                "Python profile requirements must be exact name==version pins; "
                f"line {line_number} is {raw_line!r}"
            )
        name, version = match.groups()
        if _EXACT_VERSION_RE.fullmatch(version) is None:
            raise ValueError(
                "Python profile requirements must be exact name==version pins; "
                f"line {line_number} is {raw_line!r}"
            )
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name in pinned:
            raise ValueError(f"Python profile requirements declare {name!r} more than once")
        pinned[normalized_name] = version
    return pinned


def _pinned_version_matches(expected: str, actual: str) -> bool:
    if actual == expected:
        return True
    return "+" not in expected and actual.partition("+")[0] == expected


def _verify_exact_requirements(
    profile_name: str,
    profile_python: str,
    pinned: Mapping[str, str],
) -> None:
    if not pinned:
        return
    script = (
        "import importlib.metadata as m, json, sys; "
        "from tensorrt_model_connect.python_profiles "
        "import _pinned_version_matches as matches; "
        "expected=json.loads(sys.argv[1]); "
        "actual={name:m.version(name) for name in expected}; "
        "bad={name:(expected[name], actual[name]) for name in expected "
        "if not matches(expected[name], actual[name])}; "
        "assert not bad, f'exact profile pins do not match: {bad}'"
    )
    _run_profile_command(
        [profile_python, "-c", script, json.dumps(dict(pinned), sort_keys=True)],
        description=f"verify exact package pins for Python profile {profile_name!r}",
        timeout=300,
        env=_profile_subprocess_environment(),
    )


def _prebuilt_only() -> bool:
    return os.environ.get(PREBUILT_ONLY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _profile_spec(profile_name: str) -> dict[str, Any]:
    registry = load_python_profile_registry()
    profiles = registry.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("python_profiles.toml is missing a [profiles] table")
    spec = profiles.get(profile_name)
    if not isinstance(spec, dict):
        raise ValueError(f"Execution profile {profile_name!r} is not declared")
    return spec


def _apply_declared_defaults(
    profiles: dict[str, str],
    defaults: Mapping[str, Any] | None,
    *,
    declared_profiles: set[str] | None = None,
    source: str,
) -> None:
    if not defaults:
        return
    for phase, profile in defaults.items():
        if phase not in PROFILE_PHASES:
            raise ValueError(
                f"{source} contains unsupported phase {phase!r}; expected one of {PROFILE_PHASES}"
            )
        name = str(profile).strip()
        if not name:
            raise ValueError(f"{source}[{phase!r}] must be a non-empty string")
        if declared_profiles is not None and name not in declared_profiles:
            raise ValueError(f"{source}[{phase!r}] selects undeclared profile {name!r}")
        profiles[phase] = name


def default_execution_profiles(
    *,
    family: str = "",
    runtime_strategy: str = "",
    reference_backend: str = "",
) -> dict[str, str]:
    """Return declarative default profile selections for a model case."""
    profiles = {phase: DEFAULT_PROFILE for phase in PROFILE_PHASES}
    registry = load_python_profile_registry()
    declared_profiles = set(registry["profiles"])

    if family:
        from .families import family_default_execution_profiles

        _apply_declared_defaults(
            profiles,
            family_default_execution_profiles(family),
            declared_profiles=declared_profiles,
            source=f"family metadata {family}",
        )

    sections = (
        ("runtime_strategy_defaults", str(runtime_strategy or "").strip()),
        ("reference_backend_defaults", str(reference_backend or "").strip()),
    )
    for section_name, key in sections:
        if not key:
            continue
        section = registry.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"python_profiles.toml [{section_name}] must be an object")
        defaults = section.get(key)
        _apply_declared_defaults(
            profiles,
            defaults,
            declared_profiles=declared_profiles,
            source=f"{section_name}.{key}",
        )

    return profiles


def normalize_execution_profiles(
    raw: Mapping[str, str] | None,
    *,
    family: str = "",
    runtime_strategy: str = "",
    reference_backend: str = "",
) -> dict[str, str]:
    """Return a normalized phase -> profile mapping with declarative defaults."""
    profiles = default_execution_profiles(
        family=family,
        runtime_strategy=runtime_strategy,
        reference_backend=reference_backend,
    )
    _apply_declared_defaults(profiles, raw, source="execution_profiles")
    return profiles


def _process_session_members(session_id: int) -> list[int]:
    members = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            _, _, remainder = stat_path.read_text(encoding="utf-8").rpartition(")")
            fields = remainder.split()
            if len(fields) > 3 and int(fields[3]) == session_id:
                members.append(int(stat_path.parent.name))
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
        ):
            continue
    return members


def _kill_profile_process_session(process: subprocess.Popen[str]) -> None:
    members = set(_process_session_members(process.pid))
    members.add(process.pid)
    for pid in members:
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            continue
    members.update(_process_session_members(process.pid))
    for pid in sorted(members, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _run_profile_command(
    cmd: list[str],
    *,
    description: str,
    timeout: float = 1800,
    env: Mapping[str, str] | None = None,
) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
        env=dict(env) if env is not None else None,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _kill_profile_process_session(process)
        stdout, stderr = process.communicate()
        detail = (error.stderr or stderr or error.stdout or stdout or "").strip()
        message = f"Failed to {description}: timed out after {timeout:g} seconds"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from error
    if process.returncode != 0:
        stderr = (stderr or stdout or "").strip()
        raise RuntimeError(
            f"Failed to {description}: {stderr or f'command exited with rc={process.returncode}'}"
        )


def _profile_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _profile_install_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = _profile_subprocess_environment()
    if overrides is not None:
        environment.update(overrides)
    if not environment.get("MAX_JOBS", "").strip():
        environment["MAX_JOBS"] = _DEFAULT_PROFILE_BUILD_JOBS
    _configure_targeted_nvcc(environment)
    return environment


def _cuda_arch_codes(value: str) -> tuple[str, ...]:
    """Translate numeric TORCH_CUDA_ARCH_LIST entries to nvcc codes."""
    codes: list[str] = []
    for raw in re.split(r"[;,\s]+", value.strip()):
        if not raw:
            continue
        token = raw.removesuffix("+PTX")
        match = re.fullmatch(r"([0-9]+)\.([0-9]+)", token)
        if match is None:
            return ()
        code = f"{int(match.group(1))}{match.group(2)}"
        if code not in codes:
            codes.append(code)
    return tuple(codes)


def _real_nvcc(environment: Mapping[str, str]) -> Path | None:
    configured_home = (
        environment.get("CUDA_HOME", "").strip()
        or environment.get("CUDA_PATH", "").strip()
    )
    candidates = []
    if configured_home:
        candidates.append(Path(configured_home) / "bin" / "nvcc")
    discovered_nvcc = shutil.which("nvcc", path=environment.get("PATH"))
    if discovered_nvcc:
        candidates.append(Path(discovered_nvcc))
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    return selected.resolve() if selected is not None else None


def _profile_install_identity(environment: Mapping[str, str]) -> dict[str, object]:
    """Return the effective CUDA inputs that can change installed artifacts."""
    identity: dict[str, object] = {
        "torch_cuda_arch_list": environment.get("TORCH_CUDA_ARCH_LIST", ""),
        "nvcc_arch_codes": environment.get("TRTMC_NVCC_ARCH_CODES", ""),
    }
    configured_nvcc = environment.get("TRTMC_REAL_NVCC", "").strip()
    real_nvcc = Path(configured_nvcc) if configured_nvcc else _real_nvcc(environment)
    if real_nvcc is None or not real_nvcc.is_file():
        identity["cuda_home"] = environment.get("CUDA_HOME", "")
        identity["cuda_path"] = environment.get("CUDA_PATH", "")
        return identity

    real_nvcc = real_nvcc.resolve()
    stat = real_nvcc.stat()
    identity.update(
        {
            "cuda_home": str(real_nvcc.parent.parent),
            "nvcc": str(real_nvcc),
            "nvcc_size": stat.st_size,
            "nvcc_mtime_ns": stat.st_mtime_ns,
        }
    )
    return identity


def _configure_targeted_nvcc(environment: dict[str, str]) -> None:
    """Filter hard-coded nvcc targets to the declared PyTorch arch list."""
    arch_codes = _cuda_arch_codes(environment.get("TORCH_CUDA_ARCH_LIST", ""))
    if not arch_codes:
        return

    real_nvcc = _real_nvcc(environment)
    if real_nvcc is None:
        return
    real_cuda_home = real_nvcc.parent.parent

    identity = hashlib.sha256(f"{real_nvcc}\0{','.join(arch_codes)}".encode("utf-8")).hexdigest()[
        :12
    ]
    wrapper_home = Path(tempfile.gettempdir()) / f"trtmc-cuda-arch-{identity}"
    wrapper_bin = wrapper_home / "bin"
    wrapper_nvcc = wrapper_bin / "nvcc"
    wrapper_bin.mkdir(parents=True, exist_ok=True)
    for name in ("include", "lib64"):
        source = real_cuda_home / name
        target = wrapper_home / name
        if source.is_dir() and not target.exists():
            target.symlink_to(source, target_is_directory=True)

    script = """#!/usr/bin/env bash
set -euo pipefail
args=()
saw_gencode=0
kept_gencode=0
while (($#)); do
    if [[ ( "$1" == "-gencode" || "$1" == "--generate-code" ) && $# -ge 2 && "$2" == arch=compute_* ]]; then
        saw_gencode=1
        code="${2#arch=compute_}"
        code="${code%%,*}"
        if [[ ",${TRTMC_NVCC_ARCH_CODES}," == *",${code},"* ]]; then
            args+=("$1" "$2")
            kept_gencode=1
        fi
        shift 2
        continue
    fi
    if [[ "$1" == -gencode=arch=compute_* || "$1" == --generate-code=arch=compute_* ]]; then
        saw_gencode=1
        code="${1#*=arch=compute_}"
        code="${code%%,*}"
        if [[ ",${TRTMC_NVCC_ARCH_CODES}," == *",${code},"* ]]; then
            args+=("$1")
            kept_gencode=1
        fi
        shift
        continue
    fi
    if [[ ( "$1" == "-arch" || "$1" == "--gpu-architecture" ) && $# -ge 2 && "$2" == sm_* ]]; then
        saw_gencode=1
        code="${2#sm_}"
        if [[ ",${TRTMC_NVCC_ARCH_CODES}," == *",${code},"* ]]; then
            args+=("$1" "$2")
            kept_gencode=1
        fi
        shift 2
        continue
    fi
    if [[ "$1" == -arch=sm_* || "$1" == --gpu-architecture=sm_* ]]; then
        saw_gencode=1
        code="${1#*=sm_}"
        if [[ ",${TRTMC_NVCC_ARCH_CODES}," == *",${code},"* ]]; then
            args+=("$1")
            kept_gencode=1
        fi
        shift
        continue
    fi
    args+=("$1")
    shift
done
if ((saw_gencode && !kept_gencode)); then
    echo "nvcc command has no requested CUDA architecture (${TRTMC_NVCC_ARCH_CODES})" >&2
    exit 2
fi
exec "${TRTMC_REAL_NVCC}" "${args[@]}"
"""
    if not wrapper_nvcc.is_file() or wrapper_nvcc.read_text(encoding="utf-8") != script:
        temporary = wrapper_nvcc.with_name(f"nvcc.tmp.{os.getpid()}")
        temporary.write_text(script, encoding="utf-8")
        temporary.chmod(0o755)
        temporary.replace(wrapper_nvcc)

    environment["CUDA_HOME"] = str(wrapper_home)
    environment["CUDA_PATH"] = str(wrapper_home)
    environment["TRTMC_REAL_NVCC"] = str(real_nvcc)
    environment["TRTMC_NVCC_ARCH_CODES"] = ",".join(arch_codes)


def _python_site_packages(python: str) -> list[str]:
    script = "import json, site; print(json.dumps(site.getsitepackages()))"
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=_profile_subprocess_environment(),
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to query site-packages for {python}: {stderr or 'unknown error'}"
        )
    payload = result.stdout.strip()
    return [str(Path(p).absolute()) for p in json.loads(payload)]


def _inherited_overlay_paths(base_paths: list[str]) -> list[str]:
    """Read only overlays emitted by this module, never arbitrary sys.path."""
    inherited: list[str] = []
    for base_path in base_paths:
        overlay_file = Path(base_path) / "trtmc_base_python_overlay.pth"
        if not overlay_file.is_file():
            continue
        for line_number, raw_line in enumerate(
            overlay_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute() or line.startswith("import "):
                raise ValueError(
                    f"Unsafe inherited profile overlay line {line_number} in "
                    f"{overlay_file}: {raw_line!r}"
                )
            normalized = str(path)
            if normalized not in inherited:
                inherited.append(normalized)
    return inherited


def _write_base_site_packages_overlay(base_python: str, profile_python: str) -> None:
    base_paths = _python_site_packages(base_python)
    inherited_paths = _inherited_overlay_paths(base_paths)
    profile_paths = _python_site_packages(profile_python)
    if not profile_paths:
        raise RuntimeError(
            f"Failed to determine site-packages for profile interpreter {profile_python}"
        )
    overlay_paths = list(dict.fromkeys([*base_paths, *inherited_paths]))
    package_root = str(_PACKAGE_DIR.parent)
    if package_root not in overlay_paths:
        overlay_paths.append(package_root)
    overlay_file = Path(profile_paths[0]) / "trtmc_base_python_overlay.pth"
    overlay_file.write_text(
        "\n".join(overlay_paths) + "\n",
        encoding="utf-8",
    )


def _materialize_venv_profile(
    profile_name: str,
    spec: Mapping[str, Any],
    base_python: str,
    *,
    on_create: Callable[[str], None] | None = None,
) -> str:
    base_python = _absolute_python(base_python)
    if not base_python:
        raise ValueError(f"Execution profile {profile_name!r} requires a base Python interpreter")

    requirements_spec = str(spec.get("requirements", "") or "").strip()
    if not requirements_spec:
        raise ValueError(f"Execution profile {profile_name!r} must declare a requirements file")
    requirements_text = _read_requirements_text(requirements_spec)
    pinned_requirements = _exact_pinned_requirements(requirements_text)
    verification_script = str(spec.get("verification_script", "") or "").strip()
    verification_script_file = str(spec.get("verification_script_file", "") or "").strip()
    if verification_script_file:
        if verification_script:
            raise ValueError(
                f"Execution profile {profile_name!r} declares both "
                "verification_script and verification_script_file"
            )
        verification_script = _read_package_text(verification_script_file).strip()
    system_site_packages = bool(spec.get("system_site_packages", True))
    build_environment = {
        str(name): str(value)
        for name, value in dict(spec.get("build_environment", {})).items()
    }
    install_environment = _profile_install_environment(build_environment)
    install_identity = _profile_install_identity(install_environment)

    hash_input = "\n".join(
        [
            base_python,
            _PROFILE_LAYOUT_VERSION,
            requirements_spec,
            requirements_text,
            verification_script,
            f"system_site_packages={int(system_site_packages)}",
            json.dumps(build_environment, separators=(",", ":"), sort_keys=True),
            json.dumps(install_identity, separators=(",", ":"), sort_keys=True),
        ]
    ).encode("utf-8")
    profile_hash = hashlib.sha256(hash_input).hexdigest()[:12]

    root = profile_root()
    env_dir = root / f"{profile_name}-{profile_hash}"
    python_path = env_dir / "bin" / "python"
    ready_path = env_dir / ".ready"
    lock_path = root / f"{profile_name}-{profile_hash}.lock"

    # Network-disabled proofs mount a separately prepared profile root read-only.
    if ready_path.is_file() and python_path.is_file():
        return str(python_path.absolute())
    if _prebuilt_only():
        raise RuntimeError(
            f"Execution profile {profile_name!r} is not prebuilt for this source "
            f"at {env_dir}. Prepare the declared profiles before entering the "
            "network-disabled execution lane."
        )

    root.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if ready_path.is_file() and python_path.is_file():
            return str(python_path.absolute())

        if on_create is not None:
            on_create(profile_name)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{profile_name}-", dir=str(root)))
        tmp_python = tmp_dir / "bin" / "python"
        requirements_file = tmp_dir / "requirements.lock.txt"
        requirements_file.write_text(requirements_text, encoding="utf-8")
        try:
            create_cmd = [base_python, "-m", "venv", str(tmp_dir)]
            _run_profile_command(
                create_cmd,
                description=f"create Python profile {profile_name!r}",
                timeout=300,
                env=_profile_subprocess_environment(),
            )

            if system_site_packages:
                _write_base_site_packages_overlay(base_python, str(tmp_python))

            if requirements_text.strip():
                _run_profile_command(
                    [
                        str(tmp_python),
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--quiet",
                        "--no-deps",
                        "--no-build-isolation",
                        "-r",
                        str(requirements_file),
                    ],
                    description=f"install Python profile {profile_name!r}",
                    timeout=_PROFILE_INSTALL_TIMEOUT_SECONDS,
                    env=install_environment,
                )

            _verify_exact_requirements(
                profile_name,
                str(tmp_python),
                pinned_requirements,
            )

            if verification_script:
                _run_profile_command(
                    [str(tmp_python), "-c", verification_script],
                    description=f"verify Python profile {profile_name!r}",
                    timeout=300,
                    env=_profile_install_environment(),
                )

            ready_path_tmp = tmp_dir / ".ready"
            ready_path_tmp.write_text(
                f"profile={profile_name}\nbase_python={base_python}\n",
                encoding="utf-8",
            )

            if env_dir.exists():
                shutil.rmtree(env_dir)
            tmp_dir.rename(env_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    return str(python_path.absolute())


def resolve_profile_python(
    profile_name: str,
    base_python: str,
    *,
    on_create: Callable[[str], None] | None = None,
) -> str:
    """Resolve a symbolic profile name to a Python executable path."""
    name = _normalize_profile_name(profile_name)
    if name == DEFAULT_PROFILE:
        return _absolute_python(base_python)

    for env_var in profile_env_var_candidates(name):
        configured = os.environ.get(env_var, "").strip()
        if not configured:
            continue
        path = Path(configured).absolute()
        if not path.exists():
            raise ValueError(
                f"Execution profile {name!r} points to missing interpreter via {env_var}: {path}"
            )
        return str(path)

    spec = _profile_spec(name)
    kind = str(spec.get("kind", "venv") or "venv").strip().lower()
    if kind in {"base", "passthrough"}:
        return _absolute_python(base_python)
    if kind != "venv":
        raise ValueError(f"Execution profile {name!r} declares unsupported kind {kind!r}")
    return _materialize_venv_profile(
        name,
        spec,
        base_python,
        on_create=on_create,
    )


def resolve_case_profile_names(case: Any) -> dict[str, str]:
    """Resolve symbolic phase -> profile names for a case-like object."""
    return normalize_execution_profiles(
        getattr(case, "execution_profiles", None),
        family=str(getattr(case, "family", "") or ""),
        runtime_strategy=str(getattr(case, "runtime_strategy", "") or ""),
        reference_backend=str(getattr(case, "reference_backend", "") or ""),
    )


def resolve_case_python_profiles(case: Any, base_python: str) -> dict[str, str]:
    """Resolve all execution profile interpreters for a case-like object."""
    profiles = resolve_case_profile_names(case)
    return {
        phase: resolve_profile_python(profile, base_python) for phase, profile in profiles.items()
    }
