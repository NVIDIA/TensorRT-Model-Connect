# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative Python profile registry and cached materialization helpers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

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
_PROFILE_LAYOUT_VERSION = "overlay-v3-exact-pins"
_EXACT_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9,._-]+\])?==([^\s;]+)$"
)


def _absolute_python(path: str) -> str:
    return str(Path(path).absolute()) if path else ""


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
    """Load the declarative profile registry bundled with the Python builder."""
    registry_file = _PACKAGE_DIR / "python_profiles.toml"
    with registry_file.open("rb") as f:
        registry = tomllib.load(f)
    if not isinstance(registry, dict):
        raise ValueError("python_profiles.toml must decode to an object")
    registry = dict(registry)
    profiles = dict(registry.get("profiles", {}))
    if not isinstance(profiles, dict):
        raise ValueError("python_profiles.toml is missing a [profiles] table")

    from .families import family_python_profile_specs

    for name, spec in family_python_profile_specs().items():
        if name in profiles:
            raise ValueError(
                f"Execution profile {name!r} is declared in both "
                "python_profiles.toml and family metadata"
            )
        profiles[name] = spec
    registry["profiles"] = profiles
    return registry


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
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        if normalized_name in pinned:
            raise ValueError(
                f"Python profile requirements declare {name!r} more than once"
            )
        pinned[normalized_name] = version
    return pinned


def _verify_exact_requirements(
    profile_name: str,
    profile_python: str,
    pinned: Mapping[str, str],
) -> None:
    if not pinned:
        return
    script = (
        "import importlib.metadata as m, json, sys; "
        "expected=json.loads(sys.argv[1]); "
        "actual={name:m.version(name) for name in expected}; "
        "bad={name:(expected[name], actual[name]) for name in expected "
        "if actual[name] != expected[name]}; "
        "assert not bad, f'exact profile pins do not match: {bad}'"
    )
    _run_profile_command(
        [profile_python, "-c", script, json.dumps(dict(pinned), sort_keys=True)],
        description=f"verify exact package pins for Python profile {profile_name!r}",
        timeout=300,
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
    source: str,
) -> None:
    if not defaults:
        return
    for phase, profile in defaults.items():
        if phase not in PROFILE_PHASES:
            raise ValueError(
                f"{source} contains unsupported phase {phase!r}; "
                f"expected one of {PROFILE_PHASES}"
            )
        name = str(profile).strip()
        if not name:
            raise ValueError(f"{source}[{phase!r}] must be a non-empty string")
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

    if family:
        from .families import family_default_execution_profiles

        _apply_declared_defaults(
            profiles,
            family_default_execution_profiles(family),
            source=f"family metadata {family}",
        )

    sections = (
        ("family_defaults", str(family or "").strip()),
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


def _run_profile_command(
    cmd: list[str],
    *,
    description: str,
    timeout: int = 1800,
) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to {description}: "
            f"{stderr or f'command exited with rc={result.returncode}'}"
        )


def _python_site_packages(python: str) -> list[str]:
    script = (
        "import json, site; "
        "print(json.dumps(site.getsitepackages()))"
    )
    result = subprocess.run(
        [python, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to query site-packages for {python}: {stderr or 'unknown error'}"
        )
    payload = result.stdout.strip()
    return [str(Path(p).absolute()) for p in json.loads(payload)]


def _write_base_site_packages_overlay(base_python: str, profile_python: str) -> None:
    base_paths = _python_site_packages(base_python)
    profile_paths = _python_site_packages(profile_python)
    if not profile_paths:
        raise RuntimeError(
            f"Failed to determine site-packages for profile interpreter {profile_python}"
        )
    package_root = str(_PACKAGE_DIR.parent)
    if package_root not in base_paths:
        base_paths.append(package_root)
    overlay_file = Path(profile_paths[0]) / "trtmc_base_python_overlay.pth"
    overlay_file.write_text(
        "\n".join(base_paths) + "\n",
        encoding="utf-8",
    )


def _materialize_venv_profile(
    profile_name: str,
    spec: Mapping[str, Any],
    base_python: str,
) -> str:
    base_python = _absolute_python(base_python)
    if not base_python:
        raise ValueError(
            f"Execution profile {profile_name!r} requires a base Python interpreter"
        )

    requirements_spec = str(spec.get("requirements", "") or "").strip()
    if not requirements_spec:
        raise ValueError(
            f"Execution profile {profile_name!r} must declare a requirements file"
        )
    requirements_text = _read_requirements_text(requirements_spec)
    pinned_requirements = _exact_pinned_requirements(requirements_text)
    verification_script = str(spec.get("verification_script", "") or "").strip()
    verification_script_file = str(
        spec.get("verification_script_file", "") or ""
    ).strip()
    if verification_script_file:
        if verification_script:
            raise ValueError(
                f"Execution profile {profile_name!r} declares both "
                "verification_script and verification_script_file"
            )
        verification_script = _read_package_text(verification_script_file).strip()
    system_site_packages = bool(spec.get("system_site_packages", True))

    hash_input = "\n".join(
        [
            base_python,
            _PROFILE_LAYOUT_VERSION,
            requirements_spec,
            requirements_text,
            verification_script,
            f"system_site_packages={int(system_site_packages)}",
        ]
    ).encode("utf-8")
    profile_hash = hashlib.sha256(hash_input).hexdigest()[:12]

    root = profile_root()
    env_dir = root / f"{profile_name}-{profile_hash}"
    python_path = env_dir / "bin" / "python"
    ready_path = env_dir / ".ready"
    lock_path = root / f"{profile_name}-{profile_hash}.lock"

    # Model-proof containers mount the source read-only and disable networking.
    # A matching image-baked profile therefore needs no writable lock or cache.
    if ready_path.is_file() and python_path.is_file():
        return str(python_path.absolute())
    if _prebuilt_only():
        raise RuntimeError(
            f"Execution profile {profile_name!r} is not prebuilt for this source "
            f"at {env_dir}. The CI image is stale or incomplete; rebuild it from "
            "the current Dockerfile and family-owned profile locks."
        )

    root.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if ready_path.is_file() and python_path.is_file():
            return str(python_path.absolute())

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


def resolve_profile_python(profile_name: str, base_python: str) -> str:
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
        raise ValueError(
            f"Execution profile {name!r} declares unsupported kind {kind!r}"
        )
    return _materialize_venv_profile(name, spec, base_python)


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
        phase: resolve_profile_python(profile, base_python)
        for phase, profile in profiles.items()
    }
