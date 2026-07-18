# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate and validate the AOT Wan2.2 TensorRT plugin companion DSO.

The companion is produced by the Model-Connect CMake/package build.  Bundle
creation only loads that already-qualified binary; it never invokes CMake,
NVCC, or a host compiler.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


_FAMILY = "wan2_2_ti2v"
_COMPANION_GLOB = "libtrtmc_model_wan2_2_ti2v_plugins_trt*_*.so"
_DEVELOPMENT_OVERRIDE_ENV = "TRTMC_WAN22_PLUGIN_LIBRARY_DEV"
_LEGACY_OVERRIDE_ENVS = (
    "TRTMC_WAN22_UMT5_CUDA_PLUGIN_LIBRARY",
    "TRTMC_WAN22_DIT_CUDA_PLUGIN_LIBRARY",
    "TRTMC_WAN22_VAE_CUDA_PLUGIN_LIBRARY",
)
_CONTRACT_KEYS = {
    "schema",
    "family",
    "semantic_abi",
    "source_digest",
    "creator_set",
    "runtime_abi",
    "cuda_architectures",
}
_RUNTIME_ABI_KEYS = {
    "tensorrt_major",
    "tensorrt_minor",
    "cuda_major",
    "cudnn_major",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_ABI_RE = re.compile(
    r"^tensorrt=([0-9]+)\.([0-9]+);cuda=([0-9]+);cudnn=([0-9]+)$"
)
_FILENAME_ABI_RE = re.compile(r"_plugins_trt([0-9]+)_([0-9]+)\.so$")
_MISSING_DEPENDENCY_RE = re.compile(
    r"(?P<soname>lib[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)+): "
    r"cannot open shared object file"
)
_ALLOWED_DEPENDENCY_RE = re.compile(
    r"^(?:"
    r"libnvinfer(?:_dispatch|_lean|_plugin)?|"
    r"libcudnn(?:_[a-z0-9_]+)?|"
    r"libcublas(?:Lt)?|libcudart|libnvrtc(?:-builtins)?|libcuda"
    r")\.so(?:\.[0-9]+)+$"
)


@dataclass(frozen=True)
class Wan22PluginCompanion:
    path: Path
    load_path: Path
    elf_bytes: bytes
    elf_sha256: str
    backing_fd: int
    contract: dict[str, Any]
    handle: ctypes.CDLL


_LOADED_COMPANIONS: dict[Path, Wan22PluginCompanion] = {}
_POISONED_COMPANION: tuple[Path, ctypes.CDLL, int, str] | None = None
_PRELOADED_DEPENDENCIES: dict[str, ctypes.CDLL] = {}


def _append_directory(candidates: list[Path], directory: Path) -> None:
    candidates.append(directory)
    if directory.name != _FAMILY:
        candidates.append(directory / _FAMILY)


def _parent_executable() -> Path | None:
    try:
        return Path(f"/proc/{os.getppid()}/exe").resolve(strict=True)
    except OSError:
        return None


def _deduplicated_directories(directories: list[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _candidate_directory_tiers() -> tuple[tuple[Path, ...], ...]:
    tiers: list[tuple[Path, ...]] = []

    configured_directories: list[Path] = []
    configured = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    if configured:
        for item in configured.split(os.pathsep):
            if item:
                _append_directory(configured_directories, Path(item).expanduser())
    if configured_directories:
        tiers.append(_deduplicated_directories(configured_directories))

    parent_executable = _parent_executable()
    if parent_executable is not None:
        parent_dir = parent_executable.parent
        parent_directories: list[Path] = []
        _append_directory(parent_directories, parent_dir)
        _append_directory(parent_directories, parent_dir / "models")
        tiers.append(_deduplicated_directories(parent_directories))

    package_dir = Path(__file__).resolve().parents[2]
    installed_directories: list[Path] = []
    _append_directory(installed_directories, package_dir / "bin")
    _append_directory(installed_directories, package_dir / "models")

    prefix = Path(sys.prefix)
    _append_directory(installed_directories, prefix / "lib" / "trtmc" / "models")
    _append_directory(installed_directories, prefix / "lib64" / "trtmc" / "models")
    if parent_executable is not None:
        _append_directory(
            installed_directories,
            parent_executable.parent.parent / "lib" / "trtmc" / "models",
        )
    tiers.append(_deduplicated_directories(installed_directories))

    cwd = Path.cwd()
    fallback_directories: list[Path] = []
    _append_directory(fallback_directories, cwd / "build" / "models")
    for build_dir in sorted(cwd.glob("build*")):
        if build_dir.is_dir():
            _append_directory(fallback_directories, build_dir / "models")
    tiers.append(_deduplicated_directories(fallback_directories))
    return tuple(tier for tier in tiers if tier)


def _candidate_directories() -> tuple[Path, ...]:
    return tuple(directory for tier in _candidate_directory_tiers() for directory in tier)


def _python_tensorrt_abi() -> tuple[int, int] | None:
    try:
        from tensorrt_model_connect import trt_compat

        version = str(trt_compat.get_trt().__version__)
    except (AttributeError, ImportError, RuntimeError):
        return None
    match = re.match(r"^([0-9]+)\.([0-9]+)", version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _resolve_companion_path() -> Path:
    legacy = [name for name in _LEGACY_OVERRIDE_ENVS if os.environ.get(name)]
    if legacy:
        raise RuntimeError(
            "The per-component Wan2.2 CUDA plugin overrides were removed; "
            "Model-Connect now ships one AOT model companion DSO. Unset "
            + ", ".join(legacy)
        )

    override = os.environ.get(_DEVELOPMENT_OVERRIDE_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{_DEVELOPMENT_OVERRIDE_ENV} does not exist: {path}")
        return path

    installed_abi = _python_tensorrt_abi()
    for tier in _candidate_directory_tiers():
        found: list[Path] = []
        seen: set[Path] = set()
        for directory in tier:
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob(_COMPANION_GLOB)):
                resolved = candidate.resolve()
                if resolved.is_file() and resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
        if installed_abi is not None:
            found = [
                path
                for path in found
                if (match := _FILENAME_ABI_RE.search(path.name))
                and (int(match.group(1)), int(match.group(2))) == installed_abi
            ]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise RuntimeError(
                "Multiple Wan2.2 plugin companions match the active TensorRT ABI "
                "in the same search tier: "
                + ", ".join(str(path) for path in found)
            )

    searched = ", ".join(str(path) for path in _candidate_directories())
    raise FileNotFoundError(
        "Wan2.2 AOT TensorRT plugin companion was not found. Build/install "
        "the trtmc_model_plugins target for the active TensorRT ABI. "
        f"Searched: {searched}"
    )


def _module_library_directories(module: str) -> list[Path]:
    """Return native-library directories owned by an installed Python package."""

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return []
    if spec is None:
        return []
    roots: list[Path] = []
    if spec.submodule_search_locations:
        roots.extend(Path(location) for location in spec.submodule_search_locations)
    elif spec.origin:
        roots.append(Path(spec.origin).parent)
    directories: list[Path] = []
    for root in roots:
        directories.extend((root, root / "lib", root / "lib64"))
    return directories


def _dependency_directory_tiers(companion_path: Path) -> tuple[tuple[Path, ...], ...]:
    """Search trusted configured/package/system roots without mutating loader state."""

    configured: list[Path] = [companion_path.parent]
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if item:
            configured.append(Path(item))
    for name in ("TRTMC_TRT_LIBRARY", "TRTMC_CUDART_LIBRARY"):
        value = os.environ.get(name)
        if value:
            configured.append(Path(value).expanduser().parent)
    for name in ("TRTMC_TRT_LIBRARY_DIR", "TRT_LIB_DIR"):
        value = os.environ.get(name)
        if value:
            configured.append(Path(value).expanduser())

    packaged: list[Path] = []
    for module in (
        "tensorrt_libs",
        "nvidia.cudnn",
        "nvidia.cublas",
        "nvidia.cuda_runtime",
        "nvidia.cuda_nvrtc",
    ):
        packaged.extend(_module_library_directories(module))
    for site_packages in (Path(sys.prefix) / "lib").glob("python*/site-packages"):
        packaged.extend(
            (
                site_packages / "tensorrt_libs",
                site_packages / "nvidia" / "cudnn" / "lib",
                site_packages / "nvidia" / "cublas" / "lib",
                site_packages / "nvidia" / "cuda_runtime" / "lib",
                site_packages / "nvidia" / "cuda_nvrtc" / "lib",
            )
        )

    system = [
        Path("/usr/local/cuda/lib64"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/lib/aarch64-linux-gnu"),
        Path("/lib/x86_64-linux-gnu"),
    ]
    system.extend(Path("/usr/local").glob("cuda*/targets/*/lib"))
    return tuple(
        _deduplicated_directories(tier)
        for tier in (configured, packaged, system)
        if tier
    )


def _resolve_dependency_path(soname: str, companion_path: Path) -> Path:
    if not _ALLOWED_DEPENDENCY_RE.fullmatch(soname):
        raise RuntimeError(
            "Wan2.2 companion requested a non-NVIDIA or unversioned dependency: "
            f"{soname}"
        )
    for tier in _dependency_directory_tiers(companion_path):
        matches: list[Path] = []
        seen: set[Path] = set()
        for directory in tier:
            candidate = directory / soname
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                matches.append(resolved)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Wan2.2 dependency {soname} is ambiguous in one search tier: "
                + ", ".join(str(path) for path in matches)
            )
    raise FileNotFoundError(
        f"Wan2.2 companion dependency {soname} was not found in configured, "
        "installed-package, or CUDA system library directories"
    )


def _missing_dependency(error: OSError) -> str | None:
    match = _MISSING_DEPENDENCY_RE.search(str(error))
    return match.group("soname") if match else None


def _capture_companion_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise ValueError(f"Wan2.2 companion must be a non-empty regular file: {path}")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise OSError(f"Wan2.2 companion changed while being captured: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise OSError(f"Wan2.2 companion grew while being captured: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _create_sealed_memfd(elf_bytes: bytes) -> tuple[int, Path]:
    if not elf_bytes:
        raise ValueError("Wan2.2 companion ELF image is empty")
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("Wan2.2 companion requires Linux memfd_create for exact-byte loading")
    base_flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(
        os, "MFD_ALLOW_SEALING", 0x0002
    )
    try:
        descriptor = os.memfd_create("trtmc-wan2-2-builder-plugins", base_flags | 0x0010)
    except OSError as error:
        if error.errno != errno.EINVAL:
            raise
        # Linux kernels older than 6.3 do not know MFD_EXEC and permit
        # executable mappings by default.
        descriptor = os.memfd_create("trtmc-wan2-2-builder-plugins", base_flags)
    try:
        written = 0
        while written < len(elf_bytes):
            count = os.write(descriptor, elf_bytes[written:])
            if count <= 0:
                raise OSError("Wan2.2 companion memfd write made no progress")
            written += count
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0x0001)
            | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
            | getattr(fcntl, "F_SEAL_GROW", 0x0004)
            | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
        )
        fcntl.fcntl(descriptor, getattr(fcntl, "F_ADD_SEALS", 1033), seals)
        actual_seals = fcntl.fcntl(descriptor, getattr(fcntl, "F_GET_SEALS", 1034))
        if actual_seals & seals != seals:
            raise OSError("Wan2.2 companion memfd did not retain all required seals")
        return descriptor, Path(f"/proc/self/fd/{descriptor}")
    except Exception:
        os.close(descriptor)
        raise


def _preload_dependency(soname: str, companion_path: Path, chain: tuple[str, ...] = ()) -> None:
    """Load one exact DT_NEEDED soname and retain it for process lifetime."""

    if soname in _PRELOADED_DEPENDENCIES:
        return
    if soname in chain or len(chain) >= 16:
        raise RuntimeError(
            "Wan2.2 companion dependency cycle or excessive dependency depth: "
            + " -> ".join((*chain, soname))
        )
    if not _ALLOWED_DEPENDENCY_RE.fullmatch(soname):
        raise RuntimeError(
            "Wan2.2 companion requested a non-NVIDIA or unversioned dependency: "
            f"{soname}"
        )

    mode = ctypes.RTLD_GLOBAL | getattr(os, "RTLD_NOW", 0)
    last_error: OSError | None = None
    for request in (soname, None):
        if request is None:
            try:
                request = str(_resolve_dependency_path(soname, companion_path))
            except FileNotFoundError:
                # The bare soname already failed, and no trusted full-path
                # candidate exists to make resolution deterministic.
                break
        for _attempt in range(17):
            try:
                _PRELOADED_DEPENDENCIES[soname] = ctypes.CDLL(request, mode=mode)
                return
            except OSError as error:
                last_error = error
                nested = _missing_dependency(error)
                if nested is None or nested == soname:
                    break
                _preload_dependency(nested, companion_path, (*chain, soname))
    detail = str(last_error) if last_error is not None else "no loader candidate"
    raise RuntimeError(f"Unable to preload Wan2.2 dependency {soname}: {detail}")


def _dlopen_companion(load_path: Path, source_path: Path) -> ctypes.CDLL:
    """Load a no-RPATH companion by resolving exact missing NVIDIA sonames."""

    mode = ctypes.RTLD_GLOBAL | getattr(os, "RTLD_NOW", 0)
    for _attempt in range(17):
        try:
            return ctypes.CDLL(str(load_path), mode=mode)
        except OSError as error:
            soname = _missing_dependency(error)
            if soname is None:
                raise
            _preload_dependency(soname, source_path)
    raise RuntimeError("Wan2.2 companion dependency resolution exceeded 16 attempts")


def _string_export(library: ctypes.CDLL, name: str) -> str:
    try:
        function = getattr(library, name)
    except AttributeError as exc:
        raise ValueError(f"Wan2.2 companion is missing required export {name}") from exc
    function.argtypes = []
    function.restype = ctypes.c_char_p
    value = function()
    if not value:
        raise ValueError(f"Wan2.2 companion export {name} returned null")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Wan2.2 companion export {name} is not UTF-8") from exc


def _int_export(library: ctypes.CDLL, name: str) -> int:
    try:
        function = getattr(library, name)
    except AttributeError as exc:
        raise ValueError(f"Wan2.2 companion is missing required export {name}") from exc
    function.argtypes = []
    function.restype = ctypes.c_int
    return int(function())


def _validate_contract(contract: object) -> dict[str, Any]:
    if not isinstance(contract, dict) or set(contract) != _CONTRACT_KEYS:
        raise ValueError(
            "Wan2.2 plugin manifest has unsupported keys; expected "
            f"{sorted(_CONTRACT_KEYS)}"
        )
    if contract["schema"] != 1 or contract["family"] != _FAMILY:
        raise ValueError("Wan2.2 plugin manifest has an unsupported schema or family")
    for key in ("semantic_abi", "creator_set"):
        if not isinstance(contract[key], str) or not contract[key]:
            raise ValueError(f"Wan2.2 plugin manifest {key} must be a non-empty string")
    if not isinstance(contract["source_digest"], str) or not _SHA256_RE.fullmatch(
        contract["source_digest"]
    ):
        raise ValueError("Wan2.2 plugin source_digest must be a lowercase SHA256")

    creator_entries = contract["creator_set"].split(";")
    if any(not entry or entry.count(":") != 2 for entry in creator_entries):
        raise ValueError("Wan2.2 plugin creator_set is not canonical name:version:namespace data")
    if creator_entries != sorted(creator_entries) or len(set(creator_entries)) != len(
        creator_entries
    ):
        raise ValueError("Wan2.2 plugin creator_set must be sorted and unique")

    runtime_abi = contract["runtime_abi"]
    if not isinstance(runtime_abi, dict) or set(runtime_abi) != _RUNTIME_ABI_KEYS:
        raise ValueError("Wan2.2 plugin runtime_abi has an unsupported schema")
    if any(
        not isinstance(runtime_abi[key], int) or isinstance(runtime_abi[key], bool)
        for key in _RUNTIME_ABI_KEYS
    ):
        raise ValueError("Wan2.2 plugin runtime_abi values must be integers")
    if runtime_abi["tensorrt_minor"] < 0 or any(
        runtime_abi[key] < 1
        for key in ("tensorrt_major", "cuda_major", "cudnn_major")
    ):
        raise ValueError(
            "Wan2.2 plugin runtime ABI majors must be positive and TensorRT minor nonnegative"
        )
    architectures = contract["cuda_architectures"]
    if architectures != [103, 110]:
        raise ValueError("Wan2.2 plugin companion must contain real SM103 and SM110 cubins")
    return contract


def _load_companion(path: Path) -> Wan22PluginCompanion:
    global _POISONED_COMPANION

    resolved = path.resolve()
    if _POISONED_COMPANION is not None:
        poisoned_path, _poisoned_handle, _poisoned_fd, error = _POISONED_COMPANION
        raise RuntimeError(
            "Wan2.2 TensorRT plugin registry is unusable after an earlier "
            f"load failure: path={poisoned_path}, error={error}"
        )
    if resolved in _LOADED_COMPANIONS:
        return _LOADED_COMPANIONS[resolved]
    if _LOADED_COMPANIONS:
        active = next(iter(_LOADED_COMPANIONS.values()))
        raise RuntimeError(
            "Wan2.2 TensorRT creators are process-global; refusing to load a "
            "different companion after one is active: "
            f"loaded={active.path}, requested={resolved}"
        )
    elf_bytes = _capture_companion_bytes(resolved)
    elf_sha256 = hashlib.sha256(elf_bytes).hexdigest()
    backing_fd, load_path = _create_sealed_memfd(elf_bytes)
    try:
        library = _dlopen_companion(load_path, resolved)
    except Exception:
        os.close(backing_fd)
        raise
    try:
        search_path_state = _int_export(
            library, "trtmc_wan22_plugin_runtime_search_path_state"
        )
        if search_path_state != 0:
            detail = "contains DT_RPATH/DT_RUNPATH" if search_path_state > 0 else "is unknown"
            raise ValueError(
                "Wan2.2 companion runtime search-path state "
                f"{detail}; distributable companions must contain neither tag"
            )
        try:
            contract = _validate_contract(
                json.loads(_string_export(library, "trtmc_wan22_plugin_manifest_json"))
            )
        except json.JSONDecodeError as exc:
            raise ValueError("Wan2.2 plugin manifest export is not valid JSON") from exc

        exports = {
            "semantic_abi": "trtmc_wan22_plugin_semantic_abi",
            "source_digest": "trtmc_wan22_plugin_source_digest",
            "creator_set": "trtmc_wan22_plugin_creator_set",
        }
        for field, export in exports.items():
            if _string_export(library, export) != contract[field]:
                raise ValueError(f"Wan2.2 plugin export {export} disagrees with its manifest")

        runtime = _string_export(library, "trtmc_wan22_plugin_runtime_abi")
        match = _RUNTIME_ABI_RE.fullmatch(runtime)
        if match is None:
            raise ValueError(f"Wan2.2 plugin runtime ABI export is invalid: {runtime!r}")
        actual = {
            "tensorrt_major": int(match.group(1)),
            "tensorrt_minor": int(match.group(2)),
            "cuda_major": int(match.group(3)),
            "cudnn_major": int(match.group(4)),
        }
        if actual != contract["runtime_abi"]:
            raise ValueError(
                "Wan2.2 plugin companion was built for a different loaded runtime ABI: "
                f"built={contract['runtime_abi']}, loaded={actual}"
            )

        _validate_registered_creators(contract)
    except Exception as exc:
        # Plugin static initializers may already have registered process-global
        # creators. Retain the handle so registry vtables stay mapped, poison
        # this process, and reject every later load attempt.
        _POISONED_COMPANION = (resolved, library, backing_fd, str(exc))
        raise

    companion = Wan22PluginCompanion(
        resolved,
        load_path,
        elf_bytes,
        elf_sha256,
        backing_fd,
        contract,
        library,
    )
    _LOADED_COMPANIONS[resolved] = companion
    return companion


def _validate_registered_creators(contract: dict[str, Any]) -> None:
    """Prove every creator named by the fingerprint registered on dlopen."""

    from tensorrt_model_connect import trt_compat

    registry = trt_compat.get_trt().get_plugin_registry()
    getter = getattr(registry, "get_creator", None)
    if getter is None:
        getter = getattr(registry, "get_plugin_creator", None)
    if getter is None:
        raise RuntimeError("TensorRT plugin registry does not expose creator lookup")
    missing = []
    for entry in contract["creator_set"].split(";"):
        name, version, namespace = entry.split(":", 2)
        if getter(name, version, namespace) is None:
            missing.append(entry)
    if missing:
        raise ValueError(
            "Wan2.2 companion did not register its declared TensorRT creators: "
            + ", ".join(missing)
        )


def load_wan22_plugin_companion(*, verbose: bool = False) -> Wan22PluginCompanion:
    """Load the installed AOT companion and return its source/ABI contract."""

    companion = _load_companion(_resolve_companion_path())
    if verbose:
        print(
            "[wan2.2] AOT TensorRT plugins: "
            f"{companion.path} (elf={companion.elf_sha256}, "
            f"source={companion.contract['source_digest']})"
        )
    return companion


__all__ = ["Wan22PluginCompanion", "load_wan22_plugin_companion"]
