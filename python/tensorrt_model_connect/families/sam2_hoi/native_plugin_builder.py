# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the model-owned SAM2 HOI TensorRT native-operator plugin."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


_PLUGIN_ENV = "TRTMC_SAM2_HOI_NATIVE_PLUGIN_LIBRARY"
_BUILD_DIR_ENV = "TRTMC_SAM2_HOI_NATIVE_PLUGIN_BUILD_DIR"
_CUDA_ARCHITECTURES = ("89", "100")
_RECEIPT_NAME = "build-receipt.json"
_PROCESS_MAPS = Path("/proc/self/maps")


def _configured_build_base() -> Path:
    configured = os.environ.get(_BUILD_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / (f"trtmc-sam2-hoi-native-plugin-{os.geteuid()}")


def _unsafe_cache_path(path: Path, reason: str) -> RuntimeError:
    return RuntimeError(f"Unsafe SAM2 HOI native plugin cache path {path}: {reason}")


def _validate_directory(
    path: Path,
    metadata: os.stat_result,
    *,
    effective_uid: int,
    system_uid: int,
    private: bool,
) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise _unsafe_cache_path(path, "symbolic links are not allowed")
    if not stat.S_ISDIR(metadata.st_mode):
        raise _unsafe_cache_path(path, "path component is not a directory")
    if private:
        if metadata.st_uid != effective_uid:
            raise _unsafe_cache_path(
                path,
                f"owner uid {metadata.st_uid} does not match effective uid {effective_uid}",
            )
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise _unsafe_cache_path(path, "directory is group/world writable")
        return

    if metadata.st_uid not in {effective_uid, system_uid}:
        raise _unsafe_cache_path(path, f"path component has untrusted owner uid {metadata.st_uid}")
    writable_by_others = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_by_others and not metadata.st_mode & stat.S_ISVTX:
        raise _unsafe_cache_path(path, "path component is group/world writable")


def _secure_private_directory(path: Path) -> Path:
    """Create and validate a private cache directory without following symlinks."""
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    anchor = Path(absolute.anchor)
    effective_uid = os.geteuid()
    try:
        anchor_metadata = anchor.lstat()
    except OSError as error:
        raise _unsafe_cache_path(anchor, f"cannot inspect path anchor: {error}") from error
    system_uid = anchor_metadata.st_uid
    _validate_directory(
        anchor,
        anchor_metadata,
        effective_uid=effective_uid,
        system_uid=system_uid,
        private=absolute == anchor,
    )

    current = anchor
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        private = index == len(parts) - 1
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                os.mkdir(current, mode=0o700)
            except FileExistsError:
                pass
            except OSError as error:
                raise _unsafe_cache_path(current, f"cannot create directory: {error}") from error
            try:
                metadata = current.lstat()
            except OSError as error:
                raise _unsafe_cache_path(current, f"cannot inspect directory: {error}") from error
        except OSError as error:
            raise _unsafe_cache_path(current, f"cannot inspect path component: {error}") from error
        _validate_directory(
            current,
            metadata,
            effective_uid=effective_uid,
            system_uid=system_uid,
            private=private,
        )
    return absolute


def _source_digest(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*")):
        relative = path.relative_to(source_dir).as_posix()
        if path.is_symlink():
            raise RuntimeError(
                f"SAM2 HOI native plugin source cannot contain a symbolic link: {relative}"
            )
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _command_identity(command: str | None, *arguments: str) -> dict[str, object]:
    if not command:
        return {"path": None, "output": "missing"}
    command_parts = shlex.split(command)
    if not command_parts:
        return {"path": None, "output": "missing"}
    resolved = shutil.which(command_parts[0]) or command_parts[0]
    try:
        completed = subprocess.run(
            [resolved, *command_parts[1:], *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        return {"path": resolved, "output": f"unavailable: {error}"}
    return {
        "path": str(Path(resolved).resolve()),
        "arguments": [*command_parts[1:], *arguments],
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def _tensorrt_identity() -> dict[str, object]:
    include_candidates = (
        Path("/usr/include/aarch64-linux-gnu/NvInferVersion.h"),
        Path("/usr/include/x86_64-linux-gnu/NvInferVersion.h"),
        Path("/usr/include/NvInferVersion.h"),
        Path("/usr/local/include/NvInferVersion.h"),
    )
    header = next((path for path in include_candidates if path.is_file()), None)
    header_identity: dict[str, object] | None = None
    runtime_header_identity: dict[str, object] | None = None
    if header is not None:
        payload = header.read_bytes()
        version_lines = [
            line.strip()
            for line in payload.decode("utf-8", errors="replace").splitlines()
            if line.startswith("#define NV_TENSORRT_")
        ]
        header_identity = {
            "path": str(header.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "version_macros": version_lines,
        }
        runtime_header = header.with_name("NvInferRuntime.h")
        if runtime_header.is_file():
            runtime_payload = runtime_header.read_bytes()
            runtime_header_identity = {
                "path": str(runtime_header.resolve()),
                "sha256": hashlib.sha256(runtime_payload).hexdigest(),
            }

    library_directories = (
        Path("/opt/venv/lib/python3.12/site-packages/tensorrt_libs"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib"),
    )
    libraries: list[dict[str, object]] = []
    seen: set[Path] = set()
    for directory in library_directories:
        for candidate in sorted(directory.glob("libnvinfer.so*")):
            try:
                resolved = candidate.resolve(strict=True)
                size = resolved.stat().st_size
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            libraries.append(
                {
                    "name": candidate.name,
                    "path": str(resolved),
                    "size": size,
                    "sha256": _file_sha256(resolved),
                }
            )
    return {
        "header": header_identity,
        "runtime_header": runtime_header_identity,
        "libraries": libraries,
    }


def _cuda_toolkit_root(nvcc_command: str) -> Path:
    """Resolve the toolkit root that will be pinned into the CMake configure."""

    command_parts = shlex.split(nvcc_command)
    if len(command_parts) != 1:
        raise RuntimeError("SAM2 HOI native plugin CUDACXX must be one direct nvcc executable path")
    executable_value = shutil.which(command_parts[0])
    if executable_value is None:
        raise RuntimeError(
            f"SAM2 HOI native plugin cannot resolve CUDA compiler {command_parts[0]!r}"
        )
    executable = Path(executable_value).resolve(strict=True)
    if executable.name != "nvcc":
        raise RuntimeError(
            "SAM2 HOI native plugin CUDACXX must resolve directly to nvcc, not a launcher"
        )
    compiler_root = executable.parent.parent
    configured = os.environ.get("CUDAToolkit_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve(strict=True)
        try:
            executable.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                "SAM2 HOI CUDACXX resolves outside the configured CUDA toolkit root"
            ) from error
    else:
        root = compiler_root
    if not root.is_dir():
        raise RuntimeError(f"SAM2 HOI CUDA toolkit root is not a directory: {root}")
    return root


def _cublaslt_identity(toolkit_root: Path) -> list[dict[str, object]]:
    """Resolve exactly the cuBLASLt DSO from the CMake-pinned CUDA toolkit."""

    machine = platform.machine().lower()
    target = "sbsa-linux" if machine in {"aarch64", "arm64"} else "x86_64-linux"
    candidates = (
        toolkit_root / "targets" / target / "lib" / "libcublasLt.so",
        toolkit_root / "lib64" / "libcublasLt.so",
        toolkit_root / "lib" / "libcublasLt.so",
    )
    resolved_candidates: dict[Path, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            resolved_candidates.setdefault(resolved, candidate)
    if not resolved_candidates:
        raise RuntimeError(
            f"SAM2 HOI native plugin cannot resolve libcublasLt.so under {toolkit_root}"
        )
    if len(resolved_candidates) != 1:
        paths = ", ".join(str(path) for path in sorted(resolved_candidates))
        raise RuntimeError(f"SAM2 HOI CUDA toolkit has ambiguous cuBLASLt DSOs: {paths}")
    resolved, link_path = next(iter(resolved_candidates.items()))
    try:
        resolved.relative_to(toolkit_root.resolve(strict=True))
    except ValueError as error:
        raise RuntimeError(
            f"SAM2 HOI cuBLASLt resolves outside the pinned CUDA toolkit: {resolved}"
        ) from error
    metadata = resolved.stat()
    return [
        {
            "name": resolved.name,
            "link_path": str(link_path),
            "path": str(resolved),
            "size": metadata.st_size,
            "sha256": _file_sha256(resolved),
        }
    ]


def _verify_configured_cublaslt(build_dir: Path, identity: dict[str, object]) -> None:
    """Fail closed unless CMake selected the exact cuBLASLt DSO in the identity."""

    cache = build_dir / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"SAM2 HOI cannot read configured CMake cache {cache}") from error
    prefix = "CUDA_cublasLt_LIBRARY:FILEPATH="
    values = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
    if len(values) != 1 or not values[0] or values[0].endswith("-NOTFOUND"):
        raise RuntimeError("SAM2 HOI CMake cache has no unique CUDA_cublasLt_LIBRARY")
    configured = Path(values[0]).resolve(strict=True)
    configured_identity = {
        "name": configured.name,
        "path": str(configured),
        "size": configured.stat().st_size,
        "sha256": _file_sha256(configured),
    }
    expected_entries = identity.get("cublaslt")
    if not isinstance(expected_entries, list) or len(expected_entries) != 1:
        raise RuntimeError("SAM2 HOI build identity has no unique cuBLASLt entry")
    expected = expected_entries[0]
    if not isinstance(expected, dict) or any(
        expected.get(key) != value for key, value in configured_identity.items()
    ):
        raise RuntimeError(
            "SAM2 HOI CMake selected a different cuBLASLt DSO than the build identity"
        )


def _verify_configured_cuda_compiler(build_dir: Path, identity: dict[str, object]) -> None:
    """Fail closed unless CMake selected the identity-pinned nvcc executable."""

    cache = build_dir / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"SAM2 HOI cannot read configured CMake cache {cache}") from error
    prefix = "CMAKE_CUDA_COMPILER:FILEPATH="
    values = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
    if len(values) != 1 or not values[0] or values[0].endswith("-NOTFOUND"):
        raise RuntimeError("SAM2 HOI CMake cache has no unique CMAKE_CUDA_COMPILER")
    configured = Path(values[0]).resolve(strict=True)
    recorded = identity.get("cuda")
    if not isinstance(recorded, dict) or recorded.get("path") != str(configured):
        raise RuntimeError(
            "SAM2 HOI CMake selected a different CUDA compiler than the build identity"
        )


def _verify_configured_cxx_compiler(build_dir: Path, identity: dict[str, object]) -> None:
    """Fail closed unless CMake selected the identity-recorded C++ compiler."""

    cache = build_dir / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"SAM2 HOI cannot read configured CMake cache {cache}") from error
    prefix = "CMAKE_CXX_COMPILER:FILEPATH="
    values = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
    if len(values) != 1 or not values[0] or values[0].endswith("-NOTFOUND"):
        raise RuntimeError("SAM2 HOI CMake cache has no unique CMAKE_CXX_COMPILER")
    configured = Path(values[0]).resolve(strict=True)
    recorded = identity.get("compiler")
    if not isinstance(recorded, dict) or recorded.get("path") != str(configured):
        raise RuntimeError(
            "SAM2 HOI CMake selected a different C++ compiler than the build identity"
        )


def _verify_configured_tensorrt(build_dir: Path, identity: dict[str, object]) -> None:
    """Fail closed unless CMake selected the identity-recorded TensorRT install."""

    cache = build_dir / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"SAM2 HOI cannot read configured CMake cache {cache}") from error

    def cache_path(variable: str, value_type: str) -> Path:
        prefix = f"{variable}:{value_type}="
        values = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
        if len(values) != 1 or not values[0] or values[0].endswith("-NOTFOUND"):
            raise RuntimeError(f"SAM2 HOI CMake cache has no unique {variable}")
        return Path(values[0]).resolve(strict=True)

    configured_include = cache_path("SAM2_HOI_TRT_INCLUDE_DIR", "PATH")
    configured_library = cache_path("SAM2_HOI_TRT_LIBRARY", "FILEPATH")
    recorded = identity.get("tensorrt")
    if not isinstance(recorded, dict):
        raise RuntimeError("SAM2 HOI build identity has no TensorRT entry")
    header = recorded.get("header")
    runtime_header = recorded.get("runtime_header")
    if not isinstance(header, dict) or not isinstance(runtime_header, dict):
        raise RuntimeError("SAM2 HOI build identity has incomplete TensorRT headers")
    recorded_header = Path(str(header.get("path"))).resolve(strict=True)
    recorded_runtime_header = Path(str(runtime_header.get("path"))).resolve(strict=True)
    if (
        recorded_header.parent != configured_include
        or recorded_runtime_header.parent != configured_include
        or _file_sha256(recorded_header) != header.get("sha256")
        or _file_sha256(recorded_runtime_header) != runtime_header.get("sha256")
    ):
        raise RuntimeError(
            "SAM2 HOI CMake selected different TensorRT headers than the build identity"
        )

    configured_library_identity = {
        "path": str(configured_library),
        "size": configured_library.stat().st_size,
        "sha256": _file_sha256(configured_library),
    }
    libraries = recorded.get("libraries")
    if not isinstance(libraries, list) or not any(
        isinstance(library, dict)
        and all(library.get(key) == value for key, value in configured_library_identity.items())
        for library in libraries
    ):
        raise RuntimeError(
            "SAM2 HOI CMake selected a different TensorRT DSO than the build identity"
        )


def _build_identity(source_dir: Path) -> dict[str, object]:
    compiler = os.environ.get("CXX") or "c++"
    nvcc = os.environ.get("CUDACXX") or "nvcc"
    cuda_toolkit_root = _cuda_toolkit_root(nvcc)
    return {
        "schema_version": 3,
        "source_digest": _source_digest(source_dir),
        "cuda_architectures": list(_CUDA_ARCHITECTURES),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "cmake": _command_identity("cmake", "--version"),
        "compiler": _command_identity(compiler, "--version"),
        "cuda": _command_identity(nvcc, "--version"),
        "cuda_toolkit_root": str(cuda_toolkit_root),
        "cublaslt": _cublaslt_identity(cuda_toolkit_root),
        "tensorrt": _tensorrt_identity(),
    }


def _identity_digest(identity: dict[str, object]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_receipt_identity(plugin: Path) -> dict[str, object]:
    """Return the authenticated build identity associated with ``plugin``."""

    plugin = plugin.resolve(strict=True)
    receipt = plugin.parent / _RECEIPT_NAME
    if not _private_cache_file_exists(plugin) or not _private_cache_file_exists(receipt):
        raise RuntimeError(
            "SAM2 HOI native plugin runtime closure requires a private build receipt"
        )
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"SAM2 HOI cannot read native plugin receipt {receipt}") from error
    if not isinstance(recorded, dict) or recorded.get("output_sha256") != _file_sha256(plugin):
        raise RuntimeError("SAM2 HOI native plugin does not match its build receipt")
    identity = recorded.get("identity")
    if not isinstance(identity, dict) or identity.get("schema_version") != 3:
        raise RuntimeError("SAM2 HOI native plugin has no schema-3 build identity")
    return identity


def _expected_runtime_cublaslt(plugin: Path) -> dict[str, object]:
    """Return and revalidate the single recorded cuBLASLt runtime identity."""

    entries = _runtime_receipt_identity(plugin).get("cublaslt")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise RuntimeError("SAM2 HOI native plugin receipt has no unique cuBLASLt identity")
    expected = entries[0]
    expected_path_value = expected.get("path")
    if not isinstance(expected_path_value, str):
        raise RuntimeError("SAM2 HOI native plugin cuBLASLt identity has no path")
    try:
        expected_path = Path(expected_path_value).resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"SAM2 HOI recorded cuBLASLt DSO is unavailable: {expected_path_value}"
        ) from error
    current = {
        "name": expected_path.name,
        "path": str(expected_path),
        "size": expected_path.stat().st_size,
        "sha256": _file_sha256(expected_path),
    }
    if any(expected.get(key) != value for key, value in current.items()):
        raise RuntimeError("SAM2 HOI recorded cuBLASLt DSO identity has changed")
    return current


def _loaded_cublaslt_paths(maps_path: Path = _PROCESS_MAPS) -> list[Path]:
    """Resolve every cuBLASLt DSO actually mapped into this Linux process."""

    try:
        lines = maps_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(
            f"SAM2 HOI cannot inspect process library mappings {maps_path}"
        ) from error
    loaded: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        mapped = fields[5]
        if mapped.endswith(" (deleted)"):
            candidate = Path(mapped.removesuffix(" (deleted)"))
            if candidate.name == "libcublasLt.so" or candidate.name.startswith("libcublasLt.so."):
                raise RuntimeError("SAM2 HOI process has a deleted cuBLASLt DSO mapping")
            continue
        candidate = Path(mapped)
        if candidate.name != "libcublasLt.so" and not candidate.name.startswith("libcublasLt.so."):
            continue
        try:
            loaded.add(candidate.resolve(strict=True))
        except OSError as error:
            raise RuntimeError(
                f"SAM2 HOI cannot resolve loaded cuBLASLt DSO {candidate}"
            ) from error
    return sorted(loaded)


def _verify_loaded_cublaslt(
    plugin: Path,
    *,
    allow_unloaded: bool = False,
    maps_path: Path = _PROCESS_MAPS,
) -> Path | None:
    """Fail closed unless the process maps the receipt-pinned cuBLASLt DSO."""

    expected = _expected_runtime_cublaslt(plugin)
    loaded = _loaded_cublaslt_paths(maps_path)
    if not loaded and allow_unloaded:
        return None
    if len(loaded) != 1:
        raise RuntimeError("SAM2 HOI process must map exactly one receipt-pinned cuBLASLt DSO")
    actual_path = loaded[0]
    actual = {
        "name": actual_path.name,
        "path": str(actual_path),
        "size": actual_path.stat().st_size,
        "sha256": _file_sha256(actual_path),
    }
    if actual != expected:
        raise RuntimeError(
            "SAM2 HOI process loaded a different cuBLASLt DSO than the build receipt"
        )
    return actual_path


def _private_cache_file_exists(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise _unsafe_cache_path(path, f"cannot inspect cache file: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise _unsafe_cache_path(path, "symbolic links are not allowed")
    if not stat.S_ISREG(metadata.st_mode):
        raise _unsafe_cache_path(path, "cache entry is not a regular file")
    effective_uid = os.geteuid()
    if metadata.st_uid != effective_uid:
        raise _unsafe_cache_path(
            path,
            f"owner uid {metadata.st_uid} does not match effective uid {effective_uid}",
        )
    if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise _unsafe_cache_path(path, "cache file is group/world writable")
    return True


def _harden_built_output(path: Path) -> bool:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise _unsafe_cache_path(path, "platform does not support O_NOFOLLOW")
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise _unsafe_cache_path(path, f"cannot open build output: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _unsafe_cache_path(path, "build output is not a regular file")
        if metadata.st_uid != os.geteuid():
            raise _unsafe_cache_path(
                path,
                "build output owner does not match the effective user",
            )
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)
    return True


def _cached_output_matches(output: Path, receipt: Path, identity: dict[str, object]) -> bool:
    output_exists = _private_cache_file_exists(output)
    receipt_exists = _private_cache_file_exists(receipt)
    if not output_exists or not receipt_exists:
        return False
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        output_digest = _file_sha256(output)
    except (OSError, ValueError, TypeError):
        return False
    return recorded == {"identity": identity, "output_sha256": output_digest}


def _write_build_receipt(output: Path, receipt: Path, identity: dict[str, object]) -> None:
    payload = {
        "identity": identity,
        "output_sha256": _file_sha256(output),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{receipt.name}.",
        suffix=".tmp",
        dir=receipt.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_build_lock(build_base: Path, source_digest: str) -> Iterator[None]:
    build_base = _secure_private_directory(build_base)
    lock_path = build_base / f".{source_digest}.lock"
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise _unsafe_cache_path(lock_path, "platform does not support O_NOFOLLOW")
    flags = os.O_CREAT | os.O_RDWR | no_follow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise _unsafe_cache_path(lock_path, f"cannot open build lock: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise _unsafe_cache_path(lock_path, "build lock is not a regular file")
        if metadata.st_uid != os.geteuid():
            raise _unsafe_cache_path(
                lock_path,
                f"owner uid {metadata.st_uid} does not match effective uid {os.geteuid()}",
            )
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise _unsafe_cache_path(lock_path, "build lock is group/world writable")
        lock_file = os.fdopen(descriptor, "a+b")
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    with lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    override = os.environ.get(_PLUGIN_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{_PLUGIN_ENV} does not exist: {path}")
        return path

    source_dir = Path(__file__).with_name("native_plugins")
    build_base = _secure_private_directory(_configured_build_base())
    identity = _build_identity(source_dir)
    identity_digest = _identity_digest(identity)
    build_dir = _secure_private_directory(build_base / identity_digest)
    output = build_dir / "libtrtmc_sam2_hoi_native_plugin.so"
    receipt = build_dir / _RECEIPT_NAME
    if _cached_output_matches(output, receipt, identity):
        return output

    with _exclusive_build_lock(build_base, identity_digest):
        build_dir = _secure_private_directory(build_dir)
        if _cached_output_matches(output, receipt, identity):
            return output
        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCUDAToolkit_ROOT={identity['cuda_toolkit_root']}",
        ]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "trtmc_sam2_hoi_native_plugin",
            "-j2",
        ]
        kwargs = (
            {}
            if verbose
            else {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
        )
        try:
            subprocess.run(configure, check=True, **kwargs)
            _verify_configured_cxx_compiler(build_dir, identity)
            _verify_configured_cuda_compiler(build_dir, identity)
            _verify_configured_cublaslt(build_dir, identity)
            _verify_configured_tensorrt(build_dir, identity)
            subprocess.run(build, check=True, **kwargs)
        except subprocess.CalledProcessError as error:
            output_text = getattr(error, "stdout", "") or ""
            raise RuntimeError(f"SAM2 HOI native plugin build failed\n{output_text}") from error
        if not _harden_built_output(output) or not _private_cache_file_exists(output):
            raise RuntimeError(f"SAM2 HOI native plugin build did not produce {output}")
        _write_build_receipt(output, receipt, identity)
    return output
