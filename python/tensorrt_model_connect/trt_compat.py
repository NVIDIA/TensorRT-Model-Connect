# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT Python compatibility boundary.

All builder-side TensorRT imports and version-sensitive Python API calls flow
through this module.  The rest of tensorrt_model_connect binds ``get_trt()`` from here
instead of importing ``tensorrt`` directly, which gives us one place to adapt
TRT API drift while keeping the active Python environment as the source of
truth.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import importlib
import importlib.util
import os
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any


_STANDARD_MODULE = "tensorrt"
_RTX_MODULE = "tensorrt_rtx"
_backend_module_name = _STANDARD_MODULE
_backend_label = "TensorRT"
_module: ModuleType | None = None
_internal_library_path_state: tuple[int, str, str, str] | None = None
_internal_library_handle: Any | None = None
_internal_library_path_lock = threading.Lock()

_LIBNVINFER_NAME_RE = re.compile(r"^libnvinfer\.so(?:\.(\d+)(?:\.(\d+))?(?:\.\d+)*)?$")
_SHARED_LIBRARY_VERSION_RE = re.compile(r"\.so\.(\d+)\.(\d+)\.(\d+)$")

_TIMING_CACHE_PATH_ENV = "TRTMC_TRT_TIMING_CACHE_PATH"
_TIMING_CACHE_DIR_ENV = "TRTMC_TRT_TIMING_CACHE_DIR"
_BUILDER_OPT_LEVEL_ENV = "TRTMC_BUILDER_OPTIMIZATION_LEVEL"
_MAX_NUM_TACTICS_ENV = "TRTMC_MAX_NUM_TACTICS"
_AVG_TIMING_ITERATIONS_ENV = "TRTMC_AVG_TIMING_ITERATIONS"


def configure_backend(*, rtx: bool = False) -> None:
    """Select the TensorRT Python module before any TRT API is touched."""
    global _backend_module_name, _backend_label, _module

    requested = _RTX_MODULE if rtx else _STANDARD_MODULE
    requested_label = "TensorRT-RTX" if rtx else "TensorRT"
    if _module is not None and _backend_module_name != requested:
        raise RuntimeError(
            f"{_backend_label} is already loaded; cannot switch to "
            f"{requested_label} in the same Python process"
        )

    if rtx:
        try:
            rtx_module = importlib.import_module(_RTX_MODULE)
        except ImportError as exc:
            raise ImportError(
                "TensorRT-RTX is required for --rtx builds. "
                "Install it with: pip install tensorrt_rtx"
            ) from exc
        # Some builder code and third-party helpers still import the standard
        # module name.  Keep that alias process-local and configured here.
        sys.modules[_STANDARD_MODULE] = rtx_module

    _backend_module_name = requested
    _backend_label = requested_label


def is_available(module_name: str | None = None) -> bool:
    """Return whether the selected TensorRT Python module can be imported."""
    name = module_name or _backend_module_name
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def load_module() -> ModuleType:
    """Import and return the selected TensorRT Python module."""
    global _module
    if _backend_module_name in sys.modules and sys.modules[_backend_module_name] is None:
        _module = None
        raise ImportError(f"{_backend_module_name} is not available")
    active = sys.modules.get(_backend_module_name)
    if active is not None and active is not _module:
        _module = active
    if _module is None:
        _module = importlib.import_module(_backend_module_name)
    return _module


def module_version(module_name: str | None = None) -> str:
    """Return a TensorRT module's Python package version, or an empty string."""
    try:
        module = (
            load_module()
            if module_name is None
            else importlib.import_module(module_name)
        )
    except (ImportError, AttributeError):
        return ""
    return str(getattr(module, "__version__", ""))


def tensorrt_version() -> str:
    return module_version()


def tensorrt_abi(version: str | None = None) -> str:
    match = re.search(r"(\d+)\.(\d+)", version or tensorrt_version() or "")
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def module_file(module_name: str | None = None) -> str:
    """Return the resolved Python module path for diagnostics."""
    try:
        module = (
            load_module()
            if module_name is None
            else importlib.import_module(module_name)
        )
    except (ImportError, AttributeError):
        return ""
    return str(getattr(module, "__file__", "") or "")


def loaded_libnvinfer_paths() -> list[str]:
    """Best-effort list of libnvinfer DSOs already mapped in this process."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths: list[str] = []
    try:
        for line in maps.read_text(errors="ignore").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6:
                continue
            path = fields[5].removesuffix(" (deleted)")
            if _LIBNVINFER_NAME_RE.fullmatch(Path(path).name) and path not in paths:
                paths.append(path)
    except OSError:
        return []
    return paths


def _version_triplet(version: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _native_library_version(handle: Any) -> tuple[int, int, int] | None:
    try:
        major_fn = handle.getInferLibMajorVersion
        minor_fn = handle.getInferLibMinorVersion
        patch_fn = handle.getInferLibPatchVersion
    except AttributeError:
        return None
    major_fn.argtypes = []
    major_fn.restype = ctypes.c_int
    minor_fn.argtypes = []
    minor_fn.restype = ctypes.c_int
    patch_fn.argtypes = []
    patch_fn.restype = ctypes.c_int
    return int(major_fn()), int(minor_fn()), int(patch_fn())


def _resolved_unique_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value).expanduser().resolve(strict=False)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _select_loaded_libnvinfer(
    expected_version: tuple[int, int, int],
) -> tuple[Path, Any] | None:
    candidates = _resolved_unique_paths(loaded_libnvinfer_paths())
    if not candidates:
        # Fake TRT modules and older/system installations may not expose a
        # mapped libnvinfer path. Preserve their existing loader behavior.
        return None

    matching: list[tuple[Path, Any]] = []
    mismatched: list[tuple[Path, tuple[int, int, int]]] = []
    unknown: list[tuple[Path, Any]] = []
    load_errors: list[tuple[Path, str]] = []
    for path in candidates:
        try:
            handle = ctypes.CDLL(str(path))
        except OSError as exc:
            load_errors.append((path, str(exc)))
            continue
        native_version = _native_library_version(handle)
        if native_version is None:
            unknown.append((path, handle))
        elif native_version == expected_version:
            matching.append((path, handle))
        else:
            mismatched.append((path, native_version))

    if load_errors:
        rendered = ", ".join(f"{path}: {error}" for path, error in load_errors)
        raise RuntimeError(
            "Unable to inspect every mapped libnvinfer before configuring TensorRT "
            f"builder libraries: {rendered}"
        )
    modern_unknown = [path for path, handle in unknown if hasattr(handle, "setInternalLibraryPath")]
    if modern_unknown:
        rendered = ", ".join(str(path) for path in modern_unknown)
        raise RuntimeError(
            "Unable to determine the exact version of mapped libnvinfer DSOs that "
            f"expose setInternalLibraryPath: {rendered}"
        )
    if matching and unknown:
        rendered = ", ".join(str(path) for path, _ in unknown)
        raise RuntimeError(
            "A TensorRT version-matched libnvinfer is loaded alongside DSOs whose version "
            f"cannot be verified: {rendered}"
        )
    if mismatched:
        expected = ".".join(str(value) for value in expected_version)
        rendered = ", ".join(
            f"{path} (version {version[0]}.{version[1]}.{version[2]})"
            for path, version in mismatched
        )
        if matching:
            raise RuntimeError(
                "Multiple TensorRT versions are loaded in one builder process: "
                f"Python version {expected}, incompatible libnvinfer DSOs: {rendered}"
            )
        raise RuntimeError(
            f"TensorRT Python version {expected} does not match loaded libnvinfer: {rendered}"
        )

    matching_dirs = {path.parent for path, _ in matching}
    if len(matching_dirs) > 1:
        rendered = ", ".join(str(path) for path, _ in matching)
        expected = ".".join(str(value) for value in expected_version)
        raise RuntimeError(
            "TensorRT internal builder library path is ambiguous: Python version "
            f"{expected} has matching libnvinfer DSOs in multiple directories: {rendered}"
        )
    if matching:
        return matching[0]

    # A library without version-query exports predates this compatibility
    # contract. Leave it on its original loader path instead of guessing.
    assert unknown
    return None


def _builder_resource_matches_version(path: Path, expected_version: tuple[int, int, int]) -> bool:
    if not path.name.startswith("libnvinfer_builder_resource"):
        return False
    match = _SHARED_LIBRARY_VERSION_RE.search(path.name)
    if match is None:
        return False
    version = tuple(int(match.group(index)) for index in (1, 2, 3))
    if version != expected_version:
        return False
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return resolved.is_file()


def _require_internal_library_directory(
    libnvinfer_path: Path,
    expected_version: tuple[int, int, int],
) -> Path:
    directory = libnvinfer_path.parent
    resources = sorted(directory.glob("libnvinfer_builder_resource*.so*"))
    if any(_builder_resource_matches_version(path, expected_version) for path in resources):
        return directory
    expected = ".".join(str(value) for value in expected_version)
    suffix = ""
    if resources:
        suffix = "; incompatible resources: " + ", ".join(str(path) for path in resources)
    raise RuntimeError(
        "TensorRT internal builder resources for version "
        f"{expected} were not found beside loaded {libnvinfer_path}{suffix}"
    )


def _configure_standard_internal_library_path(module: ModuleType) -> None:
    if _backend_module_name != _STANDARD_MODULE:
        return
    # Unit-test doubles and legacy pure-Python shims do not own a native module.
    if not getattr(module, "__file__", None):
        return
    expected_version = _version_triplet(str(getattr(module, "__version__", "")))
    if expected_version is None:
        raise RuntimeError(
            "Cannot configure TensorRT internal builder libraries because "
            f"{_STANDARD_MODULE}.__version__ is missing or invalid"
        )

    # setInternalLibraryPath changes process-global TensorRT state. Serialize
    # both the map validation and the first setter call so concurrent Builder
    # creation cannot race the set-once guard.
    with _internal_library_path_lock:
        _configure_standard_internal_library_path_locked(module, expected_version)


def _configure_standard_internal_library_path_locked(
    module: ModuleType,
    expected_version: tuple[int, int, int],
) -> None:
    global _internal_library_handle, _internal_library_path_state

    expected = ".".join(str(value) for value in expected_version)
    selected = _select_loaded_libnvinfer(expected_version)
    if selected is None:
        return
    libnvinfer_path, handle = selected
    try:
        configure = handle.setInternalLibraryPath
    except AttributeError:
        # Older TensorRT libraries do not expose the supported internal-path
        # hook. Preserve their prior loader behavior.
        return
    library_dir = _require_internal_library_directory(libnvinfer_path, expected_version)
    state = (id(module), expected, str(libnvinfer_path), str(library_dir))
    if _internal_library_path_state == state:
        return
    configure.argtypes = [ctypes.c_char_p]
    configure.restype = ctypes.c_bool
    try:
        configured = bool(configure(os.fsencode(library_dir)))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "TensorRT failed to configure its internal builder library path "
            f"to {library_dir}: {exc}"
        ) from exc
    if not configured:
        raise RuntimeError(f"TensorRT rejected its internal builder library path: {library_dir}")

    _internal_library_handle = handle
    _internal_library_path_state = state


def resolved_summary() -> str:
    version = tensorrt_version() or "unknown"
    abi = tensorrt_abi(version) or "unknown"
    module_path = module_file() or "unknown"
    libs = loaded_libnvinfer_paths()
    lib_text = libs[0] if libs else "not loaded yet"
    return (
        f"{_backend_label}: version={version}, abi={abi}, "
        f"python={module_path}, native={lib_text}"
    )


def network_creation_flags(
    *,
    strongly_typed: bool = True,
    explicit_batch: bool = False,
) -> int:
    """Build network creation flags across TRT versions.

    TRT 10+ uses explicit batch by default and TRT 11 Python bindings may not
    expose EXPLICIT_BATCH.  When the flag does not exist, requesting it is a
    no-op instead of a build-time AttributeError.
    """
    flags = 0
    creation_flag = getattr(load_module(), "NetworkDefinitionCreationFlag", None)
    if creation_flag is None:
        return flags
    if strongly_typed and hasattr(creation_flag, "STRONGLY_TYPED"):
        flags |= 1 << int(creation_flag.STRONGLY_TYPED)
    if explicit_batch and hasattr(creation_flag, "EXPLICIT_BATCH"):
        flags |= 1 << int(creation_flag.EXPLICIT_BATCH)
    return flags


def get_trt() -> "TensorRTModuleProxy":
    """Return a proxy bound to the currently selected/imported TRT module."""
    return TensorRTModuleProxy(load_module())


def add_matrix_multiply(
    network: Any,
    lhs: Any,
    lhs_op: Any,
    rhs: Any,
    rhs_op: Any,
) -> Any:
    """Add a matrix multiply layer through the TRT-version compatibility hook."""
    raw_network = unwrap(network)
    return raw_network.add_matrix_multiply(lhs, lhs_op, rhs, rhs_op)


def unwrap(value: Any) -> Any:
    if isinstance(value, _HandleProxy):
        return value._raw
    if isinstance(value, tuple):
        return tuple(unwrap(item) for item in value)
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    if isinstance(value, dict):
        return {key: unwrap(item) for key, item in value.items()}
    return value


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _set_optional_builder_int(config: Any, attr: str, env_name: str) -> None:
    value = _optional_int_env(env_name)
    if value is not None and hasattr(config, attr):
        setattr(config, attr, value)


def _apply_builder_config_env(config: Any) -> None:
    _set_optional_builder_int(
        config, "builder_optimization_level", _BUILDER_OPT_LEVEL_ENV)
    _set_optional_builder_int(config, "max_num_tactics", _MAX_NUM_TACTICS_ENV)
    _set_optional_builder_int(
        config, "avg_timing_iterations", _AVG_TIMING_ITERATIONS_ENV)


def _sanitize_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "tensorrt"


def _timing_cache_path() -> Path | None:
    explicit = os.environ.get(_TIMING_CACHE_PATH_ENV)
    if explicit and explicit.strip():
        return Path(explicit)

    cache_dir = os.environ.get(_TIMING_CACHE_DIR_ENV)
    if not cache_dir or not cache_dir.strip():
        return None

    version = _sanitize_cache_name(tensorrt_version() or "unknown")
    opt_level = os.environ.get(_BUILDER_OPT_LEVEL_ENV, "default")
    opt_level = _sanitize_cache_name(opt_level)
    return Path(cache_dir) / f"{_backend_module_name}-{version}-opt{opt_level}.cache"


def _scope_cache_path(path: Path, scope: str) -> Path:
    scoped_name = _sanitize_cache_name(scope)
    return path.with_name(f"{path.stem}.{scoped_name}{path.suffix}")


@contextmanager
def scoped_timing_cache(scope: str | None):
    """Temporarily route TensorRT timing-cache IO to a scoped cache file."""
    if not scope:
        yield
        return

    path = _timing_cache_path()
    if path is None:
        yield
        return

    previous = os.environ.get(_TIMING_CACHE_PATH_ENV)
    os.environ[_TIMING_CACHE_PATH_ENV] = str(_scope_cache_path(path, scope))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_TIMING_CACHE_PATH_ENV, None)
        else:
            os.environ[_TIMING_CACHE_PATH_ENV] = previous


@contextmanager
def _locked_cache(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        locked = False
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            locked = True
        except (ImportError, OSError):
            pass
        try:
            yield
        finally:
            if locked:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


class _TimingCacheState:
    def __init__(self, path: Path, cache: Any):
        self.path = path
        self.cache = cache


def _attach_timing_cache(config: Any) -> _TimingCacheState | None:
    path = _timing_cache_path()
    if path is None:
        return None
    if not hasattr(config, "create_timing_cache") or not hasattr(config, "set_timing_cache"):
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = path.read_bytes() if path.is_file() else b""
        cache = config.create_timing_cache(payload)
        if config.set_timing_cache(cache, True):
            return _TimingCacheState(path, cache)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return None


def _serialize_cache(cache: Any) -> bytes | None:
    if cache is None or not hasattr(cache, "serialize"):
        return None
    payload = cache.serialize()
    if payload is None:
        return None
    return bytes(payload)


def _save_timing_cache(config: Any, state: _TimingCacheState | None) -> None:
    if state is None:
        return
    try:
        cache = config.get_timing_cache() if hasattr(config, "get_timing_cache") else state.cache
        if cache is None:
            cache = state.cache
        state.path.parent.mkdir(parents=True, exist_ok=True)
        with _locked_cache(state.path):
            if state.path.is_file() and hasattr(config, "create_timing_cache"):
                current_payload = state.path.read_bytes()
                if current_payload and hasattr(cache, "combine"):
                    current_cache = config.create_timing_cache(current_payload)
                    cache.combine(current_cache, True)
            payload = _serialize_cache(cache)
            if payload is None:
                return
            tmp_path = state.path.with_name(
                f".{state.path.name}.{os.getpid()}.tmp")
            tmp_path.write_bytes(payload)
            os.replace(tmp_path, state.path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return


def _wrap_result(value: Any, module: ModuleType | Any | None = None) -> Any:
    module = module or load_module()
    try:
        if isinstance(value, module.IBuilder):
            return _BuilderProxy(value, module)
        if isinstance(value, module.INetworkDefinition):
            return _NetworkProxy(value, module)
        if isinstance(value, module.IBuilderConfig):
            return _BuilderConfigProxy(value, module)
        if isinstance(value, module.IRuntime):
            return _RuntimeProxy(value, module)
    except AttributeError:
        pass
    return value


class _HandleProxy:
    def __init__(self, raw: Any, module: ModuleType | Any):
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_trt_module", module)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._raw, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            return _wrap_result(
                attr(
                    *[unwrap(arg) for arg in args],
                    **{key: unwrap(value) for key, value in kwargs.items()},
                ),
                self._trt_module,
            )

        return call

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._raw, name, unwrap(value))

    def __repr__(self) -> str:
        return repr(self._raw)


class _BuilderProxy(_HandleProxy):
    def create_network(self, flags: int = 0) -> Any:
        return _NetworkProxy(self._raw.create_network(flags), self._trt_module)

    def create_builder_config(self) -> Any:
        return _BuilderConfigProxy(
            self._raw.create_builder_config(),
            self._trt_module,
        )

    def build_serialized_network(self, network: Any, config: Any) -> Any:
        raw_config = unwrap(config)
        _apply_builder_config_env(raw_config)
        timing_cache = _attach_timing_cache(raw_config)
        try:
            return _wrap_result(
                self._raw.build_serialized_network(unwrap(network), raw_config),
                self._trt_module,
            )
        finally:
            _save_timing_cache(raw_config, timing_cache)


class _NetworkProxy(_HandleProxy):
    def add_matrix_multiply(
        self,
        lhs: Any,
        lhs_op: Any,
        rhs: Any,
        rhs_op: Any,
    ) -> Any:
        return add_matrix_multiply(self._raw, lhs, lhs_op, rhs, rhs_op)


class _BuilderConfigProxy(_HandleProxy):
    pass


class _RuntimeProxy(_HandleProxy):
    pass


class _LoggerFactory:
    def __init__(self, module_getter):
        self._module_getter = module_getter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module_getter().Logger(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module_getter().Logger, name)


class _BuilderFactory:
    def __init__(self, module_getter):
        self._module_getter = module_getter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        module = self._module_getter()
        _configure_standard_internal_library_path(module)
        return _BuilderProxy(module.Builder(*args, **kwargs), module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module_getter().Builder, name)


class _RuntimeFactory:
    def __init__(self, module_getter):
        self._module_getter = module_getter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        module = self._module_getter()
        return _RuntimeProxy(module.Runtime(*args, **kwargs), module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module_getter().Runtime, name)


class _OnnxParserFactory:
    def __init__(self, module_getter):
        self._module_getter = module_getter

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module_getter().OnnxParser(
            *[unwrap(arg) for arg in args],
            **{key: unwrap(value) for key, value in kwargs.items()},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module_getter().OnnxParser, name)


class TensorRTModuleProxy:
    """Lazy module proxy that exposes TensorRT through compatibility wrappers."""

    def __init__(self, module: ModuleType | Any | None = None):
        self._fixed_module = module

    def _module(self) -> ModuleType | Any:
        return self._fixed_module or load_module()

    def __getattr__(self, name: str) -> Any:
        if name == "Logger":
            return _LoggerFactory(self._module)
        if name == "Builder":
            return _BuilderFactory(self._module)
        if name == "Runtime":
            return _RuntimeFactory(self._module)
        if name == "OnnxParser":
            return _OnnxParserFactory(self._module)
        return getattr(self._module(), name)

    def __repr__(self) -> str:
        return f"<TensorRTModuleProxy backend={_backend_module_name!r}>"


trt = TensorRTModuleProxy()
