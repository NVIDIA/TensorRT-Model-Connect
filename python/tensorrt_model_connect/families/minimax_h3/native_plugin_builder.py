# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, load, and add the MiniMax-H3 model-owned TensorRT plugin."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


_PLUGIN_ENV = "TRTMC_MINIMAX_H3_NATIVE_PLUGIN_LIBRARY"
_BUILD_DIR_ENV = "TRTMC_MINIMAX_H3_NATIVE_PLUGIN_BUILD_DIR"
_VISION_PLUGIN_NAME = "MiniMaxH3VisionAttention"
_AUDIO_ENCODER_PLUGIN_NAME = "MiniMaxH3AudioEncoder"
_LAYER_NORM_PLUGIN_NAME = "MiniMaxH3LayerNorm"
_LINEAR_PLUGIN_NAME = "MiniMaxH3Linear"
_PATCH_EMBED_PLUGIN_NAME = "MiniMaxH3PatchEmbed"
_PLUGIN_NAME = _VISION_PLUGIN_NAME
_PLUGIN_VERSION = "1"
_PLUGIN_IDENTITY = "trtmc.minimax_h3.native_plugin:aten-ops:1"
_PLUGIN_ABI_VERSION = 1
_PLUGIN_IDENTITY_SYMBOL = "trtmc_minimax_h3_native_plugin_identity"
_PLUGIN_ABI_SYMBOL = "trtmc_minimax_h3_native_plugin_abi_version"
_PLUGIN_BUILD_IDENTITY_SYMBOL = "trtmc_minimax_h3_native_plugin_build_identity"
_PLUGIN_REGISTRY_SYMBOL = "trtmc_minimax_h3_native_plugin_registry_matches"
_PLUGIN_HANDLES: dict[Path, Any] = {}
_FAILED_PLUGIN_HANDLES: dict[Path, Any] = {}
_AUDIO_ENCODER_MODULE_KEEPALIVE: dict[int, np.ndarray] = {}
_AUDIO_ENCODER_MODULE_MIN_BYTES = 300 << 20
_AUDIO_ENCODER_MODULE_MAX_BYTES = 400 << 20


def native_plugin_source_files() -> tuple[Path, ...]:
    """Return every source file that defines the model-owned plugin DSO."""

    source_dir = Path(__file__).with_name("native_plugins")
    if not source_dir.is_dir():
        source_dir = (
            Path(__file__).resolve().parents[4] / "src/runtime/models/minimax_h3/native_plugins"
        )
    paths = tuple(
        sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix in {".cpp", ".cu", ".h", ".txt"}
        )
    )
    if not paths or not (source_dir / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"MiniMax-H3 native plugin sources are missing: {source_dir}")
    return paths


def _source_digest(source_files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_files):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _normalized_tensorrt_version(raw_version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)\.(\d+)", raw_version)
    if match is None:
        raise RuntimeError(
            "Cannot validate TensorRT C++ headers: active TensorRT version "
            f"{raw_version!r} is not major.minor.patch.build"
        )
    return ".".join(match.groups())


def _active_tensorrt_cmake_hints() -> tuple[str, ...]:
    """Pin CMake to the exact TensorRT ABI already active in this process."""

    from tensorrt_model_connect import trt_compat

    runtime_version = _normalized_tensorrt_version(trt_compat.tensorrt_version())
    hints: list[str] = [f"-DMINIMAX_H3_TRT_EXPECTED_VERSION={runtime_version}"]
    libraries = list(
        dict.fromkeys(
            Path(candidate).resolve()
            for candidate in trt_compat.loaded_libnvinfer_paths()
            if Path(candidate).is_file()
        )
    )
    if libraries:
        active_major = runtime_version.split(".", 1)[0]
        identities = {
            (
                library.parent,
                (
                    match.group(1)
                    if (match := re.search(r"\.so\.(\d+)", library.name))
                    else active_major
                ),
            )
            for library in libraries
        }
        if len(identities) > 1:
            raise RuntimeError(f"Multiple TensorRT runtimes are loaded: {libraries}")
        library = max(libraries, key=lambda candidate: len(candidate.name))
        hints.append(f"-DMINIMAX_H3_TRT_LIBRARY={library}")

    for variable in ("TRTMC_TRT_INCLUDE_DIR", "TRT_INC_DIR"):
        candidate = os.environ.get(variable)
        if candidate and (Path(candidate) / "NvInferRuntime.h").is_file():
            hints.append(f"-DMINIMAX_H3_TRT_INCLUDE_DIR={Path(candidate).resolve()}")
            break
    return tuple(hints)


def _cuda_toolkit_version() -> str:
    """Return the exact nvcc toolkit release selected by CMake."""

    configured = os.environ.get("CUDACXX")
    compiler = configured or shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("MiniMax-H3 native plugin build requires nvcc on PATH or CUDACXX")
    result = subprocess.run(
        [compiler, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", result.stdout)
    if match is None:
        raise RuntimeError(f"Cannot determine exact CUDA toolkit version from {compiler}")
    return match.group(1)


def _cxx_compiler_identity() -> tuple[str, str, str]:
    """Return resolved compiler path, numeric version, and complete first-line identity."""

    configured = os.environ.get("CXX")
    compiler = configured or shutil.which("c++")
    if not compiler:
        raise RuntimeError("MiniMax-H3 native plugin build requires CXX or c++ on PATH")
    compiler_path = str(Path(compiler).resolve())
    version_result = subprocess.run(
        [compiler_path, "-dumpfullversion", "-dumpversion"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    version = version_result.stdout.strip()
    identity_result = subprocess.run(
        [compiler_path, "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    identity = identity_result.stdout.splitlines()[0].strip()
    if not re.fullmatch(r"\d+(?:\.\d+)+", version) or not identity:
        raise RuntimeError(f"Cannot determine exact C++ compiler identity from {compiler_path}")
    return compiler_path, version, identity


def _torch_cxx11_abi(torch_module: Any) -> int:
    probe = getattr(torch_module, "compiled_with_cxx11_abi", None)
    if callable(probe):
        return int(bool(probe()))
    torch_c = getattr(torch_module, "_C", None)
    if torch_c is None or not hasattr(torch_c, "_GLIBCXX_USE_CXX11_ABI"):
        raise RuntimeError("Cannot determine the active Torch C++11 ABI")
    return int(bool(torch_c._GLIBCXX_USE_CXX11_ABI))


def _torch_identity(torch_module: Any) -> dict[str, str]:
    version = str(getattr(torch_module, "__version__", ""))
    cuda_version = str(getattr(getattr(torch_module, "version", None), "cuda", "") or "")
    torch_file = Path(str(getattr(torch_module, "__file__", ""))).resolve()
    cmake_prefix = str(getattr(getattr(torch_module, "utils", None), "cmake_prefix_path", ""))
    if not version or not cuda_version or not torch_file.is_file() or not cmake_prefix:
        raise RuntimeError(
            "MiniMax-H3 native plugin build requires a CUDA Torch package with CMake metadata"
        )
    return {
        "version": version,
        "cuda_version": cuda_version,
        "cxx11_abi": str(_torch_cxx11_abi(torch_module)),
        "root": str(torch_file.parent),
        "cmake_prefix": cmake_prefix,
    }


def _plugin_cache_key(
    source_files: Sequence[Path],
    *,
    tensorrt_version: str,
    torch_version: str,
    torch_cuda_version: str,
    cuda_toolkit_version: str,
    torch_cxx11_abi: str,
    cxx_compiler_identity: str,
    cmake_hints: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    values = (
        "source-layout=minimax-h3-native-plugin-v1",
        f"source={_source_digest(source_files)}",
        f"tensorrt={tensorrt_version}",
        f"torch={torch_version}",
        f"torch-cuda={torch_cuda_version}",
        f"cuda-toolkit={cuda_toolkit_version}",
        f"torch-cxx11-abi={torch_cxx11_abi}",
        f"cxx-compiler={cxx_compiler_identity}",
        *cmake_hints,
    )
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _prepare_private_directory(path: Path, description: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{description} cannot be a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    status = path.stat()
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.geteuid():
        raise RuntimeError(f"{description} is not a private owned directory: {path}")
    path.chmod(0o700)


@contextmanager
def _exclusive_build_lock(build_base: Path, cache_key: str) -> Iterator[None]:
    _prepare_private_directory(build_base, "MiniMax-H3 native plugin build cache")
    lock_path = build_base / f".{cache_key}.lock"
    with open(lock_path, "a+b", opener=lambda path, flags: os.open(path, flags, 0o600)) as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _is_owned_regular_file(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(status.st_mode) and status.st_uid == os.geteuid()


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    """Return the exact dependency-keyed H3 plugin DSO, building it if needed."""

    override = os.environ.get(_PLUGIN_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{_PLUGIN_ENV} does not exist: {path}")
        return path

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-H3 native plugin build requires the CUDA C++ libtorch package"
        ) from exc
    from tensorrt_model_connect import trt_compat

    trt_compat.load_module()
    trt_version = _normalized_tensorrt_version(trt_compat.tensorrt_version())
    torch_identity = _torch_identity(torch)
    cuda_toolkit_version = _cuda_toolkit_version()
    cxx_compiler, cxx_compiler_version, cxx_compiler_identity = _cxx_compiler_identity()
    source_files = native_plugin_source_files()
    source_dir = source_files[0].parent
    cmake_hints = _active_tensorrt_cmake_hints()
    cache_key = _plugin_cache_key(
        source_files,
        tensorrt_version=trt_version,
        torch_version=torch_identity["version"],
        torch_cuda_version=torch_identity["cuda_version"],
        cuda_toolkit_version=cuda_toolkit_version,
        torch_cxx11_abi=torch_identity["cxx11_abi"],
        cxx_compiler_identity=(f"{cxx_compiler}|{cxx_compiler_version}|{cxx_compiler_identity}"),
        cmake_hints=cmake_hints,
    )
    default_build_base = Path(tempfile.gettempdir()) / (
        f"trtmc-minimax-h3-native-plugin-{os.geteuid()}"
    )
    build_base = Path(os.environ.get(_BUILD_DIR_ENV, str(default_build_base))).expanduser()
    build_dir = build_base / cache_key
    output = build_dir / "libtrtmc_minimax_h3_native_plugin.so"
    complete = build_dir / ".complete"

    with _exclusive_build_lock(build_base, cache_key):
        if (
            _is_owned_regular_file(output)
            and _is_owned_regular_file(complete)
            and complete.read_text(encoding="utf-8") == f"{cache_key}\n"
        ):
            return output
        _prepare_private_directory(build_dir, "MiniMax-H3 native plugin build directory")
        if output.exists() or output.is_symlink():
            output.unlink()
        if complete.exists() or complete.is_symlink():
            complete.unlink()

        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={torch_identity['cmake_prefix']}",
            f"-DMINIMAX_H3_TORCH_ROOT={torch_identity['root']}",
            f"-DMINIMAX_H3_TORCH_EXPECTED_VERSION={torch_identity['version']}",
            (f"-DMINIMAX_H3_TORCH_EXPECTED_CUDA_VERSION={torch_identity['cuda_version']}"),
            (f"-DMINIMAX_H3_TORCH_EXPECTED_CXX11_ABI={torch_identity['cxx11_abi']}"),
            f"-DMINIMAX_H3_CUDA_TOOLKIT_EXPECTED_VERSION={cuda_toolkit_version}",
            f"-DMINIMAX_H3_CXX_COMPILER_EXPECTED={cxx_compiler}",
            f"-DMINIMAX_H3_CXX_COMPILER_EXPECTED_VERSION={cxx_compiler_version}",
            f"-DMINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY={cache_key}",
            *cmake_hints,
        ]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "trtmc_minimax_h3_native_plugin",
            "-j2",
        ]
        kwargs = (
            {}
            if verbose
            else {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
        )
        try:
            subprocess.run(configure, check=True, **kwargs)
            subprocess.run(build, check=True, **kwargs)
        except subprocess.CalledProcessError as exc:
            output_text = getattr(exc, "stdout", "") or ""
            raise RuntimeError(f"MiniMax-H3 native plugin build failed\n{output_text}") from exc
        if not _is_owned_regular_file(output):
            raise RuntimeError(f"Native plugin build did not produce a private DSO: {output}")
        complete.write_text(f"{cache_key}\n", encoding="utf-8")
        complete.chmod(0o600)
    return output


def _read_c_string_symbol(handle: Any, symbol: str) -> str:
    try:
        function = getattr(handle, symbol)
    except AttributeError as exc:
        raise RuntimeError(f"MiniMax-H3 native plugin is missing {symbol}") from exc
    function.argtypes = []
    function.restype = ctypes.c_char_p
    value = function()
    if value is None:
        raise RuntimeError(f"MiniMax-H3 native plugin returned null from {symbol}")
    return value.decode("utf-8")


def _validate_loaded_plugin(handle: Any, *, expected_build_identity: str | None) -> None:
    identity = _read_c_string_symbol(handle, _PLUGIN_IDENTITY_SYMBOL)
    if identity != _PLUGIN_IDENTITY:
        raise RuntimeError(
            f"MiniMax-H3 native plugin identity mismatch: {identity!r} != {_PLUGIN_IDENTITY!r}"
        )
    try:
        abi_function = getattr(handle, _PLUGIN_ABI_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError(f"MiniMax-H3 native plugin is missing {_PLUGIN_ABI_SYMBOL}") from exc
    abi_function.argtypes = []
    abi_function.restype = ctypes.c_uint32
    abi_version = int(abi_function())
    if abi_version != _PLUGIN_ABI_VERSION:
        raise RuntimeError(
            f"MiniMax-H3 native plugin ABI mismatch: {abi_version} != {_PLUGIN_ABI_VERSION}"
        )
    build_identity = _read_c_string_symbol(handle, _PLUGIN_BUILD_IDENTITY_SYMBOL)
    if not build_identity:
        raise RuntimeError("MiniMax-H3 native plugin build identity is empty")
    if expected_build_identity is not None and build_identity != expected_build_identity:
        raise RuntimeError(
            "MiniMax-H3 native plugin build identity mismatch: "
            f"{build_identity!r} != {expected_build_identity!r}"
        )
    try:
        registry_function = getattr(handle, _PLUGIN_REGISTRY_SYMBOL)
    except AttributeError as exc:
        raise RuntimeError(
            f"MiniMax-H3 native plugin is missing {_PLUGIN_REGISTRY_SYMBOL}"
        ) from exc
    registry_function.argtypes = []
    registry_function.restype = ctypes.c_bool
    if not bool(registry_function()):
        raise RuntimeError(
            "MiniMax-H3 native plugin creators do not own their TensorRT registry entries"
        )


def load_native_plugin(*, verbose: bool = False) -> Path:
    """Load exactly one validated H3 native plugin DSO into this process."""

    path = ensure_native_plugin(verbose=verbose).resolve()
    if _FAILED_PLUGIN_HANDLES:
        failed = ", ".join(str(failed_path) for failed_path in _FAILED_PLUGIN_HANDLES)
        raise RuntimeError(f"A previous MiniMax-H3 native plugin load failed: {failed}")
    if path in _PLUGIN_HANDLES:
        return path
    if _PLUGIN_HANDLES:
        loaded = ", ".join(str(loaded_path) for loaded_path in _PLUGIN_HANDLES)
        raise RuntimeError(f"A different MiniMax-H3 native plugin is already loaded: {loaded}")
    handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    expected_build_identity = None
    if not os.environ.get(_PLUGIN_ENV) and re.fullmatch(r"[0-9a-f]{16}", path.parent.name):
        expected_build_identity = path.parent.name
    try:
        _validate_loaded_plugin(handle, expected_build_identity=expected_build_identity)
    except Exception:
        # Plugin constructors may already have registered creators. Retain the
        # rejected DSO and poison this process rather than allowing ctypes to
        # dlclose it while TensorRT still holds creator pointers.
        _FAILED_PLUGIN_HANDLES[path] = handle
        raise
    _PLUGIN_HANDLES[path] = handle
    return path


def _plugin_creator(trt_module: Any, plugin_name: str = _PLUGIN_NAME) -> Any:
    load_native_plugin()
    registry_fn = getattr(trt_module, "get_plugin_registry", None)
    if registry_fn is None:
        raise RuntimeError("TensorRT does not expose a plugin registry")
    registry = registry_fn()
    get_creator = getattr(registry, "get_creator", None)
    if get_creator is None:
        raise RuntimeError("TensorRT does not expose the IPluginV3 creator registry API")
    try:
        creator = get_creator(plugin_name, _PLUGIN_VERSION, "")
    except TypeError:
        creator = get_creator(plugin_name, _PLUGIN_VERSION)
    if creator is None:
        raise RuntimeError(
            f"TensorRT V3 plugin creator {plugin_name} v{_PLUGIN_VERSION} was not registered"
        )
    return creator


def _add_fixed_plugin(
    network: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    trt_module: Any,
    name: str,
    plugin_name: str,
) -> Any:
    add_plugin = getattr(network, "add_plugin_v3", None)
    if add_plugin is None:
        raise RuntimeError("TensorRT network does not support IPluginV3 layers")
    fields = trt_module.PluginFieldCollection([])
    plugin = _plugin_creator(trt_module, plugin_name).create_plugin(
        name, fields, trt_module.TensorRTPhase.BUILD
    )
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create the {name} plugin")
    layer = add_plugin([query, key, value], [], plugin)
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add the {name} plugin layer")
    layer.name = name
    layer.metadata = f"trtmc.native_op={plugin_name};source={name}"
    output = layer.get_output(0)
    if output is None:
        raise RuntimeError(f"TensorRT {name} plugin layer has no output")
    output.name = name
    return output


def add_vision_attention_plugin(
    network: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    trt_module: Any,
    name: str,
) -> Any:
    """Add exact HF MiniMax-H3 vision SDPA for row-major ``[L, 1152]`` Q/K/V."""

    return _add_fixed_plugin(
        network,
        query,
        key,
        value,
        trt_module=trt_module,
        name=name,
        plugin_name=_VISION_PLUGIN_NAME,
    )


def add_patch_embed_plugin(
    network: Any,
    pixels: Any,
    weight: Any,
    bias: Any,
    *,
    trt_module: Any,
    name: str,
) -> Any:
    """Add exact HF Conv3d patch embedding for row-major pixels ``[L,1536]``."""

    return _add_fixed_plugin(
        network,
        pixels,
        weight,
        bias,
        trt_module=trt_module,
        name=name,
        plugin_name=_PATCH_EMBED_PLUGIN_NAME,
    )


def add_linear_plugin(
    network: Any,
    tensor: Any,
    weight: Any,
    bias: Any,
    *,
    trt_module: Any,
    name: str,
) -> Any:
    """Add exact HF BF16 linear for ``[rows,in] @ [out,in].T + [out]``."""

    return _add_fixed_plugin(
        network,
        tensor,
        weight,
        bias,
        trt_module=trt_module,
        name=name,
        plugin_name=_LINEAR_PLUGIN_NAME,
    )


def add_layer_norm_plugin(
    network: Any,
    tensor: Any,
    weight: Any,
    bias: Any,
    *,
    trt_module: Any,
    name: str,
) -> Any:
    """Add exact HF BF16 LayerNorm over the final width with epsilon ``1e-6``."""

    return _add_fixed_plugin(
        network,
        tensor,
        weight,
        bias,
        trt_module=trt_module,
        name=name,
        plugin_name=_LAYER_NORM_PLUGIN_NAME,
    )


def add_audio_encoder_plugin(
    network: Any,
    audio_samples: Any,
    module_bytes: bytes,
    *,
    trt_module: Any,
    name: str = "audio_encoder",
) -> Any:
    """Add the exact FP32 TorchScript audio encoder with its module embedded in the plan."""

    if not isinstance(module_bytes, bytes):
        raise TypeError("MiniMax-H3 audio encoder TorchScript module must be bytes")
    if not _AUDIO_ENCODER_MODULE_MIN_BYTES <= len(module_bytes) <= _AUDIO_ENCODER_MODULE_MAX_BYTES:
        raise ValueError(
            "MiniMax-H3 audio encoder TorchScript module must be between 300 and 400 MiB"
        )
    if not module_bytes.startswith(b"PK\x03\x04"):
        raise ValueError("MiniMax-H3 audio encoder TorchScript module must be a ZIP archive")
    add_plugin = getattr(network, "add_plugin_v3", None)
    if add_plugin is None:
        raise RuntimeError("TensorRT network does not support IPluginV3 layers")
    module_array = np.frombuffer(module_bytes, dtype=np.int8)
    _AUDIO_ENCODER_MODULE_KEEPALIVE[id(network)] = module_array
    try:
        module_layer = network.add_constant((len(module_bytes),), module_array)
        if module_layer is None:
            raise RuntimeError("TensorRT rejected the embedded audio encoder module constant")
        module_tensor = module_layer.get_output(0)
        if module_tensor is None:
            raise RuntimeError("TensorRT audio encoder module constant has no output")
        fields = trt_module.PluginFieldCollection([])
        plugin = _plugin_creator(trt_module, _AUDIO_ENCODER_PLUGIN_NAME).create_plugin(
            name, fields, trt_module.TensorRTPhase.BUILD
        )
        if plugin is None:
            raise RuntimeError(f"TensorRT failed to create the {name} plugin")
        layer = add_plugin([audio_samples, module_tensor], [], plugin)
        if layer is None:
            raise RuntimeError(f"TensorRT failed to add the {name} plugin layer")
        layer.name = name
        layer.metadata = (
            f"trtmc.native_op={_AUDIO_ENCODER_PLUGIN_NAME};source={name};"
            f"module_bytes={len(module_bytes)};"
            f"module_sha256={hashlib.sha256(module_bytes).hexdigest()}"
        )
        output = layer.get_output(0)
        if output is None:
            raise RuntimeError(f"TensorRT {name} plugin layer has no output")
        output.name = name
        return output
    except BaseException:
        _AUDIO_ENCODER_MODULE_KEEPALIVE.pop(id(network), None)
        raise


def release_audio_encoder_module_storage(network: Any) -> None:
    """Release the embedded TorchScript backing buffer after engine serialization."""

    _AUDIO_ENCODER_MODULE_KEEPALIVE.pop(id(network), None)


__all__ = [
    "add_audio_encoder_plugin",
    "add_layer_norm_plugin",
    "add_linear_plugin",
    "add_patch_embed_plugin",
    "add_vision_attention_plugin",
    "ensure_native_plugin",
    "load_native_plugin",
    "native_plugin_source_files",
    "release_audio_encoder_module_storage",
]
