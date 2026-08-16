# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and register the model-owned TensorRT groupwise-correlation plugin."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_PLUGIN_ENV = "TRTMC_FAST_FOUNDATION_STEREO_NATIVE_PLUGIN_LIBRARY"
_BUILD_DIR_ENV = "TRTMC_FAST_FOUNDATION_STEREO_NATIVE_PLUGIN_BUILD_DIR"
_PLUGIN_NAME = "FastFoundationStereoGwc"
_PLUGIN_VERSION = "1"
_PLUGIN_HANDLES: dict[Path, Any] = {}


def _source_digest(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix in {".cpp", ".cu", ".h", ".txt"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _active_tensorrt_cmake_hints() -> list[str]:
    """Pin the plugin build to the TensorRT ABI loaded by this process."""

    from tensorrt_model_connect import trt_compat

    hints: list[str] = []
    libraries = list(
        dict.fromkeys(
            Path(candidate).resolve()
            for candidate in trt_compat.loaded_libnvinfer_paths()
            if Path(candidate).is_file()
        )
    )
    if libraries:
        active_major = trt_compat.tensorrt_version().split(".", 1)[0]
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
        hints.append(f"-DFAST_FOUNDATION_STEREO_TRT_LIBRARY={library}")

    for variable in ("TRTMC_TRT_INCLUDE_DIR", "TRT_INC_DIR"):
        candidate = os.environ.get(variable)
        if candidate and (Path(candidate) / "NvInferRuntime.h").is_file():
            hints.append(f"-DFAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR={Path(candidate).resolve()}")
            break
    return hints


def _plugin_cache_key(
    source_dir: Path,
    cmake_hints: list[str],
    *,
    runtime_version: str = "",
) -> str:
    digest = hashlib.sha256()
    digest.update(_source_digest(source_dir).encode("ascii"))
    for hint in (
        f"tensorrt={runtime_version}",
        f"cuda={os.environ.get('CUDA_VERSION', '')}",
        f"architectures={os.environ.get('CMAKE_CUDA_ARCHITECTURES', '89-real;89-virtual')}",
        *cmake_hints,
    ):
        digest.update(b"\0")
        digest.update(hint.encode("utf-8"))
    return digest.hexdigest()[:16]


@contextmanager
def _exclusive_build_lock(build_base: Path, source_digest: str) -> Iterator[None]:
    if build_base.is_symlink():
        raise RuntimeError(f"Native plugin build cache cannot be a symlink: {build_base}")
    build_base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if build_base.stat().st_uid != os.geteuid():
        raise RuntimeError(f"Native plugin build cache is not owned by this user: {build_base}")
    build_base.chmod(0o700)
    lock_path = build_base / f".{source_digest}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    """Return a built plugin DSO, compiling the small standalone target if needed."""

    override = os.environ.get(_PLUGIN_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{_PLUGIN_ENV} does not exist: {path}")
        return path

    from tensorrt_model_connect import trt_compat

    trt_compat.load_module()
    source_dir = Path(__file__).with_name("native_plugins")
    default_build_base = Path(tempfile.gettempdir()) / (
        f"trtmc-fast-foundation-stereo-native-plugin-{os.geteuid()}"
    )
    build_base = Path(
        os.environ.get(
            _BUILD_DIR_ENV,
            str(default_build_base),
        )
    ).expanduser()
    cmake_hints = _active_tensorrt_cmake_hints()
    source_digest = _plugin_cache_key(
        source_dir,
        cmake_hints,
        runtime_version=trt_compat.tensorrt_version(),
    )
    build_dir = build_base / source_digest
    output = build_dir / "libtrtmc_fast_foundation_stereo_native_plugin.so"
    complete = build_dir / ".complete"

    with _exclusive_build_lock(build_base, source_digest):
        if output.is_file() and complete.is_file():
            return output
        build_dir.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        complete.unlink(missing_ok=True)
        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            *cmake_hints,
        ]
        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "trtmc_fast_foundation_stereo_native_plugin",
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
            subprocess.run(build, check=True, **kwargs)
        except subprocess.CalledProcessError as exc:
            output_text = getattr(exc, "stdout", "") or ""
            raise RuntimeError(
                f"Fast Foundation Stereo native plugin build failed\n{output_text}"
            ) from exc
        if not output.is_file():
            raise RuntimeError(f"Native plugin build did not produce {output}")
        complete.write_text("complete\n", encoding="utf-8")
    return output


def load_native_plugin(*, verbose: bool = False) -> Path:
    """Load the DSO globally so TensorRT can discover its creator."""

    path = ensure_native_plugin(verbose=verbose).resolve()
    if path in _PLUGIN_HANDLES:
        return path
    if _PLUGIN_HANDLES:
        loaded = ", ".join(str(loaded_path) for loaded_path in _PLUGIN_HANDLES)
        raise RuntimeError(
            "A different Fast Foundation Stereo native plugin is already loaded: " + loaded
        )
    handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PLUGIN_HANDLES[path] = handle
    return path


def _plugin_creator(trt_module: Any) -> Any:
    load_native_plugin()
    registry_fn = getattr(trt_module, "get_plugin_registry", None)
    if registry_fn is None:
        raise RuntimeError("TensorRT does not expose a plugin registry")
    registry = registry_fn()
    creator = None
    get_creator = getattr(registry, "get_plugin_creator", None)
    if get_creator is not None:
        try:
            creator = get_creator(_PLUGIN_NAME, _PLUGIN_VERSION, "")
        except TypeError:
            creator = get_creator(_PLUGIN_NAME, _PLUGIN_VERSION)
    if creator is None:
        get_creator = getattr(registry, "get_creator", None)
        if get_creator is not None:
            try:
                creator = get_creator(_PLUGIN_NAME, _PLUGIN_VERSION, "")
            except TypeError:
                creator = get_creator(_PLUGIN_NAME, _PLUGIN_VERSION)
    if creator is None:
        raise RuntimeError(
            f"TensorRT plugin creator {_PLUGIN_NAME} v{_PLUGIN_VERSION} was not registered"
        )
    return creator


def add_gwc_plugin(
    network: Any,
    reference: Any,
    target: Any,
    *,
    trt_module: Any,
    name: str = "gwc_volume",
) -> Any:
    """Add the fixed L4-tuned GWC plugin to a native TensorRT network."""

    add_plugin = getattr(network, "add_plugin_v2", None)
    if add_plugin is None:
        raise RuntimeError("TensorRT network does not support IPluginV2 layers")
    creator = _plugin_creator(trt_module)
    fields = trt_module.PluginFieldCollection([])
    plugin = creator.create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError("TensorRT failed to create the Fast Foundation Stereo GWC plugin")
    layer = add_plugin([reference, target], plugin)
    if layer is None:
        raise RuntimeError("TensorRT failed to add the Fast Foundation Stereo GWC plugin layer")
    layer.name = name
    output = layer.get_output(0)
    output.name = name
    return output


__all__ = ["add_gwc_plugin", "ensure_native_plugin", "load_native_plugin"]
