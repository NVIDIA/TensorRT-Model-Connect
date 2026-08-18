# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and register the model-owned TensorRT native plugins."""

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
_PLUGIN_NAME = "FastFoundationStereoCombinedVolume"
_GEOMETRY_VOLUME_CONVC1_PLUGIN_NAME = "FastFoundationStereoGeometryVolumeConvc1"
_SPATIAL_ATTENTION_REDUCE_PLUGIN_NAME = "FastFoundationStereoSpatialAttentionReduce"
_POST8_SUM_PLUGIN_NAME = "FastFoundationStereoPost8Sum"
_FULL_VOLUME_LEAKY_PLUGIN_NAME = "FastFoundationStereoFullVolumeLeaky"
_DEFAULT_PLUGIN_VERSION = "1"
_PLUGIN_VERSIONS = {_PLUGIN_NAME: "2"}
_DEFAULT_CUDA_ARCHITECTURES = "89-real;89-virtual"
_PLUGIN_HANDLES: dict[Path, Any] = {}


def _source_digest(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix in {".cpp", ".cu", ".h", ".txt"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _normalized_tensorrt_version(raw_version: str | None = None) -> str:
    """Return the exact four-component version used to validate C++ headers."""

    if raw_version is None:
        from tensorrt_model_connect import trt_compat

        raw_version = trt_compat.tensorrt_version()
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)\.(\d+)", raw_version)
    if match is None:
        raise RuntimeError(
            "Cannot validate TensorRT C++ headers: active TensorRT version "
            f"{raw_version!r} is not major.minor.patch.build"
        )
    return ".".join(match.groups())


def _active_tensorrt_cmake_hints() -> list[str]:
    """Pin the plugin build to the TensorRT ABI loaded by this process."""

    from tensorrt_model_connect import trt_compat

    runtime_version = _normalized_tensorrt_version(trt_compat.tensorrt_version())
    hints: list[str] = [f"-DFAST_FOUNDATION_STEREO_TRT_EXPECTED_VERSION={runtime_version}"]
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
        hints.append(f"-DFAST_FOUNDATION_STEREO_TRT_LIBRARY={library}")

    for variable in ("TRTMC_TRT_INCLUDE_DIR", "TRT_INC_DIR"):
        candidate = os.environ.get(variable)
        if candidate and (Path(candidate) / "NvInferRuntime.h").is_file():
            hints.append(f"-DFAST_FOUNDATION_STEREO_TRT_INCLUDE_DIR={Path(candidate).resolve()}")
            break
    return hints


def _cuda_architectures() -> str:
    """Return the CUDA architectures used by both cache identity and CMake."""

    return os.environ.get("CMAKE_CUDA_ARCHITECTURES", _DEFAULT_CUDA_ARCHITECTURES)


def _plugin_version(plugin_name: str) -> str:
    """Keep ABI revisions scoped to the plugin whose tensor contract changed."""

    return _PLUGIN_VERSIONS.get(plugin_name, _DEFAULT_PLUGIN_VERSION)


def _plugin_cache_key(
    source_dir: Path,
    cmake_hints: list[str],
    *,
    cuda_architectures: str,
    runtime_version: str = "",
) -> str:
    digest = hashlib.sha256()
    digest.update(_source_digest(source_dir).encode("ascii"))
    for hint in (
        "source-layout=runtime-model-v2",
        f"tensorrt={runtime_version}",
        f"cuda={os.environ.get('CUDA_VERSION', '')}",
        f"architectures={cuda_architectures}",
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
    if not source_dir.is_dir():
        source_dir = (
            Path(__file__).resolve().parents[4]
            / "src/runtime/models/fast_foundation_stereo/native_plugins"
        )
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
    cuda_architectures = _cuda_architectures()
    source_digest = _plugin_cache_key(
        source_dir,
        cmake_hints,
        cuda_architectures=cuda_architectures,
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
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
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


def _resolve_plugin_creator(trt_module: Any, plugin_name: str, *, v3_only: bool) -> Any:
    load_native_plugin()
    plugin_version = _plugin_version(plugin_name)
    registry_fn = getattr(trt_module, "get_plugin_registry", None)
    if registry_fn is None:
        raise RuntimeError("TensorRT does not expose a plugin registry")
    registry = registry_fn()
    methods = ("get_creator",) if v3_only else ("get_plugin_creator", "get_creator")
    for method in methods:
        get_creator = getattr(registry, method, None)
        if get_creator is None:
            continue
        try:
            creator = get_creator(plugin_name, plugin_version, "")
        except TypeError:
            creator = get_creator(plugin_name, plugin_version)
        if creator is not None:
            return creator
    kind = "V3 plugin creator" if v3_only else "plugin creator"
    raise RuntimeError(f"TensorRT {kind} {plugin_name} v{plugin_version} was not registered")


def _plugin_creator(trt_module: Any, plugin_name: str = _PLUGIN_NAME) -> Any:
    return _resolve_plugin_creator(trt_module, plugin_name, v3_only=False)


def _plugin_v3_creator(trt_module: Any, plugin_name: str) -> Any:
    """Resolve a V3 creator without probing the V2-only registry API first."""

    return _resolve_plugin_creator(trt_module, plugin_name, v3_only=True)


def _named_plugin_outputs(layer: Any, name: str, output_names: tuple[str, ...]) -> tuple[Any, ...]:
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add the {name} plugin layer")
    layer.name = name
    outputs = tuple(layer.get_output(index) for index in range(len(output_names)))
    for output, output_name in zip(outputs, output_names, strict=True):
        output.name = output_name
    return outputs


def _add_v2_plugin(
    network: Any,
    inputs: list[Any],
    *,
    trt_module: Any,
    plugin_name: str,
    name: str,
    fields: Any,
    output_names: tuple[str, ...],
) -> tuple[Any, ...]:
    add_plugin = getattr(network, "add_plugin_v2", None)
    if add_plugin is None:
        raise RuntimeError("TensorRT network does not support IPluginV2 layers")
    plugin = _plugin_creator(trt_module, plugin_name).create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create the {name} plugin")
    return _named_plugin_outputs(add_plugin(inputs, plugin), name, output_names)


def _add_v3_plugin(
    network: Any,
    inputs: list[Any],
    *,
    trt_module: Any,
    plugin_name: str,
    name: str,
    fields: Any,
    output_names: tuple[str, ...],
) -> tuple[Any, ...]:
    add_plugin = getattr(network, "add_plugin_v3", None)
    if add_plugin is None:
        raise RuntimeError("TensorRT network does not support IPluginV3 layers")
    creator = _plugin_v3_creator(trt_module, plugin_name)
    plugin = creator.create_plugin(name, fields, trt_module.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create the {name} plugin")
    return _named_plugin_outputs(add_plugin(inputs, [], plugin), name, output_names)


def add_combined_volume_plugin(
    network: Any,
    reference: Any,
    target: Any,
    left_projected: Any,
    right_projected: Any,
    *,
    trt_module: Any,
    name: str = "combined_volume",
) -> Any:
    """Add the fixed-shape fused stereo volume plugin to a TensorRT network."""

    fields = trt_module.PluginFieldCollection([])
    return _add_v2_plugin(
        network,
        [reference, target, left_projected, right_projected],
        trt_module=trt_module,
        plugin_name=_PLUGIN_NAME,
        name=name,
        fields=fields,
        output_names=(name,),
    )[0]


def add_geometry_volume_convc1_plugin(
    network: Any,
    disparity: Any,
    volume: Any,
    correlation0: Any,
    correlation1: Any,
    packed_weight: Any,
    packed_bias: Any,
    *,
    trt_module: Any,
    name: str,
) -> Any:
    """Fuse direct DHWC8 volume sampling with the first motion convolution."""

    fields = trt_module.PluginFieldCollection([])
    return _add_v2_plugin(
        network,
        [disparity, volume, correlation0, correlation1, packed_weight, packed_bias],
        trt_module=trt_module,
        plugin_name=_GEOMETRY_VOLUME_CONVC1_PLUGIN_NAME,
        name=name,
        fields=fields,
        output_names=(name,),
    )[0]


def add_spatial_attention_reduce_plugin(
    network: Any,
    tensor: Any,
    *,
    trt_module: Any,
    name: str = "spatial_attention_reduce",
) -> tuple[Any, Any]:
    """Add the fixed-shape channel mean/max plugin to a TensorRT network."""

    fields = trt_module.PluginFieldCollection([])
    average, maximum = _add_v2_plugin(
        network,
        [tensor],
        trt_module=trt_module,
        plugin_name=_SPATIAL_ATTENTION_REDUCE_PLUGIN_NAME,
        name=name,
        fields=fields,
        output_names=(f"{name}_average", f"{name}_maximum"),
    )
    return average, maximum


def add_post8_sum_plugin(
    network: Any,
    linear: Any,
    skip: Any,
    *,
    trt_module: Any,
    name: str = "post8_to_4_sum",
) -> Any:
    """Transpose the fixed post8 LINEAR tensor and add its DHWC8 skip."""

    fields = trt_module.PluginFieldCollection([])
    return _add_v3_plugin(
        network,
        [linear, skip],
        trt_module=trt_module,
        plugin_name=_POST8_SUM_PLUGIN_NAME,
        name=name,
        fields=fields,
        output_names=(name,),
    )[0]


def add_full_volume_leaky_plugin(
    network: Any,
    tensor: Any,
    *,
    trt_module: Any,
    name: str,
) -> Any:
    """Replace one exact FP16 DHWC8 full-volume LeakyReLU with its V3 kernel."""

    fields = trt_module.PluginFieldCollection([])
    return _add_v3_plugin(
        network,
        [tensor],
        trt_module=trt_module,
        plugin_name=_FULL_VOLUME_LEAKY_PLUGIN_NAME,
        name=name,
        fields=fields,
        output_names=(name,),
    )[0]


__all__ = [
    "add_combined_volume_plugin",
    "add_full_volume_leaky_plugin",
    "add_geometry_volume_convc1_plugin",
    "add_post8_sum_plugin",
    "add_spatial_attention_reduce_plugin",
    "ensure_native_plugin",
    "load_native_plugin",
]
