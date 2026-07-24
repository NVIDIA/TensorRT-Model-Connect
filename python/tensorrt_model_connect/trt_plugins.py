# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and validate TensorRT plugins shared by native model builders."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import trt_compat


_PLUGIN_DSO = "libtrtmc_trt_plugins.so"
_RUNTIME_KV_CAPABILITY_CUDNN_SDPA = 1 << 0
_RUNTIME_STACK_KEYS = {
    "sm",
    "tensorrt",
    "cuda_runtime",
    "cudnn_backend",
    "cudnn_frontend_revision",
    "nvrtc",
    "driver",
}
_loaded_library: ctypes.CDLL | None = None


def enable_runtime_memory_features(
    builder_config: Any,
) -> Any:
    """Enable the TensorRT preview feature required by runtime KV.

    Runtime activation resizing lets USER_MANAGED context memory follow actual
    invocation shapes. The segmented-attention ABI keeps history read-only and
    returns current K/V through ordinary exact-Sq engine outputs, so it does not
    require TensorRT's aliased-plugin-I/O preview feature.
    """

    trt = trt_compat.get_trt()
    feature_name = "RUNTIME_ACTIVATION_RESIZE_10_10"
    feature = getattr(
        trt.PreviewFeature,
        feature_name,
        None,
    )
    if feature is None:
        raise RuntimeError(
            "qualified runtime-memory graph requires TensorRT "
            f"PreviewFeature.{feature_name}"
        )
    builder_config.set_preview_feature(feature, True)
    if not builder_config.get_preview_feature(feature):
        raise RuntimeError(
            "TensorRT refused to enable required runtime-memory "
            f"preview feature {feature_name}"
        )
    return feature


def _plugin_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("TRTMC_TRT_PLUGIN_LIBRARY")
    if override:
        candidates.append(Path(override))

    package_dir = Path(__file__).resolve().parent
    candidates.extend(
        (
            package_dir / "bin" / _PLUGIN_DSO,
            package_dir / _PLUGIN_DSO,
            package_dir / "lib" / _PLUGIN_DSO,
            package_dir.parent / _PLUGIN_DSO,
            package_dir.parent / "lib" / _PLUGIN_DSO,
        )
    )

    # Source-tree builds keep the common DSO in the selected CMake build
    # directory. This fallback is deterministic and only considered after the
    # explicit/package locations.
    try:
        repo_root = package_dir.parents[1]
        candidates.extend(
            sorted(repo_root.glob(f"build*/{_PLUGIN_DSO}"))
        )
    except IndexError:
        pass

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def load_runtime_kv_plugins() -> ctypes.CDLL:
    """Load the common DSO globally and verify its ABI-v2 handshake."""

    global _loaded_library
    if _loaded_library is not None:
        return _loaded_library

    attempted: list[str] = []
    for candidate in _plugin_candidates():
        attempted.append(str(candidate))
        if not candidate.is_file():
            continue
        library = ctypes.CDLL(
            str(candidate),
            mode=ctypes.RTLD_GLOBAL,
        )
        abi = library.trtmc_runtime_kv_plugin_abi_version
        abi.argtypes = []
        abi.restype = ctypes.c_int32
        actual = int(abi())
        if actual != 2:
            raise RuntimeError(
                "TensorRT runtime-KV plugin ABI mismatch: "
                f"expected 2, got {actual} from {candidate}"
            )
        _loaded_library = library
        return library

    raise RuntimeError(
        f"Unable to locate {_PLUGIN_DSO}; checked: "
        + ", ".join(attempted)
    )


def query_runtime_kv_plugin_stack() -> dict[str, str]:
    """Return independently detected facts from the native plugin DSO.

    The returned values describe the libraries and device that will actually
    build/execute the segmented-attention plugin. They are not derived from a
    model manifest or bundle header.
    """

    library = load_runtime_kv_plugins()
    try:
        query = library.trtmc_runtime_kv_plugin_runtime_stack_json_v1
    except AttributeError as error:
        raise RuntimeError(
            "TensorRT runtime-KV plugin does not export runtime-stack "
            "introspection V1"
        ) from error
    query.argtypes = []
    query.restype = ctypes.c_char_p
    raw = query()
    if raw is None:
        raise RuntimeError(
            "TensorRT runtime-KV plugin returned no runtime-stack evidence"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "TensorRT runtime-KV plugin returned invalid runtime-stack JSON"
        ) from error
    if not isinstance(value, dict) or set(value) != _RUNTIME_STACK_KEYS:
        raise RuntimeError(
            "TensorRT runtime-KV plugin returned an incompatible "
            "runtime-stack schema"
        )
    if any(not isinstance(value[key], str) or not value[key]
           for key in _RUNTIME_STACK_KEYS):
        raise RuntimeError(
            "TensorRT runtime-KV plugin runtime-stack evidence is incomplete"
        )
    return {key: value[key] for key in _RUNTIME_STACK_KEYS}


def _require_runtime_kv_capabilities(
    library: ctypes.CDLL,
    required: int,
) -> None:
    """Fail closed when a qualified plugin DSO lacks a performance path."""

    try:
        capabilities = (
            library.trtmc_runtime_kv_plugin_capabilities
        )
    except AttributeError as error:
        raise RuntimeError(
            "TensorRT runtime-KV plugin does not export its "
            "performance capabilities"
        ) from error
    capabilities.argtypes = []
    capabilities.restype = ctypes.c_uint64
    actual = int(capabilities())
    missing = required & ~actual
    if missing:
        raise RuntimeError(
            "qualified runtime-memory graph requires the "
            "cuDNN 9.20 SDPA runtime-KV capability; "
            f"plugin capabilities=0x{actual:x}"
        )


def create_native_contiguous_attention(
    network: Any,
    inputs: list[Any],
    *,
    layer_name: str,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    chunk_limit: int,
) -> Any:
    """Add ABI-v2 segmented attention and return its context output."""

    library = load_runtime_kv_plugins()
    _require_runtime_kv_capabilities(
        library,
        _RUNTIME_KV_CAPABILITY_CUDNN_SDPA,
    )
    trt = trt_compat.get_trt()
    creator = trt.get_plugin_registry().get_creator(
        "NativeContiguousAttention", "2", ""
    )
    if creator is None:
        raise RuntimeError(
            "NativeContiguousAttention v2 creator was not registered"
        )

    values = {
        "abi_version": 2,
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "chunk_limit": int(chunk_limit),
    }
    storage = {
        name: np.asarray([value], dtype=np.int32)
        for name, value in values.items()
    }
    fields = [
        trt.PluginField(
            name,
            storage[name],
            trt.PluginFieldType.INT32,
        )
        for name in values
    ]
    plugin = creator.create_plugin(
        layer_name,
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError(
            f"NativeContiguousAttention v2 creation failed for {layer_name}"
        )
    layer = network.add_plugin_v3(inputs, [], plugin)
    if layer is None:
        raise RuntimeError(
            f"TensorRT rejected NativeContiguousAttention for {layer_name}"
        )
    layer.name = layer_name
    return layer.get_output(0)
