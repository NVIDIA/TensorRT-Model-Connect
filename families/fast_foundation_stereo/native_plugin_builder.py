# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and register the model-owned TensorRT native plugins."""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import Any

_PLUGIN_NAME = "FastFoundationStereoCombinedVolume"
_GEOMETRY_VOLUME_CONVC1_PLUGIN_NAME = "FastFoundationStereoGeometryVolumeConvc1"
_SPATIAL_ATTENTION_REDUCE_PLUGIN_NAME = "FastFoundationStereoSpatialAttentionReduce"
_POST8_SUM_PLUGIN_NAME = "FastFoundationStereoPost8Sum"
_FULL_VOLUME_LEAKY_PLUGIN_NAME = "FastFoundationStereoFullVolumeLeaky"
_DEFAULT_PLUGIN_VERSION = "1"
_PLUGIN_VERSIONS = {_PLUGIN_NAME: "2"}
_PLUGIN_HANDLE: Any | None = None
_PLUGIN_PATH: Path | None = None


def _plugin_version(plugin_name: str) -> str:
    return _PLUGIN_VERSIONS.get(plugin_name, _DEFAULT_PLUGIN_VERSION)


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    """Compile one fresh family-owned plugin DSO."""
    source_dir = Path(__file__).parent / "runtime" / "native_plugins"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Fast Foundation Stereo native sources are missing: {source_dir}")
    build_dir = Path(tempfile.mkdtemp(prefix="trtmc-fast-stereo-plugin-"))
    output = build_dir / "libtrtmc_fast_foundation_stereo_native_plugin.so"
    configure = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
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
        {} if verbose else {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    )
    try:
        subprocess.run(configure, check=True, **kwargs)
        subprocess.run(build, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Fast Foundation Stereo native plugin build failed\n{getattr(error, 'stdout', '') or ''}"
        ) from error
    if not output.is_file():
        raise RuntimeError(f"Native plugin build did not produce {output}")
    return output


def load_native_plugin(*, verbose: bool = False) -> Path:
    """Load the DSO globally so TensorRT can discover its creator."""

    global _PLUGIN_HANDLE, _PLUGIN_PATH
    if _PLUGIN_PATH is not None:
        return _PLUGIN_PATH
    path = ensure_native_plugin(verbose=verbose).resolve()
    _PLUGIN_HANDLE = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PLUGIN_PATH = path
    return path


def _plugin_creator(trt_module: Any, plugin_name: str = _PLUGIN_NAME) -> Any:
    load_native_plugin()
    plugin_version = _plugin_version(plugin_name)
    creator = trt_module.get_plugin_registry().get_creator(plugin_name, plugin_version, "")
    if creator is None:
        raise RuntimeError(f"TensorRT plugin creator {plugin_name} v{plugin_version} is missing")
    return creator


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
    plugin = _plugin_creator(trt_module, plugin_name).create_plugin(name, fields)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create the {name} plugin")
    return _named_plugin_outputs(network.add_plugin_v2(inputs, plugin), name, output_names)


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
    creator = _plugin_creator(trt_module, plugin_name)
    plugin = creator.create_plugin(name, fields, trt_module.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError(f"TensorRT failed to create the {name} plugin")
    return _named_plugin_outputs(network.add_plugin_v3(inputs, [], plugin), name, output_names)


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
