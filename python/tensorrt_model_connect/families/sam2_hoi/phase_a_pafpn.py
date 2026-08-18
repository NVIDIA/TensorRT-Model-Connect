# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned Phase-A PAFPN leaf graph and TensorRT plan builders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from tensorrt_model_connect.bundle_writer import FileBundleSection, bundle_section_from_file
from tensorrt_model_connect import trt_compat

from . import native_graph_ops as graph_ops
from . import native_image_builder as image_builder
from . import pafpn_bn_invstd
from .source_export import (
    HOI_DETECTOR_SECTION,
    INTERACTION_SECTION,
    MEMORY_ENCODER_SECTION,
    NATIVE_PLUGIN_SECTION,
    PROMPT_TRACKER_SECTION,
    RECURRENT_TRACKER_SECTION,
)


PHASE_A_CONFIG_KEY = "sam2_hoi_phase_a_pafpn"
PHASE_A_BUILD_OPTION = "phase_a_pafpn"
PAFPN_MANIFEST_SECTION = "sam2_hoi_pafpn_manifest.json"
PAFPN_PLAN_COUNT = 137
PAFPN_PLAN_PREFIX = "sam2_hoi_pafpn_plan_"
PAFPN_PREFIX = "image_encoder.learnable_fpn_module.learnable_fpn_module"
EXTERNAL_INPUTS = ("fpn_input_0", "fpn_input_1", "fpn_input_2")
STAGE_COUNTS = {"Reduce": 3, "TD0": 32, "TD1": 29, "BU0": 32, "BU1": 32, "Out": 9}
PUBLIC_OUTPUTS = (
    ("detector_feature_0", 130),
    ("detector_feature_1", 133),
    ("detector_feature_2", 136),
)
PAFPN_OPT0_CONV_ORDINALS = frozenset({13, 19, 25, 45, 51, 57, 64, 77, 83, 89, 128, 134})


@dataclass(frozen=True)
class TensorABI:
    shape: tuple[int, int, int, int]
    dtype: str


PUBLIC_OUTPUT_ABIS = (
    TensorABI((1, 256, 128, 128), "bfloat16"),
    TensorABI((1, 256, 64, 64), "bfloat16"),
    TensorABI((1, 256, 32, 32), "bfloat16"),
)
EXTERNAL_INPUT_ABIS = {
    "fpn_input_0": TensorABI((1, 256, 128, 128), "bfloat16"),
    "fpn_input_1": TensorABI((1, 256, 64, 64), "float32"),
    "fpn_input_2": TensorABI((1, 256, 32, 32), "bfloat16"),
}


@dataclass(frozen=True)
class ValueRef:
    source_kind: str
    source_name: str | None
    source_node: int | None
    abi: TensorABI

    @classmethod
    def external(cls, name: str, abi: TensorABI) -> "ValueRef":
        return cls("external", name, None, abi)

    @classmethod
    def node(cls, ordinal: int, abi: TensorABI) -> "ValueRef":
        return cls("node", None, ordinal, abi)

    def manifest_source(self) -> dict[str, object]:
        if self.source_kind == "external":
            return {"external": self.source_name}
        return {"node": self.source_node, "tensor": "output"}


@dataclass(frozen=True)
class LeafInput:
    binding: str
    value: ValueRef


@dataclass(frozen=True)
class PafpnLeafSpec:
    ordinal: int
    stage: str
    label: str
    section: str
    kind: str
    inputs: tuple[LeafInput, ...]
    output: TensorABI
    module_prefix: str | None = None
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)

    @property
    def node_id(self) -> str:
        return f"{self.stage}::{self.label}"


def phase_a_plan_section(ordinal: int) -> str:
    if ordinal < 0 or ordinal >= PAFPN_PLAN_COUNT:
        raise ValueError(f"SAM2 HOI PAFPN ordinal is out of range: {ordinal}")
    return f"{PAFPN_PLAN_PREFIX}{ordinal:03d}"


def phase_a_bundle_loading() -> dict[str, object]:
    eager = ["config.json", PAFPN_MANIFEST_SECTION, NATIVE_PLUGIN_SECTION]
    lazy = [
        "engine_plan",
        HOI_DETECTOR_SECTION,
        INTERACTION_SECTION,
        PROMPT_TRACKER_SECTION,
        RECURRENT_TRACKER_SECTION,
        MEMORY_ENCODER_SECTION,
        *(phase_a_plan_section(index) for index in range(PAFPN_PLAN_COUNT)),
    ]
    if len(eager) + len(lazy) != 146 or set(eager) & set(lazy):
        raise AssertionError("SAM2 HOI Phase-A staged section inventory is invalid")
    return {"mode": "staged", "eager_sections": eager, "lazy_sections": lazy}


def phase_a_pafpn_build_policy() -> dict[str, object]:
    """Return the persisted builder policy without claiming cross-GPU tactic identity."""

    return {
        "schema": "sam2-hoi-phase-a-pafpn-build-policy/v1",
        "batch_norm": {
            "invstd": "builder_time_cuda_rsqrtf_fadd_rn",
            "source_operation_order": "CastSubMulMulAddCast",
            "inspector_fusion": "CastAddMulMulAddCast",
            "host_affine_folding": False,
            "helper_source_sha256": pafpn_bn_invstd.HELPER_SOURCE_SHA256,
            "helper_version": pafpn_bn_invstd.HELPER_VERSION,
            "inference_runtime_launch_added": False,
        },
        "silu": {
            "opmath": "fp32",
            "implementation": "native_exp_sum_div_then_output_dtype_cast",
            "plugin_used": False,
        },
        "td1_seam": {
            "ordinal": 35,
            "operation": "second_nearest_resize_output_cast_to_fp32_before_concat",
            "output_dtype": "float32",
        },
        "optimization": {
            "opt0_conv_ordinals": sorted(PAFPN_OPT0_CONV_ORDINALS),
            "all_other_leaf_optimization_level": 3,
            "avg_timing_iterations": 1,
            "timing_cache": "disabled",
            "profiling_verbosity": "DETAILED",
        },
        "qualification": {
            "generic_build_source_exact_tactics_claimed": False,
            "product_gate": "sam2-hoi-full-chain-accuracy-v2",
            "l4_per_leaf_source_exact_requires_external_137_plan_gate": True,
        },
    }


class _SpecBuilder:
    def __init__(self, weights: Mapping[str, np.ndarray]):
        self.weights = weights
        self.specs: list[PafpnLeafSpec] = []

    def _emit(
        self,
        stage: str,
        label: str,
        kind: str,
        inputs: tuple[LeafInput, ...],
        output: TensorABI,
        *,
        module_prefix: str | None = None,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> ValueRef:
        ordinal = len(self.specs)
        self.specs.append(
            PafpnLeafSpec(
                ordinal=ordinal,
                stage=stage,
                label=label,
                section=phase_a_plan_section(ordinal),
                kind=kind,
                inputs=inputs,
                output=output,
                module_prefix=module_prefix,
                stride=stride,
                padding=padding,
            )
        )
        return ValueRef.node(ordinal, output)

    def conv_bn_silu(
        self,
        stage: str,
        prefix: str,
        source: ValueRef,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> ValueRef:
        weight = np.asarray(self.weights[f"{prefix}.conv.weight"])
        if weight.ndim != 4 or source.abi.shape[1] != int(weight.shape[1]):
            raise ValueError(f"SAM2 HOI PAFPN convolution contract drift: {prefix}")
        batch, _, height, width = source.abi.shape
        out_channels, _, kernel_h, kernel_w = (int(value) for value in weight.shape)
        out_height = (height + 2 * padding[0] - kernel_h) // stride[0] + 1
        out_width = (width + 2 * padding[1] - kernel_w) // stride[1] + 1
        output = TensorABI((batch, out_channels, out_height, out_width), "bfloat16")
        conv = self._emit(
            stage,
            f"{prefix}.conv",
            "conv",
            (LeafInput("input", source),),
            output,
            module_prefix=prefix,
            stride=stride,
            padding=padding,
        )
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            value = np.asarray(self.weights[f"{prefix}.bn.{suffix}"])
            if value.shape != (out_channels,):
                raise ValueError(f"SAM2 HOI PAFPN BatchNorm contract drift: {prefix}.bn.{suffix}")
        bn = self._emit(
            stage,
            f"{prefix}.bn",
            "batch_norm",
            (LeafInput("input", conv),),
            output,
            module_prefix=prefix,
        )
        return self._emit(
            stage,
            f"{prefix}.silu",
            "silu",
            (LeafInput("input", bn),),
            output,
        )

    def concatenate(self, stage: str, label: str, lhs: ValueRef, rhs: ValueRef) -> ValueRef:
        left = lhs.abi.shape
        right = rhs.abi.shape
        if left[0] != right[0] or left[2:] != right[2:]:
            raise ValueError(f"SAM2 HOI PAFPN concatenate shape drift: {label}")
        dtype = "float32" if "float32" in {lhs.abi.dtype, rhs.abi.dtype} else "bfloat16"
        output = TensorABI((left[0], left[1] + right[1], left[2], left[3]), dtype)
        return self._emit(
            stage,
            label,
            "concat",
            (LeafInput("input_0", lhs), LeafInput("input_1", rhs)),
            output,
        )

    def resize_concatenate(
        self,
        stage: str,
        label: str,
        low_resolution: ValueRef,
        lateral: ValueRef,
        *,
        force_fp32: bool = False,
    ) -> ValueRef:
        low = low_resolution.abi.shape
        side = lateral.abi.shape
        if low[0] != side[0] or low[2] * 2 != side[2] or low[3] * 2 != side[3]:
            raise ValueError(f"SAM2 HOI PAFPN resize-concat shape drift: {label}")
        dtype = (
            "float32"
            if force_fp32 or "float32" in {low_resolution.abi.dtype, lateral.abi.dtype}
            else "bfloat16"
        )
        output = TensorABI((side[0], low[1] + side[1], side[2], side[3]), dtype)
        return self._emit(
            stage,
            label,
            "resize_concat",
            (LeafInput("input_0", low_resolution), LeafInput("input_1", lateral)),
            output,
        )

    def csp(self, stage: str, prefix: str, source: ValueRef) -> ValueRef:
        short = self.conv_bn_silu(stage, f"{prefix}.short_conv", source)
        main = self.conv_bn_silu(stage, f"{prefix}.main_conv", source)
        for index in range(3):
            block = f"{prefix}.blocks.{index}"
            main = self.conv_bn_silu(stage, f"{block}.conv1", main)
            main = self.conv_bn_silu(stage, f"{block}.conv2", main, padding=(1, 1))
        merged = self.concatenate(stage, f"{prefix}.concat", main, short)
        return self.conv_bn_silu(stage, f"{prefix}.final_conv", merged)


def _pafpn_parameter_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}

    def add_conv(
        prefix: str,
        input_channels: int,
        output_channels: int,
        kernel: int,
    ) -> None:
        shapes[f"{prefix}.conv.weight"] = (
            output_channels,
            input_channels,
            kernel,
            kernel,
        )
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            shapes[f"{prefix}.bn.{suffix}"] = (output_channels,)

    def add_csp(prefix: str) -> None:
        add_conv(f"{prefix}.short_conv", 512, 128, 1)
        add_conv(f"{prefix}.main_conv", 512, 128, 1)
        for index in range(3):
            # The checkpoint's MMDetection CSPNeXt bottleneck keeps the full
            # 128-channel CSP width: a 1x1 projection followed by a 3x3
            # convolution.  Do not derive this from sam2.utils.csp_layer; that
            # is a different, unused CSPNeXt implementation in the source
            # package.
            add_conv(f"{prefix}.blocks.{index}.conv1", 128, 128, 1)
            add_conv(f"{prefix}.blocks.{index}.conv2", 128, 128, 3)
        add_conv(f"{prefix}.final_conv", 256, 256, 1)

    add_conv(f"{PAFPN_PREFIX}.reduce_layers.2", 256, 256, 1)
    add_csp(f"{PAFPN_PREFIX}.top_down_layers.0.0")
    add_conv(f"{PAFPN_PREFIX}.top_down_layers.0.1", 256, 256, 1)
    add_csp(f"{PAFPN_PREFIX}.top_down_layers.1")
    add_conv(f"{PAFPN_PREFIX}.downsample_layers.0", 256, 256, 3)
    add_csp(f"{PAFPN_PREFIX}.bottom_up_layers.0")
    add_conv(f"{PAFPN_PREFIX}.downsample_layers.1", 256, 256, 3)
    add_csp(f"{PAFPN_PREFIX}.bottom_up_layers.1")
    for index in range(3):
        add_conv(f"{PAFPN_PREFIX}.out_layers.{index}", 256, 256, 3)
    return shapes


PAFPN_PARAMETER_SHAPES: Mapping[str, tuple[int, ...]] = MappingProxyType(_pafpn_parameter_shapes())


def _build_phase_a_pafpn_specs_unchecked(
    weights: Mapping[str, np.ndarray],
) -> tuple[PafpnLeafSpec, ...]:
    builder = _SpecBuilder(weights)
    fpn_0 = ValueRef.external("fpn_input_0", EXTERNAL_INPUT_ABIS["fpn_input_0"])
    fpn_1 = ValueRef.external("fpn_input_1", EXTERNAL_INPUT_ABIS["fpn_input_1"])
    fpn_2 = ValueRef.external("fpn_input_2", EXTERNAL_INPUT_ABIS["fpn_input_2"])

    reduced = builder.conv_bn_silu("Reduce", f"{PAFPN_PREFIX}.reduce_layers.2", fpn_2)
    high = builder.resize_concatenate("TD0", f"{PAFPN_PREFIX}.td0_resize_concat", reduced, fpn_1)
    high = builder.csp("TD0", f"{PAFPN_PREFIX}.top_down_layers.0.0", high)
    inner_1 = builder.conv_bn_silu("TD0", f"{PAFPN_PREFIX}.top_down_layers.0.1", high)
    high = builder.resize_concatenate(
        "TD1",
        f"{PAFPN_PREFIX}.td1_resize_concat",
        inner_1,
        fpn_0,
        force_fp32=True,
    )
    inner_0 = builder.csp("TD1", f"{PAFPN_PREFIX}.top_down_layers.1", high)
    down = builder.conv_bn_silu(
        "BU0",
        f"{PAFPN_PREFIX}.downsample_layers.0",
        inner_0,
        stride=(2, 2),
        padding=(1, 1),
    )
    out_1 = builder.csp(
        "BU0",
        f"{PAFPN_PREFIX}.bottom_up_layers.0",
        builder.concatenate("BU0", f"{PAFPN_PREFIX}.bu0_concat", down, inner_1),
    )
    down = builder.conv_bn_silu(
        "BU1",
        f"{PAFPN_PREFIX}.downsample_layers.1",
        out_1,
        stride=(2, 2),
        padding=(1, 1),
    )
    out_2 = builder.csp(
        "BU1",
        f"{PAFPN_PREFIX}.bottom_up_layers.1",
        builder.concatenate("BU1", f"{PAFPN_PREFIX}.bu1_concat", down, reduced),
    )
    for index, source in enumerate((inner_0, out_1, out_2)):
        builder.conv_bn_silu(
            "Out",
            f"{PAFPN_PREFIX}.out_layers.{index}",
            source,
            padding=(1, 1),
        )
    return tuple(builder.specs)


@lru_cache(maxsize=1)
def _canonical_phase_a_pafpn_specs() -> tuple[PafpnLeafSpec, ...]:
    scalar = np.asarray(0.0, dtype=np.float32)
    weights = {
        name: np.broadcast_to(scalar, shape) for name, shape in PAFPN_PARAMETER_SHAPES.items()
    }
    return _build_phase_a_pafpn_specs_unchecked(weights)


def build_phase_a_pafpn_specs(
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "bf16",
) -> tuple[PafpnLeafSpec, ...]:
    if precision != "bf16":
        raise ValueError("SAM2 HOI Phase-A PAFPN is qualified only for bf16")
    for name, expected_shape in PAFPN_PARAMETER_SHAPES.items():
        try:
            observed_shape = tuple(int(value) for value in np.asarray(weights[name]).shape)
        except KeyError as error:
            raise ValueError(f"SAM2 HOI Phase-A is missing parameter {name}") from error
        if observed_shape != expected_shape:
            raise ValueError(
                f"SAM2 HOI Phase-A parameter shape drift for {name}: "
                f"expected {expected_shape}, got {observed_shape}"
            )
    specs = _build_phase_a_pafpn_specs_unchecked(weights)
    validate_phase_a_pafpn_specs(specs)
    return specs


def validate_phase_a_pafpn_specs(specs: tuple[PafpnLeafSpec, ...]) -> None:
    if len(specs) != PAFPN_PLAN_COUNT:
        raise ValueError(f"SAM2 HOI Phase-A requires 137 leaf specs, got {len(specs)}")
    observed_counts = {stage: 0 for stage in STAGE_COUNTS}
    expected_stages = tuple(
        stage for stage, count in STAGE_COUNTS.items() for _index in range(count)
    )
    external_roots: dict[str, list[int]] = {name: [] for name in EXTERNAL_INPUTS}
    node_ids: set[str] = set()
    live = [False] * len(specs)
    for ordinal, spec in enumerate(specs):
        if spec.ordinal != ordinal or spec.section != phase_a_plan_section(ordinal):
            raise ValueError(f"SAM2 HOI Phase-A ordinal/section drift at {ordinal}")
        if (
            spec.stage != expected_stages[ordinal]
            or not spec.node_id.startswith(f"{spec.stage}::")
            or spec.node_id in node_ids
        ):
            raise ValueError(f"SAM2 HOI Phase-A stage drift at {ordinal}")
        node_ids.add(spec.node_id)
        observed_counts[spec.stage] += 1
        bindings = [item.binding for item in spec.inputs]
        if not bindings or len(bindings) != len(set(bindings)):
            raise ValueError(f"SAM2 HOI Phase-A input binding drift at {ordinal}")
        for item in spec.inputs:
            source = item.value
            if source.source_kind == "external":
                if source.source_name not in external_roots:
                    raise ValueError(f"SAM2 HOI Phase-A unknown external root at {ordinal}")
                if source.abi != EXTERNAL_INPUT_ABIS[source.source_name]:
                    raise ValueError(f"SAM2 HOI Phase-A external ABI drift at {ordinal}")
                external_roots[source.source_name].append(ordinal)
            elif source.source_kind == "node":
                if (
                    not isinstance(source.source_node, int)
                    or isinstance(source.source_node, bool)
                    or source.source_node < 0
                    or source.source_node >= ordinal
                ):
                    raise ValueError(f"SAM2 HOI Phase-A forward edge at {ordinal}")
                if specs[source.source_node].output != source.abi:
                    raise ValueError(f"SAM2 HOI Phase-A ABI edge drift at {ordinal}")
            else:
                raise ValueError(f"SAM2 HOI Phase-A source kind drift at {ordinal}")
    if observed_counts != STAGE_COUNTS:
        raise ValueError(f"SAM2 HOI Phase-A stage partition drift: {observed_counts}")
    if {name: rows for name, rows in external_roots.items()} != {
        "fpn_input_0": [35],
        "fpn_input_1": [3],
        "fpn_input_2": [0],
    }:
        raise ValueError(f"SAM2 HOI Phase-A root drift: {external_roots}")
    if tuple(specs[ordinal].output for _, ordinal in PUBLIC_OUTPUTS) != PUBLIC_OUTPUT_ABIS:
        raise ValueError("SAM2 HOI Phase-A public output ABI drift")
    for _, ordinal in PUBLIC_OUTPUTS:
        live[ordinal] = True
    for ordinal in range(len(specs) - 1, -1, -1):
        if not live[ordinal]:
            continue
        for item in specs[ordinal].inputs:
            if item.value.source_kind == "node":
                live[int(item.value.source_node)] = True
    if not all(live):
        raise ValueError("SAM2 HOI Phase-A graph contains a dead leaf")
    for ordinal, (spec, expected) in enumerate(
        zip(specs, _canonical_phase_a_pafpn_specs(), strict=True)
    ):
        if spec != expected:
            raise ValueError(f"SAM2 HOI Phase-A exact leaf contract drift at {ordinal}")


def phase_a_pafpn_manifest(
    specs: tuple[PafpnLeafSpec, ...],
    plan_sha256: Mapping[int, str],
) -> dict[str, object]:
    validate_phase_a_pafpn_specs(specs)
    if set(plan_sha256) != set(range(PAFPN_PLAN_COUNT)):
        raise ValueError("SAM2 HOI Phase-A plan SHA inventory is incomplete")
    nodes = []
    for spec in specs:
        digest = plan_sha256[spec.ordinal]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"SAM2 HOI Phase-A invalid plan SHA at {spec.ordinal}")
        nodes.append(
            {
                "ordinal": spec.ordinal,
                "id": spec.node_id,
                "section": spec.section,
                "plan_sha256": digest,
                "inputs": [
                    {"tensor": item.binding, "source": item.value.manifest_source()}
                    for item in spec.inputs
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "external_inputs": list(EXTERNAL_INPUTS),
        "nodes": nodes,
        "outputs": [
            {"name": name, "source": {"node": ordinal, "tensor": "output"}}
            for name, ordinal in PUBLIC_OUTPUTS
        ],
    }
    validate_phase_a_pafpn_manifest(manifest, specs)
    return manifest


def validate_phase_a_pafpn_manifest(
    manifest: Mapping[str, object],
    specs: tuple[PafpnLeafSpec, ...],
) -> None:
    validate_phase_a_pafpn_specs(specs)
    if set(manifest) != {"schema_version", "external_inputs", "nodes", "outputs"}:
        raise ValueError("SAM2 HOI Phase-A manifest root schema drift")
    if manifest.get("schema_version") != 1 or manifest.get("external_inputs") != list(
        EXTERNAL_INPUTS
    ):
        raise ValueError("SAM2 HOI Phase-A manifest identity drift")
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != len(specs):
        raise ValueError("SAM2 HOI Phase-A manifest node inventory drift")
    for spec, node in zip(specs, nodes, strict=True):
        if not isinstance(node, dict) or set(node) != {
            "ordinal",
            "id",
            "section",
            "plan_sha256",
            "inputs",
        }:
            raise ValueError(f"SAM2 HOI Phase-A manifest node schema drift at {spec.ordinal}")
        expected_inputs = [
            {"tensor": item.binding, "source": item.value.manifest_source()} for item in spec.inputs
        ]
        if (
            node.get("ordinal") != spec.ordinal
            or node.get("id") != spec.node_id
            or node.get("section") != spec.section
            or node.get("inputs") != expected_inputs
        ):
            raise ValueError(f"SAM2 HOI Phase-A manifest topology drift at {spec.ordinal}")
        digest = node.get("plan_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"SAM2 HOI Phase-A manifest plan SHA drift at {spec.ordinal}")
    expected_outputs = [
        {"name": name, "source": {"node": ordinal, "tensor": "output"}}
        for name, ordinal in PUBLIC_OUTPUTS
    ]
    if manifest.get("outputs") != expected_outputs:
        raise ValueError("SAM2 HOI Phase-A manifest public output drift")


def _trt_dtype(trt: Any, dtype: str):
    if dtype == "float32":
        return trt.float32
    if dtype == "bfloat16":
        return trt.bfloat16
    raise ValueError(f"Unsupported SAM2 HOI Phase-A dtype: {dtype}")


def _leaf_builder_config_receipt(config: Any, trt: Any, spec: PafpnLeafSpec) -> dict[str, object]:
    optimization_level = 0 if spec.ordinal in PAFPN_OPT0_CONV_ORDINALS else 3
    if optimization_level == 0 and spec.kind != "conv":
        raise RuntimeError(f"SAM2 HOI Phase-A opt0 policy reached non-conv leaf {spec.ordinal}")
    required = (
        "builder_optimization_level",
        "avg_timing_iterations",
        "max_aux_streams",
        "max_num_tactics",
        "profiling_verbosity",
    )
    missing = [name for name in required if not hasattr(config, name)]
    if missing or not hasattr(trt, "ProfilingVerbosity"):
        raise RuntimeError(f"SAM2 HOI Phase-A builder config API drift: {missing}")
    config.builder_optimization_level = optimization_level
    config.avg_timing_iterations = 1
    config.max_aux_streams = 0
    config.max_num_tactics = -1
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    timing_cache = config.get_timing_cache() if hasattr(config, "get_timing_cache") else None
    if timing_cache is not None:
        raise RuntimeError("SAM2 HOI Phase-A timing cache must remain disabled")
    enabled_flags = []
    if hasattr(config, "get_flag") and hasattr(trt, "BuilderFlag"):
        for name in (
            "DEBUG",
            "EDITABLE_TIMING_CACHE",
            "DISABLE_COMPILATION_CACHE",
            "ERROR_ON_TIMING_CACHE_MISS",
        ):
            flag = getattr(trt.BuilderFlag, name, None)
            if flag is not None and config.get_flag(flag):
                enabled_flags.append(name)
    if enabled_flags:
        raise RuntimeError(f"SAM2 HOI Phase-A forbidden builder flags enabled: {enabled_flags}")
    return {
        "ordinal": spec.ordinal,
        "kind": spec.kind,
        "builder_optimization_level": optimization_level,
        "avg_timing_iterations": 1,
        "max_aux_streams": 0,
        "max_num_tactics": -1,
        "profiling_verbosity": "DETAILED",
        "timing_cache": None,
        "enabled_builder_flags": [],
        "source_exact_tactic_claimed": False,
    }


def _validate_leaf_builder_config(
    config: Any,
    trt: Any,
    expected: Mapping[str, object],
) -> None:
    observed = {
        "builder_optimization_level": int(config.builder_optimization_level),
        "avg_timing_iterations": int(config.avg_timing_iterations),
        "max_aux_streams": int(config.max_aux_streams),
        "max_num_tactics": int(config.max_num_tactics),
        "profiling_verbosity": str(config.profiling_verbosity).rsplit(".", 1)[-1],
        "timing_cache": config.get_timing_cache() if hasattr(config, "get_timing_cache") else None,
    }
    for key in (
        "builder_optimization_level",
        "avg_timing_iterations",
        "max_aux_streams",
        "max_num_tactics",
        "profiling_verbosity",
        "timing_cache",
    ):
        if observed[key] != expected[key]:
            raise RuntimeError(f"SAM2 HOI Phase-A builder config changed at {key}")
    if hasattr(config, "get_flag") and hasattr(trt, "BuilderFlag"):
        for name in (
            "DEBUG",
            "EDITABLE_TIMING_CACHE",
            "DISABLE_COMPILATION_CACHE",
            "ERROR_ON_TIMING_CACHE_MISS",
        ):
            flag = getattr(trt.BuilderFlag, name, None)
            if flag is not None and config.get_flag(flag):
                raise RuntimeError(f"SAM2 HOI Phase-A forbidden builder flag enabled: {name}")


def _build_leaf_plan(
    spec: PafpnLeafSpec,
    weights: Mapping[str, np.ndarray],
    *,
    verbose: bool,
) -> bytes:
    trt = graph_ops._trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config_receipt = _leaf_builder_config_receipt(config, trt, spec)
    inputs = {}
    for item in spec.inputs:
        tensor = network.add_input(
            item.binding, _trt_dtype(trt, item.value.abi.dtype), item.value.abi.shape
        )
        if tensor is None:
            raise RuntimeError(f"Could not create SAM2 HOI Phase-A input {spec.ordinal}")
        inputs[item.binding] = tensor

    graph_ops.reset_weight_refs()
    try:
        first_layer = int(network.num_layers)
        if spec.kind == "conv":
            prefix = str(spec.module_prefix)
            convolution = image_builder._weight(weights, f"{prefix}.conv.weight")
            output_channels = int(convolution.shape[0])
            bias = image_builder._optional_weight(
                weights,
                f"{prefix}.conv.bias",
                (output_channels,),
            )
            output = graph_ops.add_conv2d(
                network,
                graph_ops.cast(network, inputs["input"], trt.bfloat16),
                convolution,
                bias,
                stride=spec.stride,
                padding=spec.padding,
                precision="bf16",
            )
        elif spec.kind == "batch_norm":
            prefix = str(spec.module_prefix)
            running_variance = image_builder._weight(weights, f"{prefix}.bn.running_var")
            invstd = pafpn_bn_invstd.compute_invstd(
                running_variance,
                epsilon=1.0e-5,
                verbose=verbose,
            )
            output = graph_ops.add_batch_norm2d_affine_from_invstd(
                network,
                inputs["input"],
                image_builder._weight(weights, f"{prefix}.bn.weight"),
                image_builder._weight(weights, f"{prefix}.bn.bias"),
                image_builder._weight(weights, f"{prefix}.bn.running_mean"),
                invstd,
                output_dtype=trt.bfloat16,
            )
        elif spec.kind == "silu":
            output = graph_ops.add_activation(network, inputs["input"], "silu")
        elif spec.kind == "concat":
            output = image_builder._concatenate(
                network,
                [inputs["input_0"], inputs["input_1"]],
                axis=1,
            )
        elif spec.kind == "resize_concat":
            source_shape = spec.inputs[0].value.abi.shape
            resized = graph_ops.add_resize(
                network,
                inputs["input_0"],
                (source_shape[0], source_shape[1], spec.output.shape[2], spec.output.shape[3]),
                mode="nearest",
                coordinate_transformation="asymmetric",
            )
            if spec.ordinal == 35:
                resized = graph_ops.cast(network, resized, trt.float32)
            output = image_builder._concatenate(
                network,
                [resized, inputs["input_1"]],
                axis=1,
            )
        else:
            raise ValueError(f"Unsupported SAM2 HOI Phase-A leaf kind: {spec.kind}")
        for layer_index in range(first_layer, int(network.num_layers)):
            layer = network.get_layer(layer_index)
            layer.name = (
                f"sam2_hoi_phase_a::{spec.ordinal:03d}::{spec.kind}::"
                f"{layer_index - first_layer:02d}"
            )
        if tuple(int(value) for value in output.shape) != spec.output.shape:
            raise RuntimeError(f"SAM2 HOI Phase-A output shape drift at {spec.ordinal}")
        output_dtype = _trt_dtype(trt, spec.output.dtype)
        if output.dtype != output_dtype:
            raise RuntimeError(f"SAM2 HOI Phase-A output dtype drift at {spec.ordinal}")
        graph_ops.mark_output(network, output, "output", dtype=output_dtype)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"SAM2 HOI Phase-A leaf build failed at {spec.ordinal}")
        _validate_leaf_builder_config(config, trt, config_receipt)
        return bytes(plan)
    finally:
        graph_ops.reset_weight_refs()


def build_phase_a_pafpn_file_sections(
    weights: Mapping[str, np.ndarray],
    *,
    staging_dir: Path,
    precision: str = "bf16",
    verbose: bool = False,
) -> list[FileBundleSection]:
    staging_dir = Path(staging_dir)
    if not staging_dir.is_dir() or staging_dir.is_symlink():
        raise ValueError("SAM2 HOI Phase-A staging_dir must be an existing regular directory")
    specs = build_phase_a_pafpn_specs(weights, precision=precision)
    # Compile and authenticate the builder-only helper before serializing any
    # leaf plan, so a toolchain or CUDA-helper failure leaves no partial PAFPN.
    pafpn_bn_invstd.helper_build_receipt(verbose=verbose)
    plan_sections: list[FileBundleSection] = []
    plan_shas: dict[int, str] = {}
    for spec in specs:
        payload = _build_leaf_plan(spec, weights, verbose=verbose)
        path = staging_dir / f"{spec.section}.plan"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        plan_shas[spec.ordinal] = digest
        plan_sections.append(
            bundle_section_from_file(
                spec.section,
                path,
                expected_size=len(payload),
                expected_sha256=digest,
            )
        )
    manifest = phase_a_pafpn_manifest(specs, plan_shas)
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_path = staging_dir / PAFPN_MANIFEST_SECTION
    manifest_path.write_bytes(manifest_bytes)
    manifest_section = bundle_section_from_file(
        PAFPN_MANIFEST_SECTION,
        manifest_path,
        expected_size=len(manifest_bytes),
        expected_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    return [manifest_section, *plan_sections]


__all__ = [
    "EXTERNAL_INPUTS",
    "EXTERNAL_INPUT_ABIS",
    "PAFPN_MANIFEST_SECTION",
    "PAFPN_PARAMETER_SHAPES",
    "PAFPN_OPT0_CONV_ORDINALS",
    "PAFPN_PLAN_COUNT",
    "PHASE_A_BUILD_OPTION",
    "PHASE_A_CONFIG_KEY",
    "PUBLIC_OUTPUTS",
    "STAGE_COUNTS",
    "PafpnLeafSpec",
    "build_phase_a_pafpn_file_sections",
    "build_phase_a_pafpn_specs",
    "phase_a_bundle_loading",
    "phase_a_pafpn_build_policy",
    "phase_a_pafpn_manifest",
    "phase_a_plan_section",
    "validate_phase_a_pafpn_specs",
    "validate_phase_a_pafpn_manifest",
]
