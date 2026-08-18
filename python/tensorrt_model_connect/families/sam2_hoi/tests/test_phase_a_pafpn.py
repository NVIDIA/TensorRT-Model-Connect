# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tensorrt_model_connect.engine_builder as engine_builder
from tensorrt_model_connect.bundle_writer import BundleSection, FileBundleSection
from tensorrt_model_connect.engine_builder import build_bundle
from tensorrt_model_connect.families.sam2_hoi import phase_a_pafpn
from tensorrt_model_connect.families.sam2_hoi import native_image_builder
from tensorrt_model_connect.families.sam2_hoi import source_export
from tensorrt_model_connect.families.sam2_hoi.plugin import plugin


PREFIX = phase_a_pafpn.PAFPN_PREFIX
PLUGIN_MODULE = importlib.import_module("tensorrt_model_connect.families.sam2_hoi.plugin")


def _array(shape):
    return np.broadcast_to(np.asarray(0.0, dtype=np.float32), shape)


def _conv(weights, prefix, input_channels, output_channels, kernel):
    weights[f"{prefix}.conv.weight"] = _array((output_channels, input_channels, kernel, kernel))
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        weights[f"{prefix}.bn.{suffix}"] = _array((output_channels,))


def _csp(weights, prefix, input_channels=512):
    _conv(weights, f"{prefix}.short_conv", input_channels, 128, 1)
    _conv(weights, f"{prefix}.main_conv", input_channels, 128, 1)
    for index in range(3):
        _conv(weights, f"{prefix}.blocks.{index}.conv1", 128, 128, 1)
        _conv(weights, f"{prefix}.blocks.{index}.conv2", 128, 128, 3)
    _conv(weights, f"{prefix}.final_conv", 256, 256, 1)


def _weights():
    weights = {}
    _conv(weights, f"{PREFIX}.reduce_layers.2", 256, 256, 1)
    _csp(weights, f"{PREFIX}.top_down_layers.0.0")
    _conv(weights, f"{PREFIX}.top_down_layers.0.1", 256, 256, 1)
    _csp(weights, f"{PREFIX}.top_down_layers.1")
    _conv(weights, f"{PREFIX}.downsample_layers.0", 256, 256, 3)
    _csp(weights, f"{PREFIX}.bottom_up_layers.0")
    _conv(weights, f"{PREFIX}.downsample_layers.1", 256, 256, 3)
    _csp(weights, f"{PREFIX}.bottom_up_layers.1")
    for index in range(3):
        _conv(weights, f"{PREFIX}.out_layers.{index}", 256, 256, 3)
    return weights


def _full_spec_contract_rows(specs):
    return [
        {
            "ordinal": spec.ordinal,
            "stage": spec.stage,
            "label": spec.label,
            "kind": spec.kind,
            "inputs": [
                {
                    "binding": item.binding,
                    "source_kind": item.value.source_kind,
                    "source_name": item.value.source_name,
                    "source_node": item.value.source_node,
                    "shape": list(item.value.abi.shape),
                    "dtype": item.value.abi.dtype,
                }
                for item in spec.inputs
            ],
            "output": {"shape": list(spec.output.shape), "dtype": spec.output.dtype},
            "module_prefix": spec.module_prefix,
            "stride": list(spec.stride),
            "padding": list(spec.padding),
        }
        for spec in specs
    ]


def test_phase_a_spec_partition_roots_outputs_and_ordinals():
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    assert len(specs) == 137
    assert [spec.ordinal for spec in specs] == list(range(137))
    assert [spec.section for spec in specs] == [
        f"sam2_hoi_pafpn_plan_{index:03d}" for index in range(137)
    ]
    assert {
        stage: sum(spec.stage == stage for spec in specs) for stage in phase_a_pafpn.STAGE_COUNTS
    } == phase_a_pafpn.STAGE_COUNTS
    roots = {
        item.value.source_name: spec.ordinal
        for spec in specs
        for item in spec.inputs
        if item.value.source_kind == "external"
    }
    assert roots == {"fpn_input_2": 0, "fpn_input_1": 3, "fpn_input_0": 35}
    assert specs[3].kind == specs[35].kind == "resize_concat"
    assert [(name, ordinal) for name, ordinal in phase_a_pafpn.PUBLIC_OUTPUTS] == [
        ("detector_feature_0", 130),
        ("detector_feature_1", 133),
        ("detector_feature_2", 136),
    ]
    rows = _full_spec_contract_rows(specs)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    full_spec_contract_sha256 = hashlib.sha256(encoded).hexdigest()
    assert len(rows) == 137
    assert full_spec_contract_sha256 == (
        "c2356958a6d3e6b7ec002344af40e4e0e369a20be2c167a5cf4290cf007e616d"
    )
    assert specs[35].output.dtype == "float32"
    assert specs[36].inputs[0].value.abi.dtype == "float32"
    assert specs[39].inputs[0].value.abi.dtype == "float32"


def test_phase_a_checkpoint_shape_fix_changes_only_exact_csp_leaf_contracts():
    corrected = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    legacy_weights = _weights()
    for csp in (
        "top_down_layers.0.0",
        "top_down_layers.1",
        "bottom_up_layers.0",
        "bottom_up_layers.1",
    ):
        for index in range(3):
            block = f"{PREFIX}.{csp}.blocks.{index}"
            legacy_weights[f"{block}.conv1.conv.weight"] = _array((64, 128, 1, 1))
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                legacy_weights[f"{block}.conv1.bn.{suffix}"] = _array((64,))
            legacy_weights[f"{block}.conv2.conv.weight"] = _array((128, 64, 3, 3))
    legacy = phase_a_pafpn._build_phase_a_pafpn_specs_unchecked(legacy_weights)
    changed_ordinals = [
        index
        for index, (before, after) in enumerate(zip(legacy, corrected, strict=True))
        if before != after
    ]
    assert changed_ordinals == [
        10,
        11,
        12,
        13,
        16,
        17,
        18,
        19,
        22,
        23,
        24,
        25,
        42,
        43,
        44,
        45,
        48,
        49,
        50,
        51,
        54,
        55,
        56,
        57,
        74,
        75,
        76,
        77,
        80,
        81,
        82,
        83,
        86,
        87,
        88,
        89,
        106,
        107,
        108,
        109,
        112,
        113,
        114,
        115,
        118,
        119,
        120,
        121,
    ]
    encoded = json.dumps(changed_ordinals, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "4a8e7bcaed827f012e999ebea80d82a1d8236a4955273ac040b24813d3b35905"
    )


def test_phase_a_csp_and_stage_boundary_mapping():
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    assert specs[0].label.endswith("reduce_layers.2.conv")
    assert specs[2].label.endswith("reduce_layers.2.silu")
    assert specs[4].label.endswith("top_down_layers.0.0.short_conv.conv")
    assert specs[28].label.endswith("top_down_layers.0.0.concat")
    assert specs[31].label.endswith("top_down_layers.0.0.final_conv.silu")
    assert specs[32].label.endswith("top_down_layers.0.1.conv")
    assert specs[64].label.endswith("downsample_layers.0.conv")
    assert specs[67].label.endswith("bu0_concat")
    assert specs[96].label.endswith("downsample_layers.1.conv")
    assert specs[99].label.endswith("bu1_concat")
    assert specs[130].label.endswith("out_layers.0.silu")
    assert specs[133].label.endswith("out_layers.1.silu")
    assert specs[136].label.endswith("out_layers.2.silu")


def test_phase_a_manifest_is_runtime_schema_one_and_exact():
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    shas = {index: hashlib.sha256(str(index).encode()).hexdigest() for index in range(137)}
    manifest = phase_a_pafpn.phase_a_pafpn_manifest(specs, shas)
    assert set(manifest) == {"schema_version", "external_inputs", "nodes", "outputs"}
    assert manifest["schema_version"] == 1
    assert manifest["external_inputs"] == ["fpn_input_0", "fpn_input_1", "fpn_input_2"]
    assert len(manifest["nodes"]) == 137
    assert manifest["nodes"][3]["inputs"] == [
        {"tensor": "input_0", "source": {"node": 2, "tensor": "output"}},
        {"tensor": "input_1", "source": {"external": "fpn_input_1"}},
    ]
    assert manifest["outputs"] == [
        {"name": name, "source": {"node": ordinal, "tensor": "output"}}
        for name, ordinal in phase_a_pafpn.PUBLIC_OUTPUTS
    ]
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "b5194aa6b5c9e8013a07ba35da6ee41dfff36c0567cf13858b8e1f5c871bb073"
    )


def test_phase_a_parameter_shapes_match_authoritative_checkpoint_contract():
    # Exact parameter-shape fixture extracted from checkpoint
    # 88849a8268a38ba66061093f90866af1d033d05d0f1de865534bf490e9880292.
    # It covers the 43 ConvBNAct modules (215 consumed tensors) underlying
    # 129 ConvBNAct leaves / 86 parameter-consuming leaves in the exact
    # 137-leaf graph.
    shapes = dict(phase_a_pafpn.PAFPN_PARAMETER_SHAPES)
    encoded = json.dumps(
        {name: list(shape) for name, shape in sorted(shapes.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert len(shapes) == 215
    assert hashlib.sha256(encoded).hexdigest() == (
        "7593d0fb27f7914be95737a2e12cdaade8af37f7a229a3bc16f79821402b1504"
    )

    conv_shapes = [shape for name, shape in shapes.items() if name.endswith(".conv.weight")]
    assert len(conv_shapes) == 43
    assert {shape: conv_shapes.count(shape) for shape in set(conv_shapes)} == {
        (128, 128, 1, 1): 12,
        (128, 128, 3, 3): 12,
        (128, 512, 1, 1): 8,
        (256, 256, 1, 1): 6,
        (256, 256, 3, 3): 5,
    }
    for csp in (
        "top_down_layers.0.0",
        "top_down_layers.1",
        "bottom_up_layers.0",
        "bottom_up_layers.1",
    ):
        for index in range(3):
            block = f"{PREFIX}.{csp}.blocks.{index}"
            assert shapes[f"{block}.conv1.conv.weight"] == (128, 128, 1, 1)
            assert shapes[f"{block}.conv2.conv.weight"] == (128, 128, 3, 3)


def test_phase_a_all_137_leaves_have_exact_parameter_coverage():
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    assert {
        kind: sum(spec.kind == kind for spec in specs) for kind in set(s.kind for s in specs)
    } == {
        "conv": 43,
        "batch_norm": 43,
        "silu": 43,
        "concat": 6,
        "resize_concat": 2,
    }
    shapes = phase_a_pafpn.PAFPN_PARAMETER_SHAPES
    for spec in specs:
        if spec.kind == "conv":
            assert f"{spec.module_prefix}.conv.weight" in shapes
        elif spec.kind == "batch_norm":
            assert all(
                f"{spec.module_prefix}.bn.{suffix}" in shapes
                for suffix in ("weight", "bias", "running_mean", "running_var")
            )


@pytest.mark.parametrize("mutation", ["extra", "edge", "output", "sha"])
def test_phase_a_manifest_validator_rejects_schema_topology_and_sha_drift(mutation):
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    shas = {index: hashlib.sha256(str(index).encode()).hexdigest() for index in range(137)}
    manifest = phase_a_pafpn.phase_a_pafpn_manifest(specs, shas)
    broken = copy.deepcopy(manifest)
    if mutation == "extra":
        broken["extra"] = True
    elif mutation == "edge":
        broken["nodes"][3]["inputs"][1]["source"] = {"external": "fpn_input_0"}
    elif mutation == "output":
        broken["outputs"][0]["source"]["node"] = 129
    else:
        broken["nodes"][0]["plan_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="manifest"):
        phase_a_pafpn.validate_phase_a_pafpn_manifest(broken, specs)


def test_phase_a_spec_validator_rejects_forward_abi_and_dead_edges():
    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    original = specs[10].inputs[0]
    specs[10] = replace(
        specs[10],
        inputs=(
            replace(
                original,
                value=phase_a_pafpn.ValueRef.node(11, original.value.abi),
            ),
        ),
    )
    with pytest.raises(ValueError, match="forward edge"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))

    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    input_136 = specs[136].inputs[0]
    specs[136] = replace(
        specs[136],
        inputs=(replace(input_136, value=phase_a_pafpn.ValueRef.node(134, input_136.value.abi)),),
    )
    with pytest.raises(ValueError, match="dead leaf"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))

    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    input_20 = specs[20].inputs[0]
    wrong = phase_a_pafpn.TensorABI(input_20.value.abi.shape, "float32")
    specs[20] = replace(
        specs[20],
        inputs=(replace(input_20, value=phase_a_pafpn.ValueRef.node(19, wrong)),),
    )
    with pytest.raises(ValueError, match="ABI edge drift"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))


def test_phase_a_spec_validator_rejects_external_abi_duplicate_ids_and_source_kind():
    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    root = specs[0].inputs[0]
    wrong_abi = phase_a_pafpn.TensorABI(root.value.abi.shape, "float32")
    specs[0] = replace(
        specs[0],
        inputs=(replace(root, value=phase_a_pafpn.ValueRef.external("fpn_input_2", wrong_abi)),),
    )
    with pytest.raises(ValueError, match="external ABI drift"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))

    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    specs[1] = replace(specs[1], label=specs[0].label)
    with pytest.raises(ValueError, match="stage drift"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))

    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    root = specs[0].inputs[0]
    unknown = phase_a_pafpn.ValueRef("unknown", None, None, root.value.abi)
    specs[0] = replace(specs[0], inputs=(replace(root, value=unknown),))
    with pytest.raises(ValueError, match="source kind drift"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))


def test_phase_a_spec_validator_rejects_semantic_edge_and_consistent_abi_drift():
    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    concat = specs[67]
    specs[67] = replace(
        concat,
        inputs=(
            replace(concat.inputs[0], value=concat.inputs[1].value),
            replace(concat.inputs[1], value=concat.inputs[0].value),
        ),
    )
    with pytest.raises(ValueError, match="exact leaf contract drift at 67"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))

    specs = list(phase_a_pafpn.build_phase_a_pafpn_specs(_weights()))
    drifted = phase_a_pafpn.TensorABI(specs[0].output.shape, "float32")
    specs[0] = replace(specs[0], output=drifted)
    specs[1] = replace(
        specs[1],
        inputs=(
            replace(
                specs[1].inputs[0],
                value=phase_a_pafpn.ValueRef.node(0, drifted),
            ),
        ),
    )
    with pytest.raises(ValueError, match="exact leaf contract drift at 0"):
        phase_a_pafpn.validate_phase_a_pafpn_specs(tuple(specs))


def test_phase_a_parameter_contract_rejects_missing_and_wrong_shapes():
    missing = _weights()
    missing.pop(f"{PREFIX}.out_layers.2.bn.bias")
    with pytest.raises(ValueError, match="missing parameter"):
        phase_a_pafpn.build_phase_a_pafpn_specs(missing)

    wrong = _weights()
    wrong[f"{PREFIX}.reduce_layers.2.conv.weight"] = _array((255, 256, 1, 1))
    with pytest.raises(ValueError, match="parameter shape drift"):
        phase_a_pafpn.build_phase_a_pafpn_specs(wrong)

    old_csp_assumption = _weights()
    old_csp_assumption[f"{PREFIX}.top_down_layers.0.0.blocks.0.conv1.conv.weight"] = _array(
        (64, 128, 1, 1)
    )
    with pytest.raises(ValueError, match=r"expected \(128, 128, 1, 1\), got \(64, 128, 1, 1\)"):
        phase_a_pafpn.build_phase_a_pafpn_specs(old_csp_assumption)


def test_phase_a_file_hook_builds_manifest_and_137_regular_plans(monkeypatch, tmp_path):
    helper_calls = []
    monkeypatch.setattr(
        phase_a_pafpn.pafpn_bn_invstd,
        "helper_build_receipt",
        lambda *, verbose: helper_calls.append(verbose) or {"result": "PASS"},
    )
    monkeypatch.setattr(
        phase_a_pafpn,
        "_build_leaf_plan",
        lambda spec, _weights, *, verbose: f"plan-{spec.ordinal}-{verbose}".encode(),
    )
    sections = phase_a_pafpn.build_phase_a_pafpn_file_sections(
        _weights(), staging_dir=tmp_path, verbose=True
    )
    assert helper_calls == [True]
    assert len(sections) == 138
    assert all(isinstance(section, FileBundleSection) for section in sections)
    assert [section.name for section in sections[1:]] == [
        f"sam2_hoi_pafpn_plan_{index:03d}" for index in range(137)
    ]
    assert all(
        section.source_path.is_file() and not section.source_path.is_symlink()
        for section in sections
    )
    manifest = json.loads(sections[0].source_path.read_text(encoding="utf-8"))
    assert len(manifest["nodes"]) == 137
    for node, section in zip(manifest["nodes"], sections[1:], strict=True):
        assert node["section"] == section.name
        assert node["plan_sha256"] == section.expected_sha256
    for section in sections:
        payload = section.source_path.read_bytes()
        assert section.expected_size == len(payload)
        assert section.expected_sha256 == hashlib.sha256(payload).hexdigest()


def test_phase_a_bundle_loading_is_exact_146_section_partition():
    policy = phase_a_pafpn.phase_a_bundle_loading()
    assert policy["mode"] == "staged"
    assert policy["eager_sections"] == [
        "config.json",
        "sam2_hoi_pafpn_manifest.json",
        "sam2_hoi_native_plugin_so",
    ]
    assert len(policy["lazy_sections"]) == 143
    assert len(policy["eager_sections"]) + len(policy["lazy_sections"]) == 146
    assert len(set(policy["eager_sections"]) | set(policy["lazy_sections"])) == 146
    assert not any(
        "invstd" in name for name in (*policy["eager_sections"], *policy["lazy_sections"])
    )


def test_phase_a_leaf_builder_policy_is_exact_but_tactics_are_l4_external_gate():
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())
    assert phase_a_pafpn.PAFPN_OPT0_CONV_ORDINALS == {
        13,
        19,
        25,
        45,
        51,
        57,
        64,
        77,
        83,
        89,
        128,
        134,
    }
    assert all(specs[ordinal].kind == "conv" for ordinal in phase_a_pafpn.PAFPN_OPT0_CONV_ORDINALS)
    assert [specs[ordinal].stage for ordinal in sorted(phase_a_pafpn.PAFPN_OPT0_CONV_ORDINALS)] == [
        "TD0",
        "TD0",
        "TD0",
        "TD1",
        "TD1",
        "TD1",
        "BU0",
        "BU0",
        "BU0",
        "BU0",
        "Out",
        "Out",
    ]
    policy = phase_a_pafpn.phase_a_pafpn_build_policy()
    assert policy["optimization"] == {
        "opt0_conv_ordinals": sorted(phase_a_pafpn.PAFPN_OPT0_CONV_ORDINALS),
        "all_other_leaf_optimization_level": 3,
        "avg_timing_iterations": 1,
        "timing_cache": "disabled",
        "profiling_verbosity": "DETAILED",
    }
    assert policy["batch_norm"]["helper_source_sha256"] == (
        "4d0fad825f75412c968764ed2baade5c652963dd956db385c4eed3ce932089c0"
    )
    assert policy["batch_norm"]["inference_runtime_launch_added"] is False
    assert policy["batch_norm"]["source_operation_order"] == "CastSubMulMulAddCast"
    assert policy["batch_norm"]["inspector_fusion"] == "CastAddMulMulAddCast"
    assert policy["batch_norm"]["host_affine_folding"] is False
    assert policy["silu"] == {
        "opmath": "fp32",
        "implementation": "native_exp_sum_div_then_output_dtype_cast",
        "plugin_used": False,
    }
    assert policy["td1_seam"] == {
        "ordinal": 35,
        "operation": "second_nearest_resize_output_cast_to_fp32_before_concat",
        "output_dtype": "float32",
    }
    assert policy["qualification"] == {
        "generic_build_source_exact_tactics_claimed": False,
        "product_gate": "sam2-hoi-full-chain-accuracy-v2",
        "l4_per_leaf_source_exact_requires_external_137_plan_gate": True,
    }

    builder_source = inspect.getsource(phase_a_pafpn._build_leaf_plan)
    config_source = inspect.getsource(phase_a_pafpn._leaf_builder_config_receipt)
    assert "0 if spec.ordinal in PAFPN_OPT0_CONV_ORDINALS else 3" in config_source
    assert "config.avg_timing_iterations = 1" in config_source
    assert "config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED" in config_source
    assert "add_batch_norm2d_affine_from_invstd" in builder_source
    assert "pafpn_bn_invstd.compute_invstd" in builder_source
    assert "sam2_hoi_phase_a::" in builder_source
    assert "if spec.ordinal == 35:" in builder_source
    assert builder_source.index("resized = graph_ops.add_resize") < builder_source.index(
        "if spec.ordinal == 35:"
    )

    legacy_rows = _full_spec_contract_rows(specs)
    legacy_rows[35]["output"]["dtype"] = "bfloat16"
    legacy_rows[36]["inputs"][0]["dtype"] = "bfloat16"
    legacy_rows[39]["inputs"][0]["dtype"] = "bfloat16"
    current_rows = _full_spec_contract_rows(specs)
    seam_changed_ordinals = [
        index
        for index, (before, after) in enumerate(zip(legacy_rows, current_rows, strict=True))
        if before != after
    ]
    assert seam_changed_ordinals == [35, 36, 39]
    encoded = json.dumps(seam_changed_ordinals, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "a033321f597ecbb2d716ce8ac8dd8b1f6360a1aa6402bb650487b71fd80cd4c2"
    )


def test_phase_a_leaf_config_applies_opt0_only_to_allowlisted_conv_and_stays_stable():
    specs = phase_a_pafpn.build_phase_a_pafpn_specs(_weights())

    class Config:
        builder_optimization_level = -1
        avg_timing_iterations = -1
        max_aux_streams = -1
        max_num_tactics = 0
        profiling_verbosity = "NONE"

        def __init__(self):
            self.timing_cache = None

        def get_timing_cache(self):
            return self.timing_cache

        def get_flag(self, _flag):
            return False

    trt = SimpleNamespace(
        ProfilingVerbosity=SimpleNamespace(DETAILED="DETAILED"),
        BuilderFlag=SimpleNamespace(
            DEBUG="DEBUG",
            EDITABLE_TIMING_CACHE="EDITABLE_TIMING_CACHE",
            DISABLE_COMPILATION_CACHE="DISABLE_COMPILATION_CACHE",
            ERROR_ON_TIMING_CACHE_MISS="ERROR_ON_TIMING_CACHE_MISS",
        ),
    )
    for ordinal, expected_level in ((13, 0), (12, 3)):
        config = Config()
        receipt = phase_a_pafpn._leaf_builder_config_receipt(config, trt, specs[ordinal])
        assert receipt["builder_optimization_level"] == expected_level
        assert receipt["source_exact_tactic_claimed"] is False
        phase_a_pafpn._validate_leaf_builder_config(config, trt, receipt)
        config.builder_optimization_level = 2
        with pytest.raises(RuntimeError, match="config changed"):
            phase_a_pafpn._validate_leaf_builder_config(config, trt, receipt)

    config = Config()
    config.timing_cache = object()
    with pytest.raises(RuntimeError, match="timing cache"):
        phase_a_pafpn._leaf_builder_config_receipt(config, trt, specs[13])


def test_phase_a_extends_the_exact_legacy_eight_section_inventory():
    legacy = [
        *source_export.ENGINE_PLAN_SECTIONS,
        source_export.NATIVE_PLUGIN_SECTION,
        "config.json",
    ]
    assert len(legacy) == len(set(legacy)) == 8
    phase_a = [
        *legacy,
        phase_a_pafpn.PAFPN_MANIFEST_SECTION,
        *(phase_a_pafpn.phase_a_plan_section(index) for index in range(137)),
    ]
    policy = phase_a_pafpn.phase_a_bundle_loading()
    assert len(phase_a) == len(set(phase_a)) == 146
    assert set(policy["eager_sections"]) | set(policy["lazy_sections"]) == set(phase_a)


def test_phase_a_is_opt_in_and_legacy_config_remains_unchanged(monkeypatch, tmp_path):
    raw = {"_model_dir": "/reviewed", "image_size": 1024, "sam2_hoi": {}}
    config = SimpleNamespace(raw=raw)
    monkeypatch.setattr(PLUGIN_MODULE, "validate_architecture", lambda _raw: None)
    monkeypatch.setattr(
        PLUGIN_MODULE, "ensure_native_plugin_loaded", lambda **_kwargs: tmp_path / "plugin.so"
    )
    (tmp_path / "plugin.so").write_bytes(b"plugin")
    monkeypatch.setattr(PLUGIN_MODULE, "build_image_feature_engine", lambda *_a, **_k: b"legacy")
    monkeypatch.setattr(
        PLUGIN_MODULE, "build_phase_a_image_front_engine", lambda *_a, **_k: b"front"
    )
    assert plugin.build_engine(config, {}, 32, precision="bf16") == b"legacy"
    assert (
        plugin.build_file_backed_bundle_sections(
            config, {}, 32, staging_dir=tmp_path, precision="bf16"
        )
        == []
    )
    assert "bundle_loading" not in plugin.get_bundle_config_overrides(config)

    config.raw["_family_build_options"] = {"sam2_hoi": {phase_a_pafpn.PHASE_A_BUILD_OPTION: True}}
    monkeypatch.setattr(
        PLUGIN_MODULE, "build_phase_a_pafpn_file_sections", lambda *_a, **_k: ["leaf"]
    )
    assert plugin.build_engine(config, {}, 32, precision="bf16") == b"front"
    assert plugin.build_file_backed_bundle_sections(
        config, {}, 32, staging_dir=tmp_path, precision="bf16"
    ) == ["leaf"]
    overrides = plugin.get_bundle_config_overrides(config)
    assert overrides["bundle_loading"] == phase_a_pafpn.phase_a_bundle_loading()
    assert overrides["phase_a_pafpn_build_policy"] == phase_a_pafpn.phase_a_pafpn_build_policy()
    assert overrides["semantic_gate_policy"] == {
        "id": "sam2-hoi-full-chain-accuracy-v2",
        "detection_score_max_abs": 0.01,
        "detection_box_max_abs_pixels": 2.0,
        "detection_box_min_iou": 0.99,
        "mask_min_iou": 0.99,
        "mask_min_dice": 0.9949748743718593,
        "mask_min_pixel_agreement": 0.999,
        "exact_fields": ["object_ids", "det_labels", "interaction_pairs"],
        "required_frame_count": 5,
    }


def test_generic_builder_emits_exact_legacy_and_phase_a_section_inventories(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "sam2_hoi",
                "architectures": ["SAM2HoiVideoTracker"],
                "vocab_size": 0,
                "hidden_size": 256,
                "intermediate_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "image_size": 1024,
                "sam2_hoi": {},
            }
        ),
        encoding="utf-8",
    )
    native_plugin = tmp_path / "sam2_hoi_native_plugin.so"
    native_plugin.write_bytes(b"plugin")
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)
    monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "11.1")
    monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "test-gpu")
    monkeypatch.setattr(PLUGIN_MODULE, "validate_architecture", lambda _raw: None)
    monkeypatch.setattr(plugin, "load_weights", lambda *_a, **_k: _weights())
    monkeypatch.setattr(
        PLUGIN_MODULE,
        "ensure_native_plugin_loaded",
        lambda **_kwargs: native_plugin,
    )
    monkeypatch.setattr(PLUGIN_MODULE, "build_image_feature_engine", lambda *_a, **_k: b"legacy")
    monkeypatch.setattr(
        PLUGIN_MODULE,
        "build_phase_a_image_front_engine",
        lambda *_a, **_k: b"front",
    )
    monkeypatch.setattr(
        PLUGIN_MODULE,
        "build_hoi_detector_engine",
        lambda *_a, **_k: b"detector",
    )
    monkeypatch.setattr(
        PLUGIN_MODULE,
        "build_interaction_engine",
        lambda *_a, **_k: b"interaction",
    )
    monkeypatch.setattr(
        PLUGIN_MODULE,
        "build_tracker_engines",
        lambda *_a, **_k: {
            source_export.PROMPT_TRACKER_SECTION: b"prompt",
            source_export.RECURRENT_TRACKER_SECTION: b"recurrent",
            source_export.MEMORY_ENCODER_SECTION: b"memory",
        },
    )
    monkeypatch.setattr(
        phase_a_pafpn,
        "_build_leaf_plan",
        lambda spec, _weights, *, verbose: f"plan-{spec.ordinal}-{verbose}".encode(),
    )
    monkeypatch.setattr(
        phase_a_pafpn.pafpn_bn_invstd,
        "helper_build_receipt",
        lambda *, verbose: {"result": "PASS", "verbose": verbose},
    )

    publications = []

    def capture_bundle(_path, _info, sections):
        config = next(
            section
            for section in sections
            if section.name == "config.json" and isinstance(section, BundleSection)
        )
        for section in sections:
            if isinstance(section, FileBundleSection):
                assert section.source_path.is_file()
        publications.append(
            {
                "names": [section.name for section in sections],
                "config": json.loads(config.data),
            }
        )

    monkeypatch.setattr(engine_builder, "write_bundle", capture_bundle)
    build_bundle(model_dir, tmp_path / "legacy.bundle", precision="bf16")
    build_bundle(
        model_dir,
        tmp_path / "phase-a.bundle",
        precision="bf16",
        family_build_options={"sam2_hoi": {"phase_a_pafpn": True}},
    )

    legacy_names = publications[0]["names"]
    assert len(legacy_names) == len(set(legacy_names)) == 8
    assert set(legacy_names) == {
        *source_export.ENGINE_PLAN_SECTIONS,
        source_export.NATIVE_PLUGIN_SECTION,
        "config.json",
    }
    assert "bundle_loading" not in publications[0]["config"]

    phase_names = publications[1]["names"]
    assert len(phase_names) == len(set(phase_names)) == 146
    assert set(phase_names) == {
        *legacy_names,
        phase_a_pafpn.PAFPN_MANIFEST_SECTION,
        *(phase_a_pafpn.phase_a_plan_section(index) for index in range(137)),
    }
    policy = publications[1]["config"]["bundle_loading"]
    assert set(policy["eager_sections"]) | set(policy["lazy_sections"]) == set(phase_names)


@pytest.mark.parametrize(
    ("family_options", "message"),
    [
        ([], "family build options must be an object"),
        ({"sam2_hoi": []}, "sam2_hoi build options must be an object"),
        ({"sam2_hoi": {"phase_a_pafpn": "yes"}}, "must be true or false"),
        ({"sam2_hoi": {"unknown": True}}, "Unknown sam2_hoi build options"),
    ],
)
def test_phase_a_build_options_are_typed_and_namespaced(family_options, message):
    config = SimpleNamespace(raw={"_family_build_options": family_options})
    with pytest.raises(ValueError, match=message):
        plugin._phase_a_enabled(config)


def test_phase_a_source_has_no_onnx_or_prebuilt_artifact_dependency():
    source = Path(phase_a_pafpn.__file__).read_text(encoding="utf-8").lower()
    assert "onnx" not in source
    assert "build_serialized_network" in source
    assert "artifact_map" not in source and "build_report" not in source


def test_phase_a_front_is_front_only_and_exposes_runtime_roots():
    source = inspect.getsource(native_image_builder.build_phase_a_image_front_engine)
    assert "_add_pafpn" not in source
    assert '"fpn_input_0"' in source and '"fpn_input_2"' in source
    assert '"tracker_feature_2"' in source and '"tracker_position_2"' in source
    assert "detector_feature_" not in source
