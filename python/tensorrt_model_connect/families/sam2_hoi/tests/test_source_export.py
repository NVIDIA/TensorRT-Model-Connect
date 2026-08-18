# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest

from tensorrt_model_connect.families.sam2_hoi import interaction_builder, source_export


def test_family_production_python_has_no_onnx_path():
    family_dir = Path(source_export.__file__).parent
    forbidden = "on" + "nx"
    offenders = [
        path.name
        for path in sorted(family_dir.glob("*.py"))
        if forbidden in path.read_text(encoding="utf-8").casefold()
    ]
    assert offenders == []


def test_only_checkpoint_loader_may_import_torch():
    family_dir = Path(source_export.__file__).parent
    offenders = []
    for path in sorted(family_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports_torch = any(
            (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and (node.module == "torch" or node.module.startswith("torch."))
            )
            for node in ast.walk(tree)
        )
        if imports_torch and path.name != "checkpoint.py":
            offenders.append(path.name)
    assert offenders == []


def test_source_export_is_dependency_light_family_owned_and_native_only():
    path = Path(source_export.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            top_level_imports.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            top_level_imports.append(statement.module)

    assert not any(
        name == dependency or name.startswith(dependency + ".")
        for name in top_level_imports
        for dependency in ("torch", "sam2", "mmcv", "mmdet", "tensorrt")
    )
    assert "families.sam3" not in source
    assert "torch.onnx" not in source
    assert "OnnxParser" not in source
    assert "create_network" not in source  # stage modules own Network Definition details


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_fixed_stage_contracts(precision):
    image = source_export.image_feature_contract(precision)
    assert image.section == "engine_plan"
    assert image.inputs == (
        source_export.TensorSpec("pixel_values", "float32", (1, 3, 1024, 1024)),
    )
    assert image.outputs == (
        "tracker_feature_0",
        "tracker_feature_1",
        "tracker_feature_2",
        "tracker_position_2",
        "detector_feature_0",
        "detector_feature_1",
        "detector_feature_2",
    )

    detector = source_export.hoi_detector_contract(precision)
    assert [item.shape for item in detector.inputs] == [
        (1, 256, 128, 128),
        (1, 256, 64, 64),
        (1, 256, 32, 32),
    ]
    assert detector.outputs == ("class_scores", "boxes_cxcywh", "query_embeddings")

    interaction = source_export.interaction_contract(precision)
    assert interaction.inputs == (
        source_export.TensorSpec(
            "pair_features",
            "float32",
            (None, 512),
            min_shape=(1, 512),
            opt_shape=(8, 512),
            max_shape=(22_500, 512),
            dynamic_axes=((0, "pair_count"),),
        ),
    )
    assert interaction.outputs == ("interaction_probabilities",)

    prompt = source_export.prompt_tracker_contract(precision)
    assert prompt.inputs[-2].shape == (2, 3, 2)
    assert prompt.inputs[-1] == source_export.TensorSpec("point_labels", "int32", (2, 3))
    assert prompt.outputs == (
        "pred_masks",
        "object_pointer",
        "object_score_logits",
        "selected_iou",
    )

    memory = source_export.memory_encoder_contract(precision)
    assert memory.inputs[1].shape == (2, 1, 256, 256)
    assert memory.inputs[-1] == source_export.TensorSpec("is_mask_from_points", "int32", (2, 1))
    assert memory.outputs == ("new_memory_features", "new_memory_position")


def test_bf16_contract_preserves_reviewed_split_plan_dtypes():
    assert {item.dtype for item in source_export.hoi_detector_contract("bf16").inputs} == {
        "bfloat16"
    }
    prompt = source_export.prompt_tracker_contract("bf16")
    assert [item.dtype for item in prompt.inputs] == [
        "bfloat16",
        "bfloat16",
        "float32",
        "float32",
        "int32",
    ]
    recurrent = {
        item.name: item for item in source_export.recurrent_tracker_contract("bf16").inputs
    }
    assert recurrent["tracker_feature_2"].dtype == "float32"
    assert recurrent["tracker_position_2"].dtype == "float32"
    assert recurrent["memory_features"].dtype == "bfloat16"


def test_recurrent_contract_has_bounded_dynamic_profiles():
    bindings = {item.name: item for item in source_export.recurrent_tracker_contract("bf16").inputs}
    memory = bindings["memory_features"]
    assert memory.shape == (2, None, 64, 64, 64)
    assert memory.min_shape == (2, 1, 64, 64, 64)
    assert memory.opt_shape == (2, 3, 64, 64, 64)
    assert memory.max_shape == (2, 7, 64, 64, 64)
    assert memory.dynamic_axes == ((1, "memory_frames"),)
    pointers = bindings["object_pointers"]
    assert pointers.min_shape == (2, 1, 256)
    assert pointers.opt_shape == (2, 2, 256)
    assert pointers.max_shape == (2, 16, 256)


def test_tensor_spec_rejects_partial_dynamic_profile():
    broken = source_export.TensorSpec(
        "memory",
        "float32",
        (2, None, 64),
        min_shape=(2, 1, 64),
        dynamic_axes=((1, "memory_frames"),),
    )
    with pytest.raises(RuntimeError, match="must define all profile shapes"):
        broken.validate()


def _install_fake_builder(monkeypatch, suffix: str, function_name: str, result):
    package = "tensorrt_model_connect.families.sam2_hoi"
    module_name = f"{package}.{suffix}"
    module = ModuleType(module_name)
    calls = []

    def build(weights, *, precision, verbose):
        calls.append((weights, precision, verbose))
        return result

    setattr(module, function_name, build)
    monkeypatch.setitem(sys.modules, module_name, module)
    return calls


def test_public_builders_dispatch_checkpoint_weights_to_native_stages(monkeypatch):
    from tensorrt_model_connect.families.sam2_hoi import checkpoint

    weights = object()
    loads = []
    monkeypatch.setattr(
        checkpoint,
        "load_checkpoint",
        lambda model_dir: loads.append(model_dir) or weights,
    )
    image_calls = _install_fake_builder(
        monkeypatch,
        "native_image_builder",
        "build_image_feature_engine",
        b"image-plan",
    )
    detector_calls = _install_fake_builder(
        monkeypatch,
        "native_detector_builder",
        "build_hoi_detector_engine",
        b"detector-plan",
    )
    tracker_calls = _install_fake_builder(
        monkeypatch,
        "native_tracker_builder",
        "build_tracker_engines",
        {"tracker": b"plan"},
    )
    plugin_loads = []
    image_events = []
    original_image_build = sys.modules[
        "tensorrt_model_connect.families.sam2_hoi.native_image_builder"
    ].build_image_feature_engine
    sys.modules[
        "tensorrt_model_connect.families.sam2_hoi.native_image_builder"
    ].build_image_feature_engine = lambda *args, **kwargs: (
        image_events.append("build_image") or original_image_build(*args, **kwargs)
    )
    monkeypatch.setattr(
        source_export,
        "ensure_native_plugin_loaded",
        lambda *, verbose: (
            plugin_loads.append(verbose) or image_events.append("load_native_plugin")
        ),
    )

    assert source_export.build_image_feature_engine("reviewed", precision="bf16") == b"image-plan"
    assert source_export.build_image_feature_engine("reviewed", precision="fp32") == b"image-plan"
    assert (
        source_export.build_hoi_detector_engine("reviewed", precision="bf16", verbose=True)
        == b"detector-plan"
    )
    assert source_export.build_tracker_engines("reviewed", precision="bf16") == {"tracker": b"plan"}
    assert loads == ["reviewed", "reviewed", "reviewed", "reviewed"]
    assert image_calls == [(weights, "bf16", False), (weights, "fp32", False)]
    assert detector_calls == [(weights, "bf16", True)]
    assert tracker_calls == [(weights, "bf16", False)]
    assert plugin_loads == [False, False, True, False]
    assert image_events[:2] == ["load_native_plugin", "build_image"]


def test_native_plugin_loader_rejects_a_second_dso_identity(monkeypatch, tmp_path):
    from tensorrt_model_connect.families.sam2_hoi import native_plugin_builder

    first = tmp_path / "first.so"
    second = tmp_path / "second.so"
    first.write_bytes(b"reviewed-native-plugin-a")
    second.write_bytes(b"reviewed-native-plugin-b")
    paths = iter((first, second))
    dependency = tmp_path / "libcublasLt.so.13"
    dependency.write_bytes(b"reviewed-cublaslt")
    loads = []
    closure_checks = []
    monkeypatch.setattr(
        native_plugin_builder,
        "ensure_native_plugin",
        lambda *, verbose: next(paths),
    )
    monkeypatch.setattr(
        source_export.ctypes,
        "CDLL",
        lambda path, *, mode: loads.append((path, mode)) or object(),
    )
    monkeypatch.setattr(
        native_plugin_builder,
        "_expected_runtime_cublaslt",
        lambda _path: {"path": str(dependency)},
    )
    monkeypatch.setattr(
        native_plugin_builder,
        "_verify_loaded_cublaslt",
        lambda path, *, allow_unloaded=False: closure_checks.append((path, allow_unloaded)),
    )
    monkeypatch.setattr(source_export, "_LOADED_NATIVE_PLUGIN_HANDLES", {})
    monkeypatch.setattr(source_export, "_LOADED_NATIVE_DEPENDENCY_HANDLES", {})
    monkeypatch.setattr(source_export, "_LOADED_NATIVE_PLUGIN_SHA256", None)

    assert source_export.ensure_native_plugin_loaded() == first
    with pytest.raises(RuntimeError, match="different SAM2 HOI native plugin"):
        source_export.ensure_native_plugin_loaded()
    assert loads == [
        (str(dependency), source_export.ctypes.RTLD_GLOBAL),
        (str(first), source_export.ctypes.RTLD_GLOBAL),
    ]
    assert closure_checks == [
        (first, True),
        (first, False),
        (first, False),
    ]


def test_public_builders_and_section_names_are_stable():
    assert callable(source_export.build_image_feature_engine)
    assert callable(source_export.build_hoi_detector_engine)
    assert callable(source_export.build_tracker_engines)
    assert source_export.ENGINE_PLAN_SECTIONS == (
        "engine_plan",
        "sam2_hoi_detector_engine_plan",
        "sam2_hoi_interaction_engine_plan",
        "sam2_hoi_prompt_tracker_engine_plan",
        "sam2_hoi_recurrent_tracker_engine_plan",
        "sam2_hoi_memory_encoder_engine_plan",
    )
    assert source_export.BUNDLE_SECTIONS == (
        *source_export.ENGINE_PLAN_SECTIONS,
        "sam2_hoi_native_plugin_so",
    )
    assert interaction_builder.INTERACTION_SECTION == source_export.INTERACTION_SECTION


@pytest.mark.parametrize("precision", ["fp16", "int8", "", "BFLOAT16"])
def test_unsupported_precision_fails_before_dependency_loading(precision):
    with pytest.raises(ValueError, match="native build supports precision"):
        source_export.image_feature_contract(precision)
