# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the model-owned SAM3.0 tracker plan builder."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.sam3 import tracker_builder, tracker_weights


def test_sam3_tracker_step_profiles_cover_native_memory_bounds() -> None:
    assert tracker_builder._step_profile_shapes("memory_features") == (
        (1, 1, 5184, 64),
        (1, 3, 5184, 64),
        (1, 10, 5184, 64),
    )
    assert tracker_builder._step_profile_shapes("memory_temporal_offsets") == (
        (1, 1),
        (1, 3),
        (1, 10),
    )
    assert tracker_builder._step_profile_shapes("object_pointers") == (
        (1, 1, 256),
        (1, 4, 256),
        (1, 79, 256),
    )
    assert tracker_builder._step_profile_shapes("object_pointer_temporal_offsets") == (
        (1, 1),
        (1, 4),
        (1, 79),
    )
    assert tracker_builder._step_profile_shapes("tracker_feature_2") is None


def test_sam3_tracker_batch2_step_profiles_use_measured_opt_shapes() -> None:
    assert tracker_builder._step_batch2_profile_shapes("memory_features") == (
        (2, 1, 5184, 64),
        (2, 7, 5184, 64),
        (2, 10, 5184, 64),
    )
    assert tracker_builder._step_batch2_profile_shapes("memory_temporal_offsets") == (
        (2, 1),
        (2, 7),
        (2, 10),
    )
    assert tracker_builder._step_batch2_profile_shapes("object_pointers") == (
        (2, 1, 256),
        (2, 16, 256),
        (2, 79, 256),
    )
    assert tracker_builder._step_batch2_profile_shapes("object_pointer_temporal_offsets") == (
        (2, 1),
        (2, 16),
        (2, 79),
    )
    assert tracker_builder._step_batch2_profile_shapes("tracker_feature_2") is None


def _official_tracker_config() -> dict[str, object]:
    return {
        "image_size": 1008,
        "vision_config": {
            "backbone_feature_sizes": [[288, 288], [144, 144], [72, 72]],
            "fpn_hidden_size": 256,
            "num_feature_levels": 3,
        },
        "prompt_encoder_config": {
            "hidden_size": 256,
            "image_size": 1008,
            "patch_size": 14,
            "mask_input_channels": 16,
            "num_point_embeddings": 4,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu",
            "scale": 1,
        },
        "mask_decoder_config": {
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "attention_downsample_rate": 2,
            "mlp_dim": 2048,
            "num_multimask_outputs": 3,
            "iou_head_depth": 3,
            "iou_head_hidden_dim": 256,
        },
        "num_maskmem": 7,
        "max_cond_frame_num": 4,
        "max_object_pointers_in_encoder": 16,
        "memory_attention_hidden_size": 256,
        "memory_attention_num_attention_heads": 1,
        "memory_attention_num_layers": 4,
        "memory_attention_feed_forward_hidden_size": 2048,
        "memory_attention_feed_forward_hidden_act": "relu",
        "memory_attention_downsample_rate": 1,
        "memory_attention_rope_feat_sizes": [72, 72],
        "memory_attention_rope_theta": 10000,
        "memory_encoder_hidden_size": 256,
        "memory_encoder_output_channels": 64,
        "mask_downsampler_embed_dim": 256,
        "mask_downsampler_hidden_act": "gelu",
        "mask_downsampler_kernel_size": 3,
        "mask_downsampler_padding": 1,
        "mask_downsampler_stride": 2,
        "mask_downsampler_total_stride": 16,
        "memory_fuser_embed_dim": 256,
        "memory_fuser_hidden_act": "gelu",
        "memory_fuser_intermediate_dim": 1024,
        "memory_fuser_kernel_size": 7,
        "memory_fuser_padding": 3,
        "memory_fuser_num_layers": 2,
        "memory_fuser_layer_scale_init_value": 1e-6,
        "sigmoid_scale_for_mem_enc": 20.0,
        "sigmoid_bias_for_mem_enc": -10.0,
        "enable_occlusion_spatial_embedding": True,
        "enable_temporal_pos_encoding_for_object_pointers": True,
        "multimask_output_for_tracking": True,
        "multimask_output_in_sam": True,
        "multimask_min_pt_num": 0,
        "multimask_max_pt_num": 1,
    }


def _set_config_path(config: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = config
    for name in path[:-1]:
        nested = current[name]
        assert isinstance(nested, dict)
        current = nested
    current[path[-1]] = value


def test_sam3_tracker_validation_accepts_official_sam3_architecture() -> None:
    tracker_builder._validate_tracker_model(SimpleNamespace(config=_official_tracker_config()))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_size", 1024, "1008px tracker only"),
        ("num_maskmem", 8, "memory profile must match"),
        ("max_cond_frame_num", 5, "conditioning-memory profile must match"),
        ("max_object_pointers_in_encoder", 32, "pointer profile must match"),
    ],
)
def test_sam3_tracker_validation_rejects_unreviewed_variants(
    field: str, value: int, message: str
) -> None:
    config = _official_tracker_config()
    config[field] = value
    with pytest.raises(RuntimeError, match=message):
        tracker_builder._validate_tracker_model(SimpleNamespace(config=config))


@pytest.mark.parametrize(
    ("path", "unreviewed_value"),
    [
        (path, False if expected is True else "unreviewed")
        for path, expected in tracker_builder._TRACKER_ARCHITECTURE_CONTRACT
    ],
)
def test_sam3_tracker_validation_rejects_every_unreviewed_graph_config(
    path: tuple[str, ...], unreviewed_value: object
) -> None:
    config = _official_tracker_config()
    _set_config_path(config, path, unreviewed_value)

    with pytest.raises(RuntimeError, match="supports only the official architecture"):
        tracker_builder._validate_tracker_model(SimpleNamespace(config=config))


def test_sam3_tracker_model_config_rejects_unreviewed_low_resolution(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "low_res_mask_size": 288,
                "tracker_config": _official_tracker_config(),
            }
        ),
        encoding="utf-8",
    )
    tracker_builder._read_model_config(str(tmp_path))

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["low_res_mask_size"] = 256
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected low_res_mask_size=288"):
        tracker_builder._read_model_config(str(tmp_path))


def test_sam3_tracker_bundle_section_names_are_stable() -> None:
    assert tracker_builder.TRACKER_INIT_SECTION == "sam3_tracker_init_engine_plan"
    assert tracker_builder.TRACKER_STEP_SECTION == "sam3_tracker_step_engine_plan"
    assert tracker_builder.TRACKER_STEP_BATCH2_SECTION == "sam3_tracker_step_batch2_engine_plan"
    assert tracker_builder.TRACKER_MEMORY_SECTION == "sam3_tracker_memory_engine_plan"
    assert tracker_builder.TRACKER_MEMORY_BATCH2_SECTION == "sam3_tracker_memory_batch2_engine_plan"


def test_sam3_tracker_builder_has_only_required_b1_b2_plans() -> None:
    parameters = inspect.signature(tracker_builder.build_sam3_tracker_engines).parameters
    assert "fp16_engines" not in parameters
    assert "fp16_ops" not in parameters

    source = inspect.getsource(tracker_builder.build_sam3_tracker_engines)
    for section in (
        "TRACKER_INIT_SECTION",
        "TRACKER_STEP_SECTION",
        "TRACKER_STEP_BATCH2_SECTION",
        "TRACKER_MEMORY_SECTION",
        "TRACKER_MEMORY_BATCH2_SECTION",
    ):
        assert section in source


def test_sam3_tracker_reviewed_video_bound_derives_pointer_profile() -> None:
    assert tracker_builder.SAM3_TRACKER_MAX_VIDEO_FRAMES == 1024
    assert tracker_builder.SAM3_TRACKER_RECONDITION_CADENCE == 16
    assert tracker_builder.SAM3_TRACKER_MAX_CONDITIONING_POINTERS == 64
    assert tracker_builder.SAM3_TRACKER_MAX_POINTER_INPUTS == 79


def _sam3_production_sources() -> dict[Path, str]:
    family_dir = Path(tracker_builder.__file__).resolve().parent
    sources = {path: path.read_text(encoding="utf-8") for path in sorted(family_dir.glob("*.py"))}
    assert sources, "SAM3 production sources were not found"
    return sources


def _call_attribute_names(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_sam3_production_has_no_exchange_graph_path() -> None:
    """The SAM3 contract permits only direct TensorRT graph construction."""

    forbidden_import_roots = {
        "onnx",
        "onnx_ir",
        "onnxscript",
    }
    tracker_framework_roots = {"torch", "transformers"}
    forbidden_source_tokens = {
        ".onnx",
        "modelproto",
        "nvonnxparser",
        "onnxparser",
        "opset",
        "torch.export",
        "torch.onnx",
    }

    for path, source in _sam3_production_sources().items():
        tree = ast.parse(source, filename=str(path))
        imports: set[str] = set()
        symbols: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", maxsplit=1)[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name.lower())
            elif isinstance(node, ast.Attribute):
                symbols.add(node.attr.lower())
            elif isinstance(node, ast.Name):
                symbols.add(node.id.lower())

        assert not (imports & forbidden_import_roots), (
            f"{path.name} imports an exchange-graph dependency: "
            f"{sorted(imports & forbidden_import_roots)}"
        )
        if path.name.startswith("tracker"):
            assert not (imports & tracker_framework_roots), (
                f"{path.name} imports a tracker framework dependency: "
                f"{sorted(imports & tracker_framework_roots)}"
            )
        lowered = source.lower()
        matched_tokens = sorted(token for token in forbidden_source_tokens if token in lowered)
        assert not matched_tokens, (
            f"{path.name} contains a forbidden exchange-graph path: {matched_tokens}"
        )
        forbidden_symbols = sorted(
            symbol
            for symbol in symbols
            if "onnx" in symbol
            or "opset" in symbol
            or "modelproto" in symbol
            or "export" in symbol
            or symbol in {"parse", "parse_from_file"}
            or symbol.endswith("parser")
        )
        assert not forbidden_symbols, (
            f"{path.name} defines or calls a graph exporter/parser: {forbidden_symbols}"
        )


def test_sam3_tracker_graph_is_built_with_native_tensorrt_api() -> None:
    """Guard the build boundary, not just the absence of a particular parser."""

    calls: set[str] = set()
    for path, source in _sam3_production_sources().items():
        if not path.name.startswith("tracker"):
            continue
        calls.update(_call_attribute_names(ast.parse(source, filename=str(path))))

    assert {"Builder", "create_network", "add_input", "build_serialized_network"} <= calls
    native_layer_calls = {
        "add_activation",
        "add_attention",
        "add_convolution_nd",
        "add_elementwise",
        "add_matrix_multiply",
        "add_resize",
        "add_shuffle",
        "add_slice",
        "add_topk",
    }
    used_native_layers = calls & native_layer_calls
    assert len(used_native_layers) >= 4, (
        "SAM3 tracker sources must reconstruct the graph with TensorRT layers; "
        f"found only {sorted(used_native_layers)}"
    )


def test_sam3_tracker_video_policy_rejects_unreviewed_cadence(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"recondition_every_nth_frame": 16}), encoding="utf-8")
    tracker_builder._validate_video_policy(str(tmp_path))

    config_path.write_text(json.dumps({"recondition_every_nth_frame": 8}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="reviewed reconditioning cadence 16"):
        tracker_builder._validate_video_policy(str(tmp_path))


class _FakeSafeTensorReader:
    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self.tensors = tensors
        self.reads: list[str] = []

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key: str) -> np.ndarray:
        self.reads.append(key)
        return self.tensors[key]


def test_sam3_tracker_weights_are_loaded_directly_as_numpy(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    reader = _FakeSafeTensorReader(
        {
            "tracker_model.projection.weight": np.arange(6, dtype=np.float16).reshape(2, 3),
            "tracker_model.projection.bias": np.array([2.0, 3.0], dtype=np.float16),
            "vision_encoder.unrelated": np.ones((1,), dtype=np.float32),
        }
    )
    opened: list[tuple[str, str]] = []

    def fake_safe_open(path: str, *, framework: str):
        opened.append((path, framework))
        return reader

    monkeypatch.setattr(tracker_weights, "safe_open", fake_safe_open)
    weights = tracker_weights.load_tracker_weights(tmp_path)

    np.testing.assert_array_equal(
        weights.linear_weight("projection"),
        np.arange(6, dtype=np.float32).reshape(2, 3).T,
    )
    np.testing.assert_array_equal(
        weights.linear_bias("projection"), np.array([2.0, 3.0], dtype=np.float32)
    )
    assert weights["projection.bias"].flags.c_contiguous
    assert opened == [(str(checkpoint), "numpy")]
    assert reader.reads.count("tracker_model.projection.bias") == 1
    with pytest.raises(KeyError, match="Missing SAM3 tracker parameter"):
        _ = weights["vision_encoder.unrelated"]


def test_sam3_tracker_weights_load_only_tracker_shards(monkeypatch, tmp_path) -> None:
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "tracker_model.a.weight": "tracker-1.safetensors",
                    "tracker_model.b.weight": "tracker-2.safetensors",
                    "vision_encoder.weight": "vision.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    readers = {
        "tracker-1.safetensors": _FakeSafeTensorReader(
            {"tracker_model.a.weight": np.ones((1,), dtype=np.float32)}
        ),
        "tracker-2.safetensors": _FakeSafeTensorReader(
            {"tracker_model.b.weight": np.full((1,), 2.0, dtype=np.float32)}
        ),
    }
    opened: list[str] = []

    def fake_safe_open(path: str, *, framework: str):
        assert framework == "numpy"
        name = Path(path).name
        opened.append(name)
        return readers[name]

    monkeypatch.setattr(tracker_weights, "safe_open", fake_safe_open)
    weights = tracker_weights.load_tracker_weights(tmp_path)

    assert sorted(opened) == ["tracker-1.safetensors", "tracker-2.safetensors"]
    np.testing.assert_array_equal(weights["a.weight"], np.ones((1,), dtype=np.float32))
    np.testing.assert_array_equal(weights["b.weight"], np.full((1,), 2.0, dtype=np.float32))
