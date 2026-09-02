# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the native MoGe-2 ViT-L family."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from tensorrt_model_connect import engine_builder
from tensorrt_model_connect.families.moge import model as model_module


FAMILY_ROOT = Path(__file__).resolve().parents[1]
family_plugin = importlib.import_module("tensorrt_model_connect.families.moge.plugin")


def test_manifest_downloads_only_the_requested_checkpoint() -> None:
    manifest = tomllib.loads((FAMILY_ROOT / "MODEL.toml").read_text(encoding="utf-8"))

    assert manifest["id"] == "moge"
    assert manifest["module"] == "plugin"
    assert manifest["config_adapter"] == "plugin.py|config_from_dir"
    assert "model_dir_adapter" not in manifest
    assert manifest["hf_allow_patterns"] == ["model.pt"]
    assert manifest["hf_required_files"] == ["Ruicheng/moge-2-vitl|model.pt"]
    assert manifest["default_execution_profiles"] == ["reference|moge_reference"]


def test_config_adapter_claims_one_flat_checkpoint(tmp_path: Path) -> None:
    assert family_plugin.config_from_dir(tmp_path) is None
    (tmp_path / "model.pt").write_bytes(b"checkpoint")

    config = family_plugin.config_from_dir(tmp_path)

    assert config is not None
    assert config["model_type"] == "moge"
    assert config["architectures"] == ["MoGeModelV2"]
    assert config["runtime_strategy"] == "moge_monocular_geometry"
    assert config["max_position_embeddings"] == 1841
    assert config["requires_tokenizer"] is False
    assert family_plugin.plugin.matches("MoGeModelV2")
    assert not family_plugin.plugin.matches("MoGeModel")


def test_plugin_keeps_model_state_local_and_rejects_unsupported_quantization(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.pt").write_bytes(b"checkpoint")
    config = SimpleNamespace(raw={})

    assert family_plugin.plugin.default_build_precision == "fp32"
    assert family_plugin.plugin.load_weights(str(tmp_path), config) == {
        "model_dir": str(tmp_path.resolve())
    }
    assert config.raw == {}
    assert not engine_builder._call_supports_kwarg(
        family_plugin.plugin.build_engine, "parallel_config"
    )
    with pytest.raises(ValueError, match="does not support quantized"):
        family_plugin.plugin.build_engine(
            config,
            {"model_dir": str(tmp_path)},
            1,
            quant_ctx=object(),
        )


def test_checkpoint_loading_is_safe_and_memory_mapped() -> None:
    tree = ast.parse((FAMILY_ROOT / "model.py").read_text(encoding="utf-8"))
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    assert any(
        {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in call.keywords
            if keyword.arg in {"map_location", "weights_only", "mmap"}
        }
        == {"map_location": "cpu", "weights_only": True, "mmap": True}
        for call in loads
    )


def test_production_builder_is_fixed_and_tensor_rt_native_only() -> None:
    source = (FAMILY_ROOT / "model.py").read_text(encoding="utf-8")
    lowered = source.lower()

    for forbidden in (
        "torch.onnx",
        "torch.nn",
        "onnxparser",
        "onnx_path",
        "torch_tensorrt",
        "triton",
        "add_plugin",
        "get_plugin_registry",
        "trtmc_moge_",
        "add_quantize",
        "add_dequantize",
        "fp8_scale_map",
        "_fp8_dense_selection",
    ):
        assert forbidden not in lowered
    for required in (
        "create_network",
        "build_serialized_network",
        "add_attention",
        "add_normalization_v2",
        "add_convolution_nd",
        "add_deconvolution_nd",
        "InterpolationMode.CUBIC",
        "ResizeCoordinateTransformation.HALF_PIXEL",
        "SampleMode.CLAMP",
        "GELU_ERF",
        "first_transpose=(0, 3, 1, 2)",
        "_NUM_TOKENS = 1800",
        "_FOCAL_RECOVERY_SIZE = 64",
        "_FAST_MIN_IMAGE_HEIGHT = 540",
        "_FAST_MIN_IMAGE_WIDTH = 608",
        "_FAST_OPT_IMAGE_HEIGHT = 1080",
        "_FAST_OPT_IMAGE_WIDTH = 1920",
        "_FAST_MAX_IMAGE_HEIGHT = 2160",
        "_FAST_MAX_IMAGE_WIDTH = 3840",
        "attention.decomposable = not self.fast_path",
        "compute_dtype=self.trt.float16",
        "output_dtype=self.trt.float16",
        'tensor = self.cast(tensor, self.trt.float32, f"{name}.input_fp32")',
        'hidden = self.cast(hidden, self.trt.float16, "vit.residual_fp16")',
        "compute_dtype=self.trt.float16 if self.fast_path else self.trt.float32",
        "config.builder_optimization_level = 3 if fast_path else 1",
        "config.avg_timing_iterations = 3",
        "ElementWiseOperation.FLOOR_DIV",
        "add_gather",
        '"output.valid_fp16"',
        "np.full((1,) * len(tuple(tensor.shape))",
        "[[[0.5]]]",
        '"output.raw_xy"',
        '"output.raw_z"',
        '"output.focal_samples_nchw"',
        '"output.affine_depth_fp32"',
        '"output.focal_samples_fp32"',
        '"output.mask_sigmoid"',
        '"output.mask_fp32"',
    ):
        assert required in source
    assert '"output.points_nhwc"' not in source
    assert '"output.points_remap"' not in source
    assert '"output.valid.mask_finite"' not in source
    assert '"output.valid_int8"' not in source
    for output in ("affine_depth", "valid", "focal_samples", "metric_scale"):
        assert f'("{output}",' in source


def test_focal_sample_index_contract_covers_observed_shapes() -> None:
    observed_sizes = (
        (608, 1080),
        (612, 1080),
        (1066, 1920),
        (1076, 1920),
        (1078, 1920),
        (1080, 1840),
        (1080, 1904),
        (1080, 1906),
        (1080, 1912),
        (1080, 1918),
        (1080, 1920),
        (1264, 1080),
        (1428, 1080),
        (1440, 1080),
        (1674, 1080),
        (1904, 1080),
        (1906, 1080),
        (1912, 1080),
        (1918, 1074),
        (1918, 1080),
        (1920, 1076),
        (1920, 1078),
        (1920, 1080),
        (2688, 1508),
        (3840, 2156),
        (3840, 2160),
    )

    assert len(observed_sizes) == 26
    sample_size = model_module._FOCAL_RECOVERY_SIZE
    assert sample_size == 64
    for width, height in observed_sizes:
        for size in (width, height):
            indices = tuple(index * size // sample_size for index in range(sample_size))
            assert indices[0] == 0
            assert indices[-1] == (sample_size - 1) * size // sample_size
            assert all(0 <= index < size for index in indices)
            assert all(left <= right for left, right in zip(indices, indices[1:]))


def test_slim_graph_retains_legacy_mask_rounding_and_ieee_finite_edges() -> None:
    tiny_positive = np.nextafter(np.float16(0.0), np.float16(1.0))
    logits = np.asarray([-np.inf, -0.0, tiny_positive, np.inf, np.nan], dtype=np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        probabilities = np.asarray(
            1.0 / (1.0 + np.exp(-logits.astype(np.float32))), dtype=np.float16
        )
    legacy_selected = np.isfinite(probabilities) & (probabilities > np.float16(0.5))
    ordered_probability_selected = probabilities > np.float16(0.5)
    logit_selected = logits > np.float16(0.0)

    np.testing.assert_array_equal(legacy_selected, np.asarray([False, False, False, True, False]))
    np.testing.assert_array_equal(ordered_probability_selected, legacy_selected)
    np.testing.assert_array_equal(logit_selected, np.asarray([False, False, True, True, False]))
    assert probabilities[2] == np.float16(0.5)

    values = np.asarray(
        [-np.finfo(np.float16).max, np.finfo(np.float16).max, -np.inf, np.inf, np.nan],
        dtype=np.float16,
    )
    with np.errstate(invalid="ignore"):
        ordered_finite = np.abs(values) < np.float16(np.inf)
    np.testing.assert_array_equal(ordered_finite, np.isfinite(values))


def test_fp16_valid_output_uses_exact_zero_and_one_bit_patterns() -> None:
    valid = np.asarray([False, True], dtype=np.bool_).astype(np.float16)
    np.testing.assert_array_equal(valid.view(np.uint16), np.asarray([0x0000, 0x3C00], np.uint16))


def test_slim_sample_gather_preserves_the_legacy_fp16_cast_boundary() -> None:
    height, width = 67, 83
    affine_nchw = np.arange(3 * height * width, dtype=np.float32).reshape(1, 3, height, width)
    affine_nchw = np.asarray(affine_nchw / 257.0, dtype=np.float16)
    rows = np.asarray([index * height // 64 for index in range(64)])
    columns = np.asarray([index * width // 64 for index in range(64)])

    legacy_nhwc = np.transpose(affine_nchw, (0, 2, 3, 1)).astype(np.float32)
    legacy_samples = legacy_nhwc[:, rows, :, :][:, :, columns, :]
    sampled_nchw = affine_nchw[:, :, rows, :][:, :, :, columns]
    slim_samples = np.transpose(sampled_nchw, (0, 2, 3, 1)).astype(np.float32)
    np.testing.assert_array_equal(slim_samples, legacy_samples)

    legacy_depth = legacy_nhwc[..., 2]
    slim_depth = affine_nchw[:, 2, :, :].astype(np.float32)
    np.testing.assert_array_equal(slim_depth, legacy_depth)


def test_build_rejects_unknown_precision_and_wrong_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"wrong checkpoint")

    with pytest.raises(ValueError, match="supports precision='fp32' or 'fp16' only"):
        model_module.build_moge_engine(str(tmp_path), precision="bf16")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        model_module.build_moge_engine(str(tmp_path), precision="fp32")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        model_module.build_moge_engine(str(tmp_path), precision="fp16")
