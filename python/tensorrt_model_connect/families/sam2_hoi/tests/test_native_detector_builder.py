# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi import native_detector_builder as detector


def test_native_detector_has_complete_fail_closed_weight_contract() -> None:
    keys = detector.detector_required_weight_keys()
    assert len(keys) == 296
    assert len(keys) == len(set(keys))
    assert (
        "image_encoder.hoi_head.query_head.transformer.encoder.layers.5."
        "attentions.0.sampling_offsets.weight"
    ) in keys
    assert (
        "image_encoder.hoi_head.query_head.transformer.decoder.layers.5."
        "attentions.1.output_proj.weight"
    ) in keys
    assert "image_encoder.hoi_head.query_head.cls_branches.6.weight" in keys
    assert "image_encoder.hoi_head.query_head.reg_branches.6.4.bias" in keys

    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        detector.build_hoi_detector_engine({}, precision="bf16")


def test_native_detector_locks_the_qualified_builder_search_budget() -> None:
    assert detector._BUILDER_OPTIMIZATION_LEVEL == 5
    assert detector._AVG_TIMING_ITERATIONS == 8
    assert detector._MAX_AUX_STREAMS == 0


def test_native_detector_fused_bf16_linear_allowlist_is_exactly_qualified_sites() -> None:
    expected = {
        *(
            detector._TRANSFORMER + f"encoder.layers.{layer}.attentions.0.output_proj"
            for layer in range(detector._NUM_ENCODER_LAYERS)
        ),
        *(
            detector._TRANSFORMER + f"decoder.layers.{layer}.attentions.1.output_proj"
            for layer in range(detector._NUM_DECODER_LAYERS)
        ),
        detector._TRANSFORMER + "enc_output",
        detector._PREFIX + f"reg_branches.{detector._NUM_DECODER_LAYERS}.0",
        detector._PREFIX + f"reg_branches.{detector._NUM_DECODER_LAYERS}.2",
    }
    assert len(expected) == 15
    assert detector._FUSED_BF16_LINEAR_PREFIXES == frozenset(expected)


def test_native_detector_fused_bf16_linear_keeps_bias_inside_native_conv() -> None:
    helper = inspect.getsource(detector._linear_via_fused_1x1_conv)
    assert "add_convolution_nd" in helper
    assert "layer.set_input(1, weight_tensor)" in helper
    assert "layer.set_input(2, bias_tensor)" in helper
    assert "prefix not in _FUSED_BF16_LINEAR_PREFIXES" in helper

    msda = inspect.getsource(detector._msda)
    assert "_linear_via_fused_1x1_conv" in msda
    assert 'prefix + ".output_proj"' in msda

    proposal = inspect.getsource(detector._initial_references)
    assert "_linear_via_fused_1x1_conv" in proposal
    assert '_TRANSFORMER + "enc_output"' in proposal

    regression = inspect.getsource(detector._regression_branch)
    assert "branch == _NUM_DECODER_LAYERS" in regression
    assert regression.count("hidden = linear(") == 2
    assert 'prefix + ".4"' in regression


def test_native_detector_fixed_geometry_matches_reviewed_three_level_head() -> None:
    references = detector._encoder_reference_points()
    proposals, valid = detector._encoder_proposals()
    assert references.shape == (1, 21504, 3, 2)
    assert proposals.shape == (1, 21504, 4)
    assert valid.shape == (1, 21504, 1)
    np.testing.assert_allclose(
        references[0, 0],
        np.asarray([[0.5 / 128.0, 0.5 / 128.0]] * 3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        references[0, 128 * 128],
        np.asarray([[0.5 / 64.0, 0.5 / 64.0]] * 3, dtype=np.float32),
    )
    assert int(valid.sum()) == 20744
    assert np.isinf(proposals[0, 0]).all()
    assert np.isfinite(proposals[0, 129]).all()


def test_native_detector_position_encoding_is_fixed_nchw_y_then_x() -> None:
    position = detector._sine_position_encoding(2, 4)
    assert position.shape == (1, 256, 2, 4)
    y_phase = np.float32(1.0) / (np.float32(2.0) + np.float32(1.0e-6)) * np.float32(2.0 * np.pi)
    x_phase = np.float32(1.0) / (np.float32(4.0) + np.float32(1.0e-6)) * np.float32(2.0 * np.pi)
    np.testing.assert_allclose(position[0, 0, 0, 0], np.sin(y_phase), rtol=1e-6)
    np.testing.assert_allclose(position[0, 128, 0, 0], np.sin(x_phase), rtol=1e-6)
    np.testing.assert_allclose(position[0, :128, 0, 0], position[0, :128, 0, 3])


def test_native_detector_builder_has_no_interchange_graph_dependency() -> None:
    source_path = Path(detector.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported_roots
    assert "onnx" not in source.casefold()
    assert "add_topk" in source
    assert "Sam2HoiMsDeformAttn" in source
    assert "Sam2HoiLayerNorm256" in source
    assert "Sam2HoiSoftmax" in source
    assert "Sam2HoiMhaScale" in source
    assert "Sam2HoiSigmoid" in source
