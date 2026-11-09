# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.foundationpose import calibrate_fp8_scales, quantization


def _node(name: str, op_type: str, inputs: list[str]) -> SimpleNamespace:
    return SimpleNamespace(name=name, op_type=op_type, input=inputs)


REFINER_NODES = [
    _node("/encodeA/encodeA.0/net/net.0/Conv", "Conv", ["x", "w0", "b0"]),
    _node("/encodeA/encodeA.2/conv1/Conv", "Conv", ["a", "w1", "b1"]),
    _node("/encodeAB/encodeAB.0/conv1/Conv", "Conv", ["c", "w2", "b2"]),
    _node("/encodeAB/encodeAB.2/net/net.0/Conv", "Conv", ["d", "w3", "b3"]),
    _node("/trans_head/trans_head.0/self_attn/MatMul", "MatMul", ["t", "onnx::MatMul_1"]),
    _node("/trans_head/trans_head.0/linear1/MatMul", "MatMul", ["n", "onnx::MatMul_2"]),
]
SCORER_NODES = [
    _node("/encoderA/encoderA.1/net/net.0/Conv", "Conv", ["a", "w1", "b1"]),
    _node("/encoderAB/encoderAB.4/conv2/Conv", "Conv", ["e", "w4", "b4"]),
    _node("/att/MatMul", "MatMul", ["t", "onnx::MatMul_4"]),
    _node("/linear/MatMul", "MatMul", ["p", "onnx::MatMul_6"]),
]


def test_recipe_quantizes_only_the_paired_feature_convolutions() -> None:
    assert quantization.quantized_nodes(REFINER_NODES) == [
        "/encodeAB/encodeAB.0/conv1/Conv",
        "/encodeAB/encodeAB.2/net/net.0/Conv",
    ]
    assert quantization.quantized_nodes(SCORER_NODES) == ["/encoderAB/encoderAB.4/conv2/Conv"]


def test_weight_scale_is_amax_over_448_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(64, 32, 3, 3)).astype(np.float32)
    weight[0, 0, 0, 0] = 8.96
    packed, scale = quantization.quantize_fp8_weight(weight)
    assert packed.dtype == np.uint8 and packed.shape == weight.shape
    assert scale == pytest.approx(8.96 / 448.0)
    values = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    assert values[0, 0, 0, 0] == 448.0
    restored = values * scale
    # e4m3 keeps 3 mantissa bits: relative error at most 2**-4 for normal values.
    big = np.abs(weight) > 1e-2 * np.abs(weight).max()
    assert np.max(np.abs(restored - weight)[big] / np.abs(weight)[big]) <= 2.0**-4


def test_all_zero_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="amax"):
        quantization.quantize_fp8_weight(np.zeros((4, 4), dtype=np.float32))


ARRAYS = {
    "w0": np.full((64, 6, 7, 7), 0.5, np.float32),
    "w1": np.full((128, 64, 3, 3), 0.25, np.float32),
    "w4": np.full((512, 512, 3, 3), -2.0, np.float32),
}
SCORER_CONV = "/encoderAB/encoderAB.4/conv2/Conv"


def _table(tmp_path: Path, scales: dict, weights: dict | None = None) -> Path:
    path = tmp_path / quantization.ACTIVATION_SCALES_FILENAME
    entry = {"activation_scales": scales, "weight_amax": weights or {SCORER_CONV: 2.0}}
    path.write_text(json.dumps({quantization.scale_key("scorer"): entry}), encoding="utf-8")
    return path


def test_weight_amax_reads_each_quantized_convolution() -> None:
    assert quantization.weight_amax(SCORER_NODES, ARRAYS) == {SCORER_CONV: 2.0}


def test_quant_context_pairs_scales_and_skips_null_outliers(tmp_path: Path) -> None:
    path = _table(tmp_path, {SCORER_CONV: 0.5})
    context = quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)
    assert context.has(SCORER_CONV)
    assert not context.has("/encoderA/encoderA.1/net/net.0/Conv")
    path = _table(tmp_path, {SCORER_CONV: None})
    context = quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)
    assert not context.has(SCORER_CONV)


def test_quant_context_rejects_missing_unknown_and_invalid_scales(tmp_path: Path) -> None:
    path = _table(tmp_path, {})
    with pytest.raises(ValueError, match="no FP8 activation scale"):
        quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)
    path = _table(tmp_path, {SCORER_CONV: 0.5, "/att/MatMul": 1.0})
    with pytest.raises(ValueError, match="not quantized"):
        quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)
    path = _table(tmp_path, {SCORER_CONV: -1.0})
    with pytest.raises(ValueError, match="finite and positive"):
        quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)


def test_other_weights_are_refused(tmp_path: Path) -> None:
    path = _table(tmp_path, {SCORER_CONV: 0.5}, {SCORER_CONV: 2.5})
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)


def test_missing_model_config_asks_for_calibration(tmp_path: Path) -> None:
    path = tmp_path / quantization.ACTIVATION_SCALES_FILENAME
    path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="calibrate_fp8_scales.py"):
        quantization.quant_context(SCORER_NODES, ARRAYS, "scorer", scales_path=path)


def test_already_quantized_artifact_is_refused(tmp_path: Path) -> None:
    nodes = [*SCORER_NODES, _node("dq", "DequantizeLinear", ["w", "s"])]
    with pytest.raises(ValueError, match="already quantized"):
        quantization.quant_context(nodes, ARRAYS, "scorer", scales_path=tmp_path / "missing.json")


def test_committed_scales_cover_both_engines() -> None:
    table = json.loads(
        Path(quantization.__file__).with_name(quantization.ACTIVATION_SCALES_FILENAME).read_text()
    )
    assert sorted(table) == [quantization.scale_key("refiner"), quantization.scale_key("scorer")]
    for entry in table.values():
        scales = entry["activation_scales"]
        assert len(scales) == 9 and set(scales) == set(entry["weight_amax"])
        assert all(
            value is not None and np.isfinite(value) and value > 0 for value in scales.values()
        )


def test_calibration_outliers_become_null() -> None:
    amax = {"a": 1.0, "b": 1.2, "c": 0.9, "d": 50.0}
    scales = calibrate_fp8_scales.scales_from_amax(amax)
    assert scales["d"] is None
    assert scales["a"] == pytest.approx(1.0 / 448.0)
    with pytest.raises(ValueError, match="no calibration activation"):
        calibrate_fp8_scales.scales_from_amax({"a": 1.0, "b": 0.0})


def test_calibration_crops_follow_the_input_contract(tmp_path: Path) -> None:
    scene = tmp_path / "Scene"
    scene.mkdir()
    from PIL import Image

    Image.fromarray(np.full((120, 160, 3), 128, np.uint8)).save(scene / "im0.png")
    disparity = np.full((120, 160), 40.0, np.float32)
    with (scene / "disp0GT.pfm").open("wb") as handle:
        handle.write(b"Pf\n160 120\n-1.0\n")
        handle.write(disparity.astype("<f4").tobytes())
    (scene / "calib.txt").write_text(
        "cam0=[500 0 80; 0 500 60; 0 0 1]\ndoffs=0\nbaseline=100\n", encoding="utf-8"
    )
    rendered, observed = calibrate_fp8_scales.refiner_pairs([scene], 3, seed=0)
    assert rendered.shape == observed.shape == (3, 160, 160, 6)
    assert rendered.dtype == np.float32
    assert np.isfinite(rendered).all() and np.isfinite(observed).all()
    assert 0.0 <= observed[..., :3].min() and observed[..., :3].max() <= 1.0
    assert np.abs(observed[..., 3:]).max() < 2.0
