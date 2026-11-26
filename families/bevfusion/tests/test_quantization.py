# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest

from families.bevfusion.model import Weights, build_engine
from families.bevfusion.calibrate_fp8_scales import FIRST_SEED, SAMPLES, calibrate, save_checkpoint
from families.bevfusion.quantization import (
    CONVOLUTIONS,
    DEFAULT_EXCLUSIONS,
    load_scales,
    quantize_weight,
)


@pytest.mark.parametrize(
    "options",
    [
        {"first_seed": -1},
        {"samples": 0},
        {"exclude": ["unknown"]},
        {"exclude": list(CONVOLUTIONS)},
    ],
)
def test_calibration_rejects_invalid_input_before_loading_modelopt(tmp_path, options):
    with pytest.raises(ValueError, match="calibration requires"):
        calibrate(tmp_path, tmp_path / "calibrated/bevfusion-seg.pth", **options)


def test_default_recipe_has_disjoint_calibration_and_validation_seeds():
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "tests/manifests/bevfusion-bev-fp8.json").read_text())
    calibration_seeds = set(range(FIRST_SEED, FIRST_SEED + SAMPLES))
    assert calibration_seeds.isdisjoint(manifest["testcases"][0]["seeds"])
    assert len(CONVOLUTIONS) - len(DEFAULT_EXCLUSIONS) == 3


def test_fp8_weight_layout_and_per_output_channel_scales():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    weight = np.array([[0, 448, -448, 224], [0, 224, -224, 112]], np.float32).reshape(2, 1, 1, 4)
    packed, scales = quantize_weight(weight)
    assert packed.shape == weight.shape and packed.dtype == np.uint8 and packed.flags.c_contiguous
    np.testing.assert_array_equal(scales, [1, 0.5])
    decoded = packed.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    np.testing.assert_array_equal(decoded * scales[:, None, None, None], weight)
    packed, scales = quantize_weight(np.zeros((2, 1, 1, 1), np.float32))
    assert (packed == 0).all() and (scales > 0).all()


@pytest.mark.parametrize("weight", [np.ones((2, 3)), np.full((1, 1, 1, 1), np.nan)])
def test_fp8_weight_rejects_invalid_operands(weight):
    with pytest.raises(ValueError, match="finite OIHW"):
        quantize_weight(weight)


@pytest.fixture
def calibration(tmp_path):
    torch = pytest.importorskip("torch")
    weights = {"fuser.0.weight": torch.ones(2), "decoder.backbone.blocks.0.0.weight": torch.ones(1)}
    document = {
        "version": 1,
        "calibration": {},
        "layers": dict.fromkeys(CONVOLUTIONS, 0.1),
    }
    return tmp_path / "bevfusion-seg.pth", weights, document


def test_fp8_snapshot_couples_decoder_weights_and_scales(calibration):
    checkpoint, weights, document = calibration
    torch = pytest.importorskip("torch")
    document["layers"][CONVOLUTIONS[0]] = None
    save_checkpoint(checkpoint, weights, document)
    scales, _ = load_scales(Weights(checkpoint))
    assert scales[CONVOLUTIONS[0]] is None and len(scales) == 16
    # A decoder-only source mutation cannot pair new weights with cached scales.
    name = "decoder.backbone.blocks.0.0.weight"
    weights[name].neg_()
    source = checkpoint.parent / "source.pth"
    torch.save({"state_dict": weights}, source)
    (source.parent / "fp8_activation_scales.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="calibrate_fp8_scales.py checkpoint"):
        load_scales(Weights(source))
    selected = Weights(checkpoint)
    torch.testing.assert_close(selected.values[name], torch.ones(1))
    assert load_scales(selected)[0] == scales


def test_calibration_preserves_source_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        calibrate(tmp_path, tmp_path / "bevfusion-seg.pth")


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True, "0.1", 1e-10, 1e10])
def test_fp8_scale_table_rejects_invalid_scales(calibration, invalid):
    checkpoint, weights, document = calibration
    document["layers"][CONVOLUTIONS[0]] = invalid
    save_checkpoint(checkpoint, weights, document)
    with pytest.raises(ValueError, match="representable"):
        load_scales(Weights(checkpoint))


def test_fp8_scale_table_rejects_missing_extra_and_empty_recipes(calibration):
    checkpoint, weights, document = calibration
    del document["layers"][CONVOLUTIONS[0]]
    save_checkpoint(checkpoint, weights, document)
    with pytest.raises(ValueError, match="exactly"):
        load_scales(Weights(checkpoint))
    document["layers"]["unexpected"] = 0.1
    save_checkpoint(checkpoint, weights, document)
    with pytest.raises(ValueError, match="exactly"):
        load_scales(Weights(checkpoint))
    document["layers"] = dict.fromkeys(CONVOLUTIONS)
    save_checkpoint(checkpoint, weights, document)
    with pytest.raises(ValueError, match="at least one"):
        load_scales(Weights(checkpoint))


@pytest.mark.parametrize("precision,quantization", [("fp32", "fp8"), ("fp16", "int8")])
def test_fp8_rejects_invalid_build_options_before_loading_weights(
    tmp_path, precision, quantization
):
    pytest.importorskip("tensorrt")
    with pytest.raises(ValueError, match="FP8 requires|only fp8"):
        build_engine(tmp_path / "missing", precision=precision, quantization=quantization)
