# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The qualification oracle must reject representative incorrect outputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.diffusion_gemma.tests.test_e2e import compare_observations


def _observations(directory, *, broken=None):
    directory.mkdir()
    records = {}
    values = {
        "encoder_key_0": [[0.5, -0.25]],
        "encoder_value_0": [[0.75, 0.125]],
        "step_0_raw_logits": [[1.0, 4.0, 2.0], [3.0, 0.0, 1.0]],
        "step_1_raw_logits": [[-2.0, 0.0, 4.0], [1.0, 2.0, -1.0]],
        "step_0_self_conditioning": [[0.25, 0.75]],
        "step_1_self_conditioning": [[0.5, 1.0]],
    }
    for name, rows in values.items():
        array = np.asarray(rows, dtype="<f4")
        if broken == "zero_conditioning" and name == "step_1_self_conditioning":
            array[:] = 0
        elif broken == "shuffle_logits" and name.endswith("raw_logits"):
            array = array[..., ::-1].copy()
        elif broken == "wrong_encoder_cache" and name == "encoder_key_0":
            array += 0.5
        elif broken == "nonfinite" and name == "step_0_raw_logits":
            array[0, 0] = np.nan
        array.tofile(directory / f"{name}.bin")
        records[name] = {"file": f"{name}.bin", "shape": list(array.shape), "dtype": "float32"}
    document = {
        "cache_readonly": broken != "cache_mutation",
        "decoder_steps": 2,
        "checkpoint_config": {"text_config": {"num_hidden_layers": 1}},
        "tensors": records,
    }
    if broken == "missing_step":
        del document["tensors"]["step_1_self_conditioning"]
    (directory / "metadata.json").write_text(json.dumps(document))
    return directory


def test_reference_oracle_accepts_identical_values(tmp_path):
    native = _observations(tmp_path / "native")
    reference = _observations(tmp_path / "reference")
    result = compare_observations(native, reference)
    assert result["step_1_raw_logits"]["argmax_agreement"] == 1.0


@pytest.mark.parametrize("broken", [
    "zero_conditioning", "shuffle_logits", "wrong_encoder_cache", "nonfinite",
    "cache_mutation", "missing_step",
])
def test_reference_oracle_rejects_incorrect_model_outputs(tmp_path, broken):
    native = _observations(tmp_path / "native", broken=broken)
    reference = _observations(tmp_path / "reference")
    with pytest.raises(AssertionError):
        compare_observations(native, reference)
