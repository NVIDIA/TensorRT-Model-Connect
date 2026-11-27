# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local overrides must not let parity qualify the wrong checkpoint variant."""

import json

import pytest

from . import test_e2e as e2e


def _checkpoint(tmp_path, hf_id):
    if hf_id == "biohub/ESMC-300M":
        layers, hidden, heads, intermediate = 30, 960, 15, 2560
    else:
        layers, hidden, heads, intermediate = 36, 1152, 18, 3072
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "esmc",
                "num_hidden_layers": layers,
                "hidden_size": hidden,
                "num_attention_heads": heads,
                "intermediate_size": intermediate,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("case_name", sorted(e2e.CASES))
@pytest.mark.parametrize("hf_id", ["biohub/ESMC-300M", "biohub/ESMC-600M"])
def test_model_override_must_match_selected_variant(tmp_path, monkeypatch, case_name, hf_id):
    path = _checkpoint(tmp_path, hf_id)
    monkeypatch.setenv("TRTMC_ESMC_MODEL_DIR", str(path))
    _, manifest, _ = e2e.CASES[case_name]
    if manifest["hf_id"] == hf_id:
        assert e2e._model_dir(manifest) == path
    else:
        with pytest.raises(AssertionError, match="does not match selected"):
            e2e._model_dir(manifest)


def test_600m_override_cannot_skip_300m_residual_scale_gate(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, "biohub/ESMC-600M")
    monkeypatch.setenv("TRTMC_ESMC_MODEL_DIR", str(path))
    with pytest.raises(AssertionError, match="does not match selected biohub/ESMC-300M"):
        e2e.test_fp32_gate_rejects_unscaled_residuals("esmc-300m")


def test_override_must_be_an_esmc_checkpoint(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, "biohub/ESMC-300M")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_type"] = "esm"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("TRTMC_ESMC_MODEL_DIR", str(path))
    with pytest.raises(AssertionError, match="does not match selected"):
        e2e._model_dir(e2e.CASES["esmc-300m"][1])
