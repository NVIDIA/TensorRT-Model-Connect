# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported checkpoint geometry and complete speech stopping semantics."""

import json
from pathlib import Path

import pytest

from families.phonon.model import _validate_architecture
from families.phonon.support import describe
from families.phonon.tests.validation import assert_complete_eos
from tensorrt_model_connect.model_support import ModelMetadata


@pytest.fixture
def config():
    # Public metadata from gradium/phonon-31m-en at 92741c5b07d085ff840e26fea8ce43110b87852f.
    return json.loads((Path(__file__).parent / "fixtures/config.json").read_text())


def test_supported_checkpoint(config):
    _validate_architecture(config)
    assert describe(ModelMetadata(config=config, model_index={})).default_task == "text_to_speech"


@pytest.mark.parametrize("value", ["tanh", "relu", None])
def test_rejects_unimplemented_flow_activation(config, value):
    config["flow_lm"]["gelu"] = value
    with pytest.raises(ValueError, match="gelu"):
        _validate_architecture(config)
    assert describe(ModelMetadata(config=config, model_index={})) is None


def test_requires_flow_activation(config):
    del config["flow_lm"]["gelu"]
    with pytest.raises(ValueError, match="gelu"):
        _validate_architecture(config)


def test_preserves_timestep_normalization_check(config):
    config["flow_lm"]["time_rms_norm"] = "layer_norm"
    with pytest.raises(ValueError, match="time_rms_norm"):
        _validate_architecture(config)


@pytest.mark.parametrize("section", ["mimi", "flow_head_config", "flow_lm"])
def test_rejects_missing_geometry(config, section):
    del config[section]
    with pytest.raises(ValueError):
        _validate_architecture(config)


def test_complete_short_utterance_tail():
    assert assert_complete_eos([-5, 5, -5, -5, -5], 4, 0, 3) == 1


def test_complete_long_utterance_tail():
    assert assert_complete_eos([-5, 5, -5], 4, 0, 1) == 1


def test_ignores_eos_before_minimum_frame():
    assert assert_complete_eos([5, 5, -5, 6, 0], 4, 3, 1) == 3


def test_threshold_crossing_is_strict():
    assert assert_complete_eos([4, 4.01, 0], 4, 0, 1) == 1


@pytest.mark.parametrize("trace", [[-5, -5, 5], [-5, 5, -5], [-5, 5, -5, -5]])
def test_frame_budget_cannot_truncate_eos_tail(trace):
    with pytest.raises(AssertionError, match="complete its EOS tail"):
        assert_complete_eos(trace, 4, 0, 3)


def test_must_stop_at_the_first_eos_countdown():
    with pytest.raises(AssertionError, match="complete its EOS tail"):
        assert_complete_eos([-5, 5, 0, 0, 0, 0], 4, 0, 3)


def test_requires_eligible_eos():
    with pytest.raises(AssertionError, match="eligible EOS crossing"):
        assert_complete_eos([5, 0, 0], 4, 3, 1)


def test_rejects_nonfinite_eos():
    with pytest.raises(AssertionError, match="nonfinite"):
        assert_complete_eos([float("nan"), 5, 0], 4, 0, 1)
