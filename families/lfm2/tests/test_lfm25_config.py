# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2.5-2.6B configuration contract at revision 654f9463ce32b05d0429d76fe1f580b27d4c1ac0."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.lfm2.config import validate_dense_lfm2_config


def checkpoint_config():
    path = Path(__file__).parent / "fixtures" / "lfm25-2.6b-config.json"
    return json.loads(path.read_text())


def test_lfm25_2_6b_checkpoint_geometry_and_rope():
    resolved = validate_dense_lfm2_config(SimpleNamespace(raw=checkpoint_config()))
    assert resolved.rope_theta == 10_000_000.0
    assert resolved.max_position_embeddings == 131072
    assert resolved.vocab_size == 128000
    assert resolved.intermediate_size == 10752
    assert resolved.bos_token_id == 124894
    assert resolved.eos_token_id == 124900
    assert resolved.num_attention_layers == 8
    assert resolved.num_conv_layers == 22


@pytest.mark.parametrize("field", ["rope_theta", "theta"])
def test_lfm25_rejects_conflicting_legacy_rope(field):
    raw = checkpoint_config()
    raw[field] = 1_000_000.0
    with pytest.raises(ValueError, match="must agree"):
        validate_dense_lfm2_config(SimpleNamespace(raw=raw))


@pytest.mark.parametrize(
    "parameters", [[], {}, {"rope_type": "linear", "rope_theta": 10_000_000, "factor": 4}]
)
def test_lfm25_rejects_unimplemented_rope(parameters):
    raw = checkpoint_config()
    raw["rope_parameters"] = parameters
    with pytest.raises(ValueError, match="rope_parameters"):
        validate_dense_lfm2_config(SimpleNamespace(raw=raw))
