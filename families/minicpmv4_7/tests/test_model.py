# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle metadata the builder derives from the checkpoint, checked without TensorRT."""

import json

import pytest

from families.minicpmv4_7.model import _eos_ids

_VOCAB_SIZE = 248144


def test_end_of_sequence_ids_must_be_decoder_tokens(tmp_path):
    config = tmp_path / "generation_config.json"
    config.write_text(json.dumps({"eos_token_id": [248046, 248044]}))
    assert _eos_ids(tmp_path, _VOCAB_SIZE) == [248046, 248044]
    config.write_text(json.dumps({"eos_token_id": 248046}))
    assert _eos_ids(tmp_path, _VOCAB_SIZE) == [248046]
    # An id the decoder cannot emit would leave generation to stop only at the token budget.
    for value in (_VOCAB_SIZE, [248046, _VOCAB_SIZE], -1, True, 248044.0, [], None):
        config.write_text(json.dumps({"eos_token_id": value}))
        with pytest.raises(ValueError, match="eos_token_id"):
            _eos_ids(tmp_path, _VOCAB_SIZE)
