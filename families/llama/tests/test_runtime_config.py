# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""_runtime_config()'s generation_config.json override path.

Importing families.llama.model drags in the TensorRT import chain (via
standard_decoder_builder.py), so this lives separately from the
TensorRT-free families.llama.tests.test_config.
"""

from __future__ import annotations

import json

import pytest

trt = pytest.importorskip("tensorrt")

from ..config import ModelConfig  # noqa: E402
from ..model import _runtime_config  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def test_generation_config_overrides_with_full_eos_list(tmp_path) -> None:
    # The checkpoint's own config.json disagrees with generation_config.json
    # (as real checkpoints do) so this only passes if the override path is
    # actually exercised, not the config.json path.
    config = ModelConfig.create_tiny(model_type="llama", eos_token_id=1)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [128001, 128008, 128009]}),
        encoding="utf-8",
    )

    runtime = _runtime_config(tmp_path, config, precision="bf16", max_cache_length=128,
                              decoder_engine_layout="split")

    assert runtime["eos_token_id"] == [128001, 128008, 128009]


def test_generation_config_empty_eos_list_falls_back(tmp_path) -> None:
    # An empty override list must not reach the bundle as `null` or `[]` —
    # the C++ loader rejects an empty eos_token_id list.
    config = ModelConfig.create_tiny(model_type="llama", eos_token_id=1)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": []}),
        encoding="utf-8",
    )

    runtime = _runtime_config(tmp_path, config, precision="bf16", max_cache_length=128,
                              decoder_engine_layout="split")

    assert runtime["eos_token_id"] == [-1]


def test_no_generation_config_keeps_config_json_eos_list(tmp_path) -> None:
    config = ModelConfig.create_tiny(model_type="llama", eos_token_id=[1, 130073])

    runtime = _runtime_config(tmp_path, config, precision="bf16", max_cache_length=128,
                              decoder_engine_layout="split")

    assert runtime["eos_token_id"] == [1, 130073]
