# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt")
safetensors_numpy = pytest.importorskip("safetensors.numpy")

from ..config import ModelConfig  # noqa: E402
from ..model import _T5Model  # noqa: E402
from .test_quantization import D, _write_checkpoint  # noqa: E402


@pytest.mark.parametrize(
    ("config_updates", "scale"),
    [
        ({}, D**-0.5),
        ({"tie_word_embeddings": True}, D**-0.5),
        ({"tie_word_embeddings": False}, 1.0),
        ({"tie_word_embeddings": True, "scale_decoder_outputs": False}, 1.0),
    ],
)
def test_lm_head_weight_carries_the_decoder_output_scale(
    tmp_path: Path, config_updates: dict, scale: float
) -> None:
    _write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({**json.loads(config_path.read_text(encoding="utf-8")), **config_updates}),
        encoding="utf-8",
    )
    config = ModelConfig.from_dir(tmp_path)

    weights = _T5Model().load_weights(str(tmp_path), config)

    shared = safetensors_numpy.load_file(str(tmp_path / "model.safetensors"))["shared.weight"]
    np.testing.assert_allclose(weights["w_out"], shared.T * np.float32(scale), rtol=1e-6)
