# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from tensorrt_model_connect.families.minimax_h3.checkpoint import (
    load_selected_component_state_dict,
)


def test_selective_loader_supports_unsharded_audio_vae_checkpoint(tmp_path: Path) -> None:
    save_file(
        {
            "decoder.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "encoder.weight": np.ones((4, 4), dtype=np.float32),
        },
        tmp_path / "diffusion_pytorch_model.safetensors",
    )

    selected = load_selected_component_state_dict(tmp_path, ("decoder.weight",))

    assert set(selected) == {"decoder.weight"}
    np.testing.assert_array_equal(
        selected["decoder.weight"].numpy(),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def test_selective_loader_rejects_missing_unsharded_tensor(tmp_path: Path) -> None:
    save_file(
        {"decoder.weight": np.ones((1,), dtype=np.float32)},
        tmp_path / "diffusion_pytorch_model.safetensors",
    )

    with pytest.raises(ValueError, match="missing tensors"):
        load_selected_component_state_dict(tmp_path, ("decoder.bias",))


def test_selective_loader_rejects_ambiguous_unsharded_component(tmp_path: Path) -> None:
    save_file({"a": np.ones((1,), dtype=np.float32)}, tmp_path / "a.safetensors")
    save_file({"b": np.ones((1,), dtype=np.float32)}, tmp_path / "b.safetensors")

    with pytest.raises(ValueError, match="one unsharded file"):
        load_selected_component_state_dict(tmp_path, ("a",))
