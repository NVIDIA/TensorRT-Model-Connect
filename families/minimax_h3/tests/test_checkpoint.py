# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import pytest
from safetensors.torch import save_file

from families.minimax_h3.checkpoint import load_selected_component_state_dict


def test_selective_loading_supports_an_unindexed_safetensors_checkpoint(tmp_path) -> None:
    save_file(
        {
            "decoder.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "encoder.weight": torch.ones(3, dtype=torch.float32),
        },
        tmp_path / "diffusion_pytorch_model.safetensors",
    )

    state = load_selected_component_state_dict(tmp_path, ("decoder.weight",))

    assert tuple(state) == ("decoder.weight",)
    torch.testing.assert_close(
        state["decoder.weight"], torch.arange(4, dtype=torch.float32).reshape(2, 2)
    )


def test_selective_loading_rejects_duplicate_unindexed_tensor_names(tmp_path) -> None:
    save_file({"decoder.weight": torch.zeros(1)}, tmp_path / "first.safetensors")
    save_file({"decoder.weight": torch.ones(1)}, tmp_path / "second.safetensors")

    with pytest.raises(ValueError, match="Duplicate MiniMax-H3 tensor 'decoder.weight'"):
        load_selected_component_state_dict(tmp_path, ("decoder.weight",))
