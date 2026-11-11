# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free checks of the flux3 build-time layout against the reference code."""

from __future__ import annotations

import numpy as np
import pytest

from families.flux3 import bundle_config as bc
from families.flux3 import rope


def test_text_window_rounds_up_to_the_pad_multiple_and_rejects_overflow() -> None:
    assert [bc.text_window(n, 1280) for n in (1, 79, 80, 81, 1280)] == [80, 80, 80, 160, 1280]
    with pytest.raises(ValueError):
        bc.text_window(1281, 1280)
    with pytest.raises(ValueError):
        bc.text_window(0, 1280)


@pytest.mark.parametrize("value", [79, 8240, 100])
def test_text_budget_must_be_a_multiple_of_80_up_to_8192(value) -> None:
    with pytest.raises(ValueError):
        bc.text_token_budget(value)


def test_default_geometry_and_budget() -> None:
    assert bc.image_geometry(None, None) == (1024, 1024)
    assert bc.text_token_budget(None) == 4000
    for height, width in ((1000, 1024), (48, 1024), (4096, 4096)):
        with pytest.raises(ValueError):
            bc.image_geometry(height, width)


def test_image_ids_match_the_reference() -> None:
    before_denoise = pytest.importorskip("diffusers.modular_pipelines.flux3.before_denoise")
    expected = before_denoise._prepare_image_ids(5, 7, 0.5).numpy()
    np.testing.assert_array_equal(rope.image_ids(5, 7, 0.5), expected)


def test_interleaved_tables_match_flux3_pos_embed() -> None:
    torch = pytest.importorskip("torch")
    transformer = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    ids = np.concatenate([rope.text_ids(160), rope.image_ids(8, 12, 0.5)])
    cos, sin = transformer.Flux3PosEmbed(theta=10000, axes_dim=[32, 32, 32, 32])(
        torch.from_numpy(ids)
    )
    ours_cos, ours_sin = rope.interleaved_tables(ids, (32, 32, 32, 32), 10000.0)
    np.testing.assert_array_equal(ours_cos, cos.numpy())
    np.testing.assert_array_equal(ours_sin, sin.numpy())


def test_rotation_vectors() -> None:
    x = np.arange(8, dtype=np.float32)
    index, sign = rope.interleaved_rotation(8)
    np.testing.assert_array_equal(x[index] * sign, [-1, 0, -3, 2, -5, 4, -7, 6])
    index, sign = rope.rotate_half_rotation(8)
    np.testing.assert_array_equal(x[index] * sign, [-4, -5, -6, -7, 0, 1, 2, 3])


def test_text_encoder_tables_match_qwen3_vl_rotary_embedding() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    config = transformers.Qwen3VLTextConfig(
        hidden_size=256,
        num_attention_heads=2,
        head_dim=128,
        rope_theta=5000000,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        },
    )
    positions = torch.arange(300)[None].masked_fill(torch.arange(300)[None] >= 250, 1)
    cos, sin = Qwen3VLTextRotaryEmbedding(config)(torch.zeros(1, 300, 256), positions)
    table_cos, table_sin = rope.text_encoder_tables(300, 128, 5000000.0)
    # NumPy's and PyTorch's float32 pow may differ by one ulp in an inverse frequency; scaled by the
    # position (< 300 here) that stays below 4e-5, far below the BF16 step of the tables the engine uses.
    np.testing.assert_allclose(table_cos[positions[0].numpy()], cos[0].numpy(), rtol=0, atol=4e-5)
    np.testing.assert_allclose(table_sin[positions[0].numpy()], sin[0].numpy(), rtol=0, atol=4e-5)
