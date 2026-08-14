# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fallback contracts for Lance's packed FlashAttention calls."""

from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F

from tests.e2e.models.lance.e2e_plugins.references.lance_image_attention_compat.flash_attn import (
    flash_attn_varlen_func,
)


def test_varlen_fallback_keeps_sequences_isolated() -> None:
    torch.manual_seed(7)
    query = torch.randn(5, 2, 4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cumulative = torch.tensor([0, 2, 5], dtype=torch.int32)

    actual = flash_attn_varlen_func(
        query,
        key,
        value,
        cumulative,
        cumulative,
        3,
        3,
    )
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                query[start:end].transpose(0, 1).unsqueeze(0),
                key[start:end].transpose(0, 1).unsqueeze(0),
                value[start:end].transpose(0, 1).unsqueeze(0),
            )
            .squeeze(0)
            .transpose(0, 1)
            for start, end in ((0, 2), (2, 5))
        ]
    )

    torch.testing.assert_close(actual, expected)


def test_varlen_fallback_uses_bottom_right_causal_alignment() -> None:
    query = torch.tensor([[[1.0, 0.0]]])
    key = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
        ]
    )
    value = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])

    actual = flash_attn_varlen_func(
        query,
        key,
        value,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        1,
        3,
        causal=True,
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=torch.nn.attention.bias.causal_lower_right(1, 3),
    ).squeeze(0).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_varlen_fallback_supports_grouped_query_attention() -> None:
    torch.manual_seed(11)
    query = torch.randn(3, 4, 8)
    key = torch.randn(3, 2, 8)
    value = torch.randn(3, 2, 8)
    cumulative = torch.tensor([0, 3], dtype=torch.int32)

    actual = flash_attn_varlen_func(
        query,
        key,
        value,
        cumulative,
        cumulative,
        3,
        3,
    )
    expected = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        enable_gqa=True,
    ).squeeze(0).transpose(0, 1)

    torch.testing.assert_close(actual, expected)


def test_fallback_exposes_transformers_flash_attention_import_surface() -> None:
    package = (
        "tests.e2e.models.lance.e2e_plugins.references."
    "lance_image_attention_compat.flash_attn"
    )
    flash_attn = importlib.import_module(package)
    bert_padding = importlib.import_module(f"{package}.bert_padding")
    rotary = importlib.import_module(f"{package}.layers.rotary")

    assert callable(flash_attn.flash_attn_func)
    assert callable(flash_attn.flash_attn_varlen_func)
    assert callable(bert_padding.index_first_axis)
    assert callable(bert_padding.pad_input)
    assert callable(bert_padding.unpad_input)
    assert callable(rotary.apply_rotary_emb)


def test_dense_fallback_matches_sdpa() -> None:
    package = importlib.import_module(
        "tests.e2e.models.lance.e2e_plugins.references."
        "lance_image_attention_compat.flash_attn"
    )
    torch.manual_seed(13)
    query = torch.randn(2, 5, 4, 8)
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)

    actual = package.flash_attn_func(query, key, value)
    expected = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected)


def test_rotary_fallback_matches_non_interleaved_contract() -> None:
    rotary = importlib.import_module(
        "tests.e2e.models.lance.e2e_plugins.references."
        "lance_image_attention_compat.flash_attn.layers.rotary"
    )
    values = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]]
    )
    cos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sin = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    actual = rotary.apply_rotary_emb(values, cos, sin)
    expected = torch.tensor(
        [[[[1.0, -4.0, 3.0, 2.0]], [[-7.0, 6.0, 5.0, 8.0]]]]
    )

    torch.testing.assert_close(actual, expected)


def test_padding_fallback_round_trips_valid_tokens() -> None:
    padding = importlib.import_module(
        "tests.e2e.models.lance.e2e_plugins.references."
        "lance_image_attention_compat.flash_attn.bert_padding"
    )
    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)

    unpadded, indices, cumulative, maximum = padding.unpad_input(values, mask)
    restored = padding.pad_input(unpadded, indices, 2, 3)

    torch.testing.assert_close(restored[mask], values[mask])
    torch.testing.assert_close(restored[~mask], torch.zeros_like(restored[~mask]))
    assert cumulative.tolist() == [0, 2, 4]
    assert maximum == 2
