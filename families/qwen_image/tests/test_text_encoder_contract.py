# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL text-encoder attention precision contract."""

from __future__ import annotations

import inspect

from families.qwen_image import qwen25_vl_text_encoder_builder as text_builder


def test_attention_scores_and_softmax_run_in_fp32() -> None:
    # Qwen2.5-VL's q/k biases give attention logits in the thousands. The
    # reference SDPA keeps them in FP32; BF16 logits that large lose the
    # softmax (layer-0 errors of 14-25% on real prompts).
    source = inspect.getsource(text_builder.build_qwen25vl_text_encoder_engine)
    call = source[source.index("graph_ops.add_attention_from_rows(") :]
    call = call[: call.index('tag=f"{prefix}.attn"')]
    assert "fp32_scores=True" in call
    assert "mask=attn_mask_fp32" in call

