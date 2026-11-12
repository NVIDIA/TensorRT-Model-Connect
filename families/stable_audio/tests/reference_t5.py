# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the reference T5 encoder with transformers' own layer norm.

When NVIDIA Apex is importable (NGC PyTorch images ship it), transformers' T5 module binds
``T5LayerNorm`` to ``apex.normalization.FusedRMSNorm`` at import time. That kernel requires its
input in the weight's dtype, while T5 keeps ``wo`` in FP32 (``_keep_in_fp32_modules``) and so
carries an FP32 residual stream into FP16 norms: the official FP16 pipeline then fails, and the
FP32 one runs a different kernel than transformers' T5. ``restore_t5_layer_norm`` puts the
computation of transformers' ``T5LayerNorm`` back on one loaded encoder, without touching
process-wide state that other tests share.

``_T5LayerNorm.forward`` is derived from huggingface/transformers (Apache License 2.0,
Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team):
  https://github.com/huggingface/transformers, release v5.2.0, src/transformers/models/t5/modeling_t5.py
Only the forward of ``T5LayerNorm`` is reproduced.
"""

from __future__ import annotations

import torch


class _T5LayerNorm(torch.nn.Module):
    """The forward of transformers' ``T5LayerNorm`` (transformers 5.2.0, ``modeling_t5.py``)."""

    def __init__(self, weight: torch.nn.Parameter, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def restore_t5_layer_norm(encoder: torch.nn.Module) -> int:
    """Replace the Apex norms of a loaded T5 encoder; return how many were replaced."""
    replaced = 0
    for name, module in list(encoder.named_modules()):
        if not type(module).__module__.startswith("apex."):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = encoder.get_submodule(parent_name) if parent_name else encoder
        setattr(parent, child, _T5LayerNorm(module.weight, float(module.eps)))
        replaced += 1
    assert not any(type(module).__module__.startswith("apex.") for module in encoder.modules())
    return replaced
