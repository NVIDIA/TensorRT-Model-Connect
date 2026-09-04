# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR sequence-prefill limits shared by its engine builders."""

# The model contributes 145 image tokens before a short text prompt. Keeping
# the dynamic profile bounded avoids TensorRT tactic exploration over a
# 4096-row, 64-expert MoE graph; longer prompts retain the runtime's explicit
# token-by-token path.
MAX_SEQUENCE_PREFILL_LENGTH = 256


def sequence_prefill_profile_lengths(max_cache_length: int) -> tuple[int, int]:
    """Return the preferred and maximum sequence-prefill profile lengths."""
    max_prefill_length = min(max_cache_length, MAX_SEQUENCE_PREFILL_LENGTH)
    return min(64, max_prefill_length), max_prefill_length
