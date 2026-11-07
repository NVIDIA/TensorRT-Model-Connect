# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for llama."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("LLaMA", "llama"),
    tasks=("text_generation",),
    default_task="text_generation",
    # BF16 is the only precision that takes the TensorRT native KV route, which
    # bounds each prefill enqueue and therefore supports the full checkpoint
    # context. FP32/FP16 builds use the dense prefill graph instead.
    default_precision="bf16",
)
