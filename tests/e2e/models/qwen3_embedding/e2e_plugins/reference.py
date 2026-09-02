# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding model-owned Hugging Face reference plugin."""

from .references.hf_transformers import HfTransformersReference


reference = HfTransformersReference()
