# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding model-owned E2E runner plugin."""

from .runners.qwen_embedding import EmbeddingRunner


runner = EmbeddingRunner()
