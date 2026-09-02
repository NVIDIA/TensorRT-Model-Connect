# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Embedding model-owned E2E comparator plugin."""

from .comparators.qwen_embedding import EmbeddingComparator


comparator = EmbeddingComparator()
