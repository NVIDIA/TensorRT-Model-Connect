# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the pinned MiniMax-H3 reference execution profile."""

from importlib.metadata import version

from huggingface_hub import get_cached_repo_tree


assert version("huggingface-hub") == "1.23.0"
assert callable(get_cached_repo_tree)
