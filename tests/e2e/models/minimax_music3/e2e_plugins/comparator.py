# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-Music3 model-owned E2E comparator plugin."""

from .comparators.text_to_music import plugin

comparator = plugin
