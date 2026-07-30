# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 model-owned E2E runner plugin."""

from .runners.diffusion import plugin

runner = plugin
