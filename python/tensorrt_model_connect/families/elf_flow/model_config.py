# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable ELF Flow configuration-adapter entry point."""

from .model.components.config import config_from_dir

__all__ = ["config_from_dir"]
