# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-ASR family: a Whisper-style audio encoder feeding a Llama decoder."""

from .plugin import plugin

__all__ = ["plugin"]
