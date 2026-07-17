# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-owned E2E runner plugin."""

from .runners.diffusion import Wan22TI2VDiffusionRunner

runner = Wan22TI2VDiffusionRunner()
