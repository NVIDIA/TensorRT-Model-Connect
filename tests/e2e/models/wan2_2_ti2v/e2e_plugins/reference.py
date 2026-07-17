# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-owned E2E reference plugin."""

from .references.invariant_only import Wan22InvariantReference

reference = Wan22InvariantReference()
