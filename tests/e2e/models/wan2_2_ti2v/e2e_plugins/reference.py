# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-owned E2E reference plugins."""

from .references.invariant_only import plugin as invariant_reference
from .references.official_wan import plugin as official_wan_reference

reference = (invariant_reference, official_wan_reference)
