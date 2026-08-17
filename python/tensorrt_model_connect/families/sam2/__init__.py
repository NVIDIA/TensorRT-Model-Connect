# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2.1 native complete-bundle build family.

The runtime strategy is registered, but production admission requires the
explicit qualified API and an external record authorized by a compiled pin.
The initial production pin set is deliberately empty and therefore fails
closed.
"""

from .plugin import plugin


__all__ = ["plugin"]
