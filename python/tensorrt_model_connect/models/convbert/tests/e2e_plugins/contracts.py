# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local E2E contract aliases.

Contracts remain the stable harness API; concrete runners/references/comparators
are owned by the model package.
"""

from tests.e2e_harness.contracts import *  # noqa: F401,F403
