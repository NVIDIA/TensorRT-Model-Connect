# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local bridge to the active E2E registry."""

from tests.e2e_harness.registry import (  # noqa: F401
    register_comparator,
    register_reference,
    register_runner,
)
