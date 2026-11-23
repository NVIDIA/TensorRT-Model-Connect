# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intentional source-quality failure for a temporary CI acceptance PR.

This negative probe must never be merged. The undefined name should produce
Ruff F821, fail the CPU gate, and prevent any GPU allocation.
"""


def test_community_ci_rejects_invalid_source():
    return expected_source_quality_probe_failure
