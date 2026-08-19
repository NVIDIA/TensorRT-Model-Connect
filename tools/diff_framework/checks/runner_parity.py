# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner parity check — Python TrtRunner vs C++ trtmc binary."""

from diff_framework.registry import register
from diff_framework.protocol import DiffResult, TestContext


@register
class RunnerParityTest:
    name = "runner_parity"
    description = "Cross-validate Python TrtRunner vs C++ trtmc binary"
    requires_bundle = True
    requires_gpu = True

    def run(self, ctx: TestContext) -> DiffResult:
        from test_runner_parity import run_as_diff_test
        return run_as_diff_test(ctx)
