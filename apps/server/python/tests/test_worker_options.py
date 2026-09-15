# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from trtmc_server.worker import WorkerLoadOptions


def test_runtime_root_is_optional_and_override_is_preserved() -> None:
    assert WorkerLoadOptions().argv() == []
    assert WorkerLoadOptions(runtime_root="/opt/runtime").argv() == [
        "--runtime-root",
        "/opt/runtime",
    ]
