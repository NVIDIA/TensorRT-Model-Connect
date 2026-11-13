# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

from tools import model_benchmark


def test_aiperf_subcommand_delegates_to_the_aiperf_qualification(tmp_path: Path) -> None:
    arguments = model_benchmark.parser().parse_args([
        "aiperf", "--environment", str(tmp_path / "gb300.yaml"), "--aiperf-python", "/venv/bin/python",
        "--out-root", str(tmp_path / "out"), "--model", "qwen3-0.6b-fp16", "--shard", "0/2",
    ])
    command, env = model_benchmark.aiperf_command(arguments)
    assert command[:4] == ["/venv/bin/python", "-m", "trtmc_aiperf_qual", "run-all"]
    assert command[command.index("--profile") + 1] == "qwen3-0.6b-fp16" and command[-1] == "0/2"
    assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("apps/aiperf_qual")
