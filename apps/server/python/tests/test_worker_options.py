# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from trtmc_server.cli import packaged_runtime_root
from trtmc_server.worker import WorkerLoadOptions


def test_runtime_root_is_optional_and_override_is_preserved() -> None:
    assert WorkerLoadOptions().argv() == []
    assert WorkerLoadOptions(runtime_root="/opt/runtime").argv() == [
        "--runtime-root",
        "/opt/runtime",
    ]


def test_packaged_runtime_root_is_derived_from_control_plane(tmp_path: Path) -> None:
    control_plane = tmp_path / "trtmc_server/cli.py"
    control_plane.parent.mkdir()
    control_plane.write_text("")
    runtime_root = tmp_path / "tensorrt_model_connect/bin"
    runtime_root.mkdir(parents=True)

    assert packaged_runtime_root(control_plane) is None
    for name in ("trtmc-server", "libtrtmc_runtime.so", "libtrtmc_backend_trt.so"):
        (runtime_root / name).write_text("")

    assert packaged_runtime_root(control_plane) == runtime_root
