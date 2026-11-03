# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from tensorrt_model_connect import family_cli

from families.minimax_h3 import cli


def test_build_omniref_is_a_declared_family_command():
    declaration = family_cli.load_family_cli("minimax_h3")
    command, = [c for c in declaration["commands"] if c["name"] == "build-omniref"]
    assert command["handler"] == "cli:build_omniref"
    assert {a["name"] for a in command["arguments"]} == {"output", "scratch", "bridge", "kernel_library",
                                                          "denoiser_workspace_gib", "base", "student"}
    workspace, = [a for a in command["arguments"] if a["name"] == "denoiser_workspace_gib"]
    assert workspace["default"] == 24  # what the released FastH3 bundle's reuse receipt records


def test_build_omniref_needs_both_libraries_before_any_download(tmp_path: Path):
    bridge = tmp_path / "bridge.so"
    bridge.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="VSA kernel"):
        cli.build_omniref(output=tmp_path / "out.bundle", scratch=tmp_path / "scratch",
                          bridge=bridge, kernel_library=tmp_path / "missing.so")
    with pytest.raises(FileNotFoundError, match="TVM-FFI bridge"):
        cli.build_omniref(output=tmp_path / "out.bundle", scratch=tmp_path / "scratch",
                          bridge=tmp_path / "missing.so", kernel_library=bridge)
    assert not (tmp_path / "scratch").exists()
