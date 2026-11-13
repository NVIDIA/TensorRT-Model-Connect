# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject nearby model identities and unsupported topology before touching a GPU."""

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import venv

import pytest

from families.molmo2.model import build
from families.molmo2.tests.test_support import metadata
from families.molmo2.tests.test_e2e import _path
from tensorrt_model_connect.build import BuildRequest


def test_reference_python_preserves_virtual_environment(tmp_path, monkeypatch):
    environment = tmp_path / "reference"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    monkeypatch.setenv("TRTMC_MOLMO2_REFERENCE_PYTHON", str(environment / "bin/python"))
    prefix = subprocess.check_output(
        [str(_path("TRTMC_MOLMO2_REFERENCE_PYTHON")), "-c", "import sys; print(sys.prefix)"],
        text=True,
    ).strip()
    assert Path(prefix) == environment


@pytest.mark.parametrize(
    "override",
    [
        {"precision": "fp16"},
        {"quantization": "fp8"},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
        {"task": "text_generation"},
        {"backend": "trt_rtx"},
        {"video_num_frames": 2},
        {"max_batch_size": 2},
    ],
)
def test_unsupported_requests_fail_before_build(override, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(metadata()))
    request = BuildRequest(
        tmp_path, Path("output.bundle"), "molmo2", "vision_language_generation", "bf16"
    )
    error = NotImplementedError if override.get("dynamic_kv_cache") else ValueError
    with pytest.raises(error):
        build(replace(request, **override), None)


@pytest.mark.parametrize(
    "change",
    [
        {"norm_after": True},
        {"qk_norm_type": "olmo"},
        {"use_qk_norm": False},
        {"qkv_bias": True},
        {"rope_scaling": {"rope_type": "linear"}},
        {"hidden_act": "gelu"},
    ],
)
def test_topology_changes_are_rejected(change, tmp_path):
    config = metadata()
    config["text_config"].update(change)
    (tmp_path / "config.json").write_text(json.dumps(config))
    request = BuildRequest(
        tmp_path, Path("output.bundle"), "molmo2", "vision_language_generation", "bf16"
    )
    with pytest.raises(ValueError, match="topology"):
        build(request, None)
