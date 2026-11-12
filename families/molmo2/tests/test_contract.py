# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject nearby model identities and unsupported topology before touching a GPU."""

from dataclasses import replace
from pathlib import Path
import subprocess
import venv

import pytest

from families.molmo2.model import validate_request
from families.molmo2.support import describe
from families.molmo2.tests.test_e2e import _path
from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


def test_reference_python_preserves_virtual_environment(tmp_path, monkeypatch):
    environment = tmp_path / "reference"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    monkeypatch.setenv("TRTMC_MOLMO2_REFERENCE_PYTHON", str(environment / "bin/python"))
    prefix = subprocess.check_output(
        [str(_path("TRTMC_MOLMO2_REFERENCE_PYTHON")), "-c", "import sys; print(sys.prefix)"],
        text=True,
    ).strip()
    assert Path(prefix) == environment


def metadata():
    return {
        "model_type": "molmo2",
        "architectures": ["Molmo2ForConditionalGeneration"],
        "text_config": {"qk_norm_type": "qwen3", "use_qk_norm": True},
        "vit_config": {"image_default_input_size": [378, 378]},
    }


def test_exact_ownership():
    config = metadata()
    family, support = resolve_family(ModelMetadata(config, {}))
    assert family == "molmo2" and support.default_task == "vision_language_generation"
    for kind, architecture in [
        ("molmo", "MolmoForCausalLM"),
        ("qwen3", "Qwen3ForCausalLM"),
        ("molmo2", "OtherModel"),
    ]:
        assert (
            describe(ModelMetadata({"model_type": kind, "architectures": [architecture]}, {}))
            is None
        )


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
def test_unsupported_requests_fail_before_build(override):
    request = BuildRequest(
        Path("checkpoint"), Path("output.bundle"), "molmo2", "vision_language_generation", "bf16"
    )
    with pytest.raises(ValueError):
        validate_request(replace(request, **override), metadata())


@pytest.mark.parametrize(
    "change",
    [
        {"norm_after": True},
        {"qk_norm_type": "olmo"},
        {"use_qk_norm": False},
        {"qkv_bias": True},
        {"rope_scaling": {"rope_type": "linear"}},
    ],
)
def test_topology_changes_are_rejected(change):
    config = metadata()
    config["text_config"].update(change)
    request = BuildRequest(
        Path("checkpoint"), Path("output.bundle"), "molmo2", "vision_language_generation", "bf16"
    )
    with pytest.raises(ValueError, match="topology"):
        validate_request(request, config)
