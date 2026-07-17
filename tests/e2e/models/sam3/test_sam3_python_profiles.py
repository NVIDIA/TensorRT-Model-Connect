# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3-owned dependency-profile contract tests."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from tensorrt_model_connect.python_profiles import (
    default_execution_profiles,
    load_python_profile_registry,
)


def test_sam3_defaults_build_and_runtime_to_compiled_tracker_profile() -> None:
    assert default_execution_profiles(family="sam3") == {
        "build": "sam3_tracker",
        "runtime": "sam3_tracker",
        "reference": "base",
    }


def test_sam3_tracker_profile_pins_the_native_bridge_abi() -> None:
    profile = load_python_profile_registry()["profiles"]["sam3_tracker"]
    assert profile["system_site_packages"] is True
    package_root = resources.files("tensorrt_model_connect")
    requirements_path = package_root.joinpath(*Path(profile["requirements"]).parts)
    requirements = requirements_path.read_text(encoding="utf-8")

    assert requirements.splitlines() == [
        "apache-tvm-ffi==0.1.12",
        "tensorrt==11.2.0.113",
        "torch==2.12.0+cu130",
        "transformers==5.2.0",
    ]


def test_sam3_tracker_profile_verifier_is_gpu_independent() -> None:
    profile = load_python_profile_registry()["profiles"]["sam3_tracker"]
    package_root = resources.files("tensorrt_model_connect")
    verifier_path = package_root.joinpath(*Path(profile["verification_script_file"]).parts)
    verifier = verifier_path.read_text(encoding="utf-8")

    assert "torch.cuda" not in verifier
    assert 'torch.version.cuda == "13.0"' in verifier
    assert "torch._C._GLIBCXX_USE_CXX11_ABI is True" in verifier
