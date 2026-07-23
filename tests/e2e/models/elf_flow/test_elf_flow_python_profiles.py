# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_elf_profile_pins_official_optimizer_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/elf_flow/python_profile_requirements"
        / "elf_flow.lock.txt"
    ).read_text(encoding="utf-8")

    assert "chex==0.1.90" in requirements
    assert "numpy==1.26.4" in requirements
    assert "optax==0.2.5" in requirements


def test_elf_reference_profile_pins_official_pytorch_branch_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/elf_flow/python_profile_requirements"
        / "elf_flow_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert "huggingface-hub==0.24.7" in requirements
    assert "muon-optimizer==0.1.0" in requirements
    assert "numpy==1.26.4" in requirements
    assert "tokenizers==0.19.1" in requirements
    assert "transformers==4.44.2" in requirements
    assert default_execution_profiles(family="elf_flow") == {
        "build": "elf_flow",
        "runtime": "base",
        "reference": "elf_flow_reference",
    }
