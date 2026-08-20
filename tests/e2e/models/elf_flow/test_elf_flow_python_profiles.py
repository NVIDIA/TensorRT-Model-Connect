# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF-owned Python profile contracts."""

from pathlib import Path

from tensorrt_model_connect.python_profiles import default_execution_profiles


def test_elf_profile_pins_official_optimizer_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements_text = (
        repository
        / "python/tensorrt_model_connect/families/elf_flow/python_profile_requirements"
        / "elf_flow.lock.txt"
    ).read_text(encoding="utf-8")
    requirements = set(requirements_text.splitlines())

    assert {
        "absl-py==2.5.0",
        "chex==0.1.90",
        "fsspec==2026.7.0",
        "markdown-it-py==4.2.0",
        "mdurl==0.1.2",
        "msgpack==1.2.1",
        "numpy==1.26.4",
        "optax==0.2.5",
        "protobuf==7.35.1",
        "pygments==2.20.0",
        "rich==15.0.0",
        "scipy==1.17.1",
        "setuptools==83.0.0",
        "toolz==1.1.0",
        "typing-extensions==4.16.0",
        "zipp==4.1.0",
    } <= requirements


def test_elf_reference_profile_pins_inference_only_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements_text = (
        repository
        / "python/tensorrt_model_connect/families/elf_flow/python_profile_requirements"
        / "elf_flow_reference.lock.txt"
    ).read_text(encoding="utf-8")
    requirements = {
        line
        for line in requirements_text.splitlines()
        if line and not line.startswith("#")
    }

    assert requirements == {
        "colorama==0.4.6",
        "einops==0.8.1",
        "huggingface-hub==0.24.7",
        "lxml==6.1.1",
        "portalocker==3.2.0",
        "sacrebleu==2.5.1",
        "tabulate==0.10.0",
        "tokenizers==0.19.1",
        "transformers==4.44.2",
    }
    assert default_execution_profiles(family="elf_flow") == {
        "build": "elf_flow",
        "runtime": "base",
        "reference": "elf_flow_reference",
    }
