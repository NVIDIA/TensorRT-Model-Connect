# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF-owned Python profile contracts."""

from pathlib import Path


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
