# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


FAMILY_ROOT = Path(__file__).resolve().parents[1]


def test_official_reference_declares_its_real_dependencies() -> None:
    requirements = {
        line.strip()
        for line in (FAMILY_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "flash-attn==2.8.3" in requirements
    assert not any(requirement.startswith("decord") for requirement in requirements)
    assert not (Path(__file__).with_name("reference_compat")).exists()
    for path in (
        FAMILY_ROOT / "tests/official_reference.py",
        FAMILY_ROOT / "tests/vision_oracle.py",
    ):
        assert "sdpa" not in path.read_text(encoding="utf-8").lower()
