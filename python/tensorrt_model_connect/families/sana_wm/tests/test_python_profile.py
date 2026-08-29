# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SANA-WM-owned Python profile contracts."""

from pathlib import Path

from packaging.version import Version


def test_reference_profile_pins_hub_release_with_diffusers_tree_api() -> None:
    lock = (
        Path(__file__).resolve().parents[1]
        / "python_profile_requirements/sana_wm_reference.lock.txt"
    ).read_text(encoding="utf-8")
    pins = dict(line.split("==", maxsplit=1) for line in lock.splitlines() if line)

    assert Version(pins["huggingface-hub"]) >= Version("1.26.0")
