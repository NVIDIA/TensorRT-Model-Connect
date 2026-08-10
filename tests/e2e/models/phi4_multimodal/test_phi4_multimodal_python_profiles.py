# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-4 Multimodal-owned Python profile contracts."""

from pathlib import Path


def test_phi4_multimodal_profile_closes_chat_template_dependencies() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/phi4_multimodal/"
        "python_profile_requirements/phi4_multimodal.lock.txt"
    ).read_text(encoding="utf-8")

    assert "jinja2==3.1.6" in requirements.splitlines()
