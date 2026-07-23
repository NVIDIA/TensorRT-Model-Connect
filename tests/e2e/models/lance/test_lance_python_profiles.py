# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance-owned Python profile contracts."""

from pathlib import Path

def test_lance_reference_profile_pins_upstream_transformers_stack() -> None:
    repository = Path(__file__).resolve().parents[4]
    requirements = (
        repository
        / "python/tensorrt_model_connect/families/lance/python_profile_requirements"
        / "lance_reference.lock.txt"
    ).read_text(encoding="utf-8")

    assert "flash-attn==2.8.3" in requirements
    assert "huggingface-hub==0.29.1" in requirements
    assert "imageio==2.34.0" in requirements
    assert "numpy==1.26.4" in requirements
    assert "opencv-python-headless==4.7.0.72" in requirements
    assert "tokenizers==0.21.4" in requirements
    assert "transformers==4.49.0" in requirements
