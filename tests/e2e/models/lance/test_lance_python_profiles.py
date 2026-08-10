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

    assert "flash-attn==" not in requirements
    assert "huggingface-hub==1.5.0" in requirements
    assert "imageio==2.34.0" in requirements
    assert "numpy==1.26.4" in requirements
    assert "opencv-python-headless==4.8.1.78" in requirements
    assert "scikit-learn==1.5.0" in requirements
    assert "scipy==1.12.0" in requirements
    assert "tokenizers==0.22.2" in requirements
    assert "transformers==5.5.4" in requirements
