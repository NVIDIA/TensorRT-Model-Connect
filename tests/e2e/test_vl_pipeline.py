# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E: VL-specific tests for vision-language models."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]


def _is_vl_model(entry):
    return str(entry.get("runtime_strategy") or "").endswith("_vision_language")


@pytest.mark.e2e
def test_vl_vision_only(model_entry, engine_dir):
    """Run diff_vl.py --vision-only for VL bundles."""
    if not _is_vl_model(model_entry):
        pytest.skip(f"{model_entry['name']} is not a VL model")

    diff_vl = PROJECT_DIR / "tools" / "diff_vl.py"
    bundle = model_entry["bundle_path"]

    # Use a placeholder image — vision-only mode tests preprocessor sanity
    result = subprocess.run(
        [sys.executable, str(diff_vl),
         "--bundle", bundle,
         "--vision-only"],
        capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, (
        f"VL vision-only failed for {model_entry['name']}:\n"
        f"{result.stderr}\n{result.stdout}")


@pytest.mark.e2e
def test_vl_generation(model_entry, trtmc_binary, hf_python, ld_library_path):
    """Run VL inference through the C++ binary (requires image + VL model)."""
    if not _is_vl_model(model_entry):
        pytest.skip(f"{model_entry['name']} is not a VL model")

    image_path = model_entry.get("test_image")
    if not image_path:
        pytest.skip(f"No test_image configured for {model_entry['name']}")

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "run", model_entry["bundle_path"],
         "--prompt", "Describe this image.",
         "--image", image_path,
         "--max-new-tokens", "10",
         "--hf-python", str(hf_python)],
        capture_output=True, text=True, timeout=120, env=env)

    assert result.returncode == 0, f"VL inference failed: {result.stderr}"
    assert len(result.stdout.strip()) > 0
