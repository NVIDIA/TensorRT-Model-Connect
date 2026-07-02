# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest


@pytest.mark.parametrize(
    ("manifest_name", "hf_id", "user_contract", "task_mode"),
    [
        ("qwen-image.json", "Qwen/Qwen-Image", "text-to-image", None),
        ("qwen-image-2512.json", "Qwen/Qwen-Image-2512", "text-to-image", None),
        (
            "qwen-image-edit-2511.json",
            "Qwen/Qwen-Image-Edit-2511",
            "image-to-image",
            "edit",
        ),
    ],
)
def test_qwen_image_manifest_declares_hf_image_contract(
    manifest_name: str,
    hf_id: str,
    user_contract: str,
    task_mode: str | None,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    case = load_manifest(manifest_path)

    assert case.hf_id == hf_id
    assert case.task_strategy == "diffusion_media_generation"
    assert case.user_contract == user_contract
    assert case.metadata.get("task_mode") == task_mode
