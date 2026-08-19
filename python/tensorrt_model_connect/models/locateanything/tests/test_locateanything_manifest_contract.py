# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest, load_model_manifest


def test_locateanything_manifest_declares_hf_image_text_to_text_contract() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "locateanything-3b.json"
    case = load_manifest(manifest_path)

    assert case.hf_id == "nvidia/LocateAnything-3B"
    assert case.task_strategy == "vision_language_generation"
    assert case.user_contract == "image-text-to-text"
    assert case.inputs["prompt"] == (
        "Locate a single instance that matches the following description: white vehicle."
    )


def test_locateanything_manifest_covers_box_and_point_contracts() -> None:
    manifest_path = Path(__file__).with_name("manifests") / "locateanything-3b.json"
    model = load_model_manifest(manifest_path)

    assert [(case.name, case.inputs["prompt"]) for case in model.testcases] == [
        (
            "locateanything-3b",
            "Locate a single instance that matches the following description: white vehicle.",
        ),
        ("locateanything-3b-point", "Point to: white vehicle."),
    ]
