# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_instructions() -> dict[str, str]:
    config = yaml.safe_load(
        (REPO_ROOT / ".coderabbit.yaml").read_text(encoding="utf-8")
    )
    return {
        entry["path"]: entry["instructions"]
        for entry in config["reviews"]["path_instructions"]
    }


def test_coderabbit_preserves_intentional_family_duplication() -> None:
    instructions = _path_instructions()
    instruction = instructions["families/**"].lower()
    assert "intentional isolation" in instruction
    assert "similar" in instruction or "duplicated" in instruction
    assert "cross-family" in instruction or "sibling" in instruction


def test_coderabbit_covers_only_current_shared_paths() -> None:
    instructions = _path_instructions()

    assert "model-agnostic" in instructions["core/**"]
    assert "public core" in instructions["apps/**"]
    assert "timed regions" in instructions["apps/benchmark/**"]
    assert not {
        "python/tensorrt_model_connect/families/**",
        "src/runtime/models/**",
        "tests/e2e/models/**",
        "benchmarks/**",
    } & set(instructions)


def test_coderabbit_enables_semantic_architecture_checks() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".coderabbit.yaml").read_text(encoding="utf-8")
    )
    reviews = config["reviews"]
    checks = {
        check["name"]: check for check in reviews["pre_merge_checks"]["custom_checks"]
    }

    assert set(checks) == {
        "Family ownership boundary",
        "Shared semantic neutrality",
        "Benchmark validation integrity",
        "Shared change blast radius",
    }
    assert all(check["mode"] == "warning" for check in checks.values())
    assert reviews["request_changes_workflow"] is False
    assert reviews["auto_review"]["auto_incremental_review"] is True
    assert reviews["auto_review"]["auto_pause_after_reviewed_commits"] == 0
    assert "REVIEW.md" in reviews["high_level_summary_instructions"]

    guidelines = config["knowledge_base"]["code_guidelines"]
    assert guidelines["enabled"] is True
    assert guidelines["filePatterns"] == [
        {"files": "REVIEW.md", "applyTo": "**/*"}
    ]
    assert (REPO_ROOT / "REVIEW.md").is_file()
