# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for GPT-OSS model-owned reference precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.models.gpt_oss.e2e_plugins.references.hf_transformers import (
    _torch_dtype_for_case,
)
from tests.e2e_harness.manifest_loader import load_model_manifest
from tools.validation import catalog as validation_catalog
from tools.validation import engine as validation_engine


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "tests" / "e2e" / "models" / "gpt_oss" / "manifests"
GPT_OSS_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"


@pytest.mark.parametrize(
    ("manifest_name", "candidate_precision"),
    [
        ("gpt-oss-20b.json", "fp16"),
        ("gpt-oss-20b-l0.json", "fp16"),
    ],
)
def test_gpt_oss_e2e_reference_uses_bf16(
    manifest_name: str,
    candidate_precision: str,
) -> None:
    model = load_model_manifest(MANIFEST_DIR / manifest_name)

    assert model.hf_revision == GPT_OSS_REVISION
    for case in model.testcases:
        assert case.metadata.get("precision", "fp32") == candidate_precision
        assert case.metadata["reference_precision"] == "bf16"
        assert _torch_dtype_for_case(case) == "torch.bfloat16"


def _validation_model() -> dict[str, Any]:
    return validation_catalog.manifest_record(MANIFEST_DIR / "gpt-oss-20b.json")


def _work_dir(tmp_path: Path, task_config: dict[str, Any] | None = None) -> Path:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    manifest: dict[str, Any] = {"dataset_kind": "mmlu_five_shot_json"}
    if task_config is not None:
        manifest["task_eval"] = task_config
    (work_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return work_dir


def test_gpt_oss_validation_declares_model_owned_bf16(
    tmp_path: Path,
) -> None:
    model = _validation_model()
    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        model,
        _work_dir(tmp_path),
    )

    assert model["precision"] == "fp16"
    assert model["hf_revision"] == GPT_OSS_REVISION
    assert contract == {
        "trtmc_base_precision": "fp16",
        "trtmc_quantization": "none",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "reference_defined",
    }
    assert model["task_eval"]["allow_reference_precision_mismatch"] is True


def test_gpt_oss_reference_subprocess_receives_bfloat16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}

    class Result:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: Any) -> Result:
        captured["command"] = command
        return Result()

    monkeypatch.setattr(validation_engine.subprocess, "run", fake_run)
    args = argparse.Namespace(
        hf_python="/opt/reference/bin/python",
        reference_cache_dir="",
        reference_cache_identity="",
        hf_dtype="auto",
        hf_device="cuda",
        hf_device_map="",
        hf_attn_impl="",
        trust_remote_code=False,
        local_files_only=True,
        do_sample=False,
        apply_chat_template=False,
        force_hf=True,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        seed=None,
        elf_reference_repo="",
    )

    validation_engine.run_hf_reference_subprocess(
        args,
        _validation_model(),
        _work_dir(tmp_path),
    )

    command = captured["command"]
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--model-revision") + 1] == GPT_OSS_REVISION


def test_gpt_oss_rejects_conflicting_float16_reference(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="--hf-dtype fp16 conflicts with task_eval.reference_precision bf16",
    ):
        validation_engine.resolve_reference_precision_contract(
            argparse.Namespace(hf_dtype="float16"),
            _validation_model(),
            _work_dir(tmp_path),
        )
