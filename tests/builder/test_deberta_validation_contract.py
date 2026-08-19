# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation precision contract tests for DeBERTa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.validation import engine as validation_engine


_MANIFEST_PATH = (
    validation_engine.REPO_ROOT
    / "python"
    / "tensorrt_model_connect"
    / "models"
    / "deberta"
    / "tests"
    / "manifests"
    / "deberta-base.json"
)


def test_deberta_sts_parity_uses_aligned_fp32_without_loosening_gates(
    tmp_path: Path,
) -> None:
    raw_manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    model = validation_engine.manifest_record(_MANIFEST_PATH)
    suite = validation_engine.resolve_suite_for_model(
        validation_engine.suite_by_id(
            validation_engine.load_suites(),
            "stsbenchmark_encoder_embedding_parity",
        ),
        model,
    )
    config = validation_engine.effective_validation_config(suite, model)

    assert raw_manifest["precision"] == "fp16"
    assert raw_manifest["testcases"][0]["reference_precision"] == "fp32"
    assert config["comparison_precision"] == "fp32"

    comparison_model = validation_engine.apply_comparison_precision(model, config)
    work_dir = tmp_path / "deberta-sts"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "sts_pair_jsonl",
                "task_eval": config,
            }
        ),
        encoding="utf-8",
    )
    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        comparison_model,
        work_dir,
    )

    assert model["precision"] == "fp16"
    assert comparison_model["precision"] == "fp32"
    assert contract == {
        "trtmc_base_precision": "fp32",
        "trtmc_quantization": "none",
        "reference_precision": "fp32",
        "reference_dtype": "float32",
        "comparison": "aligned",
    }
    assert suite["gates"] == {
        "min_vector_cosine": 0.999,
        "min_vector_pass_rate": 1.0,
        "max_pair_cosine_abs_delta": 0.02,
    }
