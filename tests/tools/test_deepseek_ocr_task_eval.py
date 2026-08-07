# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation precision tests for DeepSeek-OCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.reference import transformers_vlm
from tools.validation import catalog as validation_catalog
from tools.validation import engine as validation_engine


_MANIFEST_PATH = (
    validation_catalog.REPO_ROOT
    / "tests"
    / "e2e"
    / "models"
    / "deepseek_ocr"
    / "manifests"
    / "deepseek-ocr.json"
)


def _write_vlm_inputs(tmp_path: Path, *, dtype: str) -> argparse.Namespace:
    prompts = tmp_path / "prompts.jsonl"
    answers = tmp_path / "answers.json"
    manifest = tmp_path / "manifest.json"
    prompts.write_text(
        json.dumps(
            {
                "sample_id": "ocrbench_v2_000000",
                "prompt": "Read the image.",
                "images": ["/dataset/image.jpg"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps({"requests": [{"answer": "recognized text"}]}),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({"generation": {"max_new_tokens": 32}}),
        encoding="utf-8",
    )
    return transformers_vlm.build_parser().parse_args(
        [
            "--model",
            "deepseek-ai/DeepSeek-OCR-2",
            "--prompts",
            str(prompts),
            "--answers",
            str(answers),
            "--manifest",
            str(manifest),
            "--predictions",
            str(tmp_path / "predictions.json"),
            "--raw-output",
            str(tmp_path / "raw.jsonl"),
            "--dtype",
            dtype,
            "--device",
            "cpu",
            "--trust-remote-code",
            "--local-files-only",
        ]
    )


def test_deepseek_ocr_declares_model_owned_bf16_reference(
    tmp_path: Path,
) -> None:
    raw_manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    model = validation_catalog.manifest_record(_MANIFEST_PATH)
    suite = validation_catalog.resolve_suite_for_model(
        validation_engine.suite_by_id(
            validation_engine.load_suites(),
            "ocrbench_v2_unified",
        ),
        model,
    )
    config = validation_engine.effective_validation_config(suite, model)

    assert raw_manifest["precision"] == "fp16"
    assert raw_manifest["fp32_layers"] == [6, 7, 8, 9, 10, 11, 12]
    assert raw_manifest["testcases"][0]["reference_precision"] == "bf16"
    assert config["reference_precision"] == "bf16"
    assert config["allow_reference_precision_mismatch"] is True

    work_dir = tmp_path / "deepseek-ocr"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "vlm_unified_json",
                "task_eval": config,
            }
        ),
        encoding="utf-8",
    )
    contract = validation_engine.resolve_reference_precision_contract(
        argparse.Namespace(hf_dtype="auto"),
        model,
        work_dir,
    )

    assert model["precision"] == "fp16"
    assert contract == {
        "trtmc_base_precision": "fp16",
        "trtmc_quantization": "none",
        "reference_precision": "bf16",
        "reference_dtype": "bfloat16",
        "comparison": "reference_defined",
    }
    assert suite["gates"] == {
        "max_accuracy_drop_from_hf": 0.02,
        "min_prediction_agreement": 0.95,
    }
    assert (
        validation_engine.resolve_hf_reference_dtype(
            argparse.Namespace(hf_dtype="auto"),
            model,
            work_dir,
        )
        == "bfloat16"
    )


def test_c829_deepseek_ocr_receipt_passes_declared_parity_gates() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "ocrbench_v2_unified",
    )
    result = validation_engine.prediction_agreement_gate_result(
        {
            # QA run trtmc-validate-gb300-2-20260726-c8291be0-all105,
            # five-sample DeepSeek-OCR receipt.
            "hf": {"overall_accuracy": 0.6},
            "trtfb": {"overall_accuracy": 0.6},
            "prediction_agreement_rate": 1.0,
        },
        suite["gates"],
    )

    assert result["status"] == "passed"
    assert result["accuracy_drop_from_hf"] == 0.0
    assert result["gate_failures"] == []


def test_deepseek_ocr_bad_receipt_fails_both_parity_gates() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "ocrbench_v2_unified",
    )
    result = validation_engine.prediction_agreement_gate_result(
        {
            "hf": {"overall_accuracy": 0.6},
            "trtfb": {"overall_accuracy": 0.4},
            "prediction_agreement_rate": 0.8,
        },
        suite["gates"],
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "BenchmarkGateError"
    assert result["accuracy_drop_from_hf"] == pytest.approx(0.2)
    assert [failure["gate"] for failure in result["gate_failures"]] == [
        "max_accuracy_drop_from_hf",
        "min_prediction_agreement",
    ]


def test_unopted_unquantized_reference_mismatch_stays_fail_closed(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_kind": "vlm_unified_json",
                "task_eval": {"reference_precision": "bf16"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="reference precision bf16 does not match TRTMC base precision fp16",
    ):
        validation_engine.resolve_reference_precision_contract(
            argparse.Namespace(hf_dtype="auto"),
            {
                "name": "unopted-model",
                "precision": "fp16",
                "task_eval": {"reference_precision": "bf16"},
            },
            work_dir,
        )


def test_deepseek_ocr_fp16_reference_fails_before_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _write_vlm_inputs(tmp_path, dtype="float16")

    def fail_if_loaded():
        raise AssertionError("model dependencies must not load")

    monkeypatch.setattr(transformers_vlm, "_runtime_dependencies", fail_if_loaded)

    with pytest.raises(
        ValueError,
        match="DeepSeek-OCR official remote-code reference requires.*BF16",
    ):
        transformers_vlm.run(arguments)


def test_deepseek_ocr_bf16_runs_official_remote_infer_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _write_vlm_inputs(tmp_path, dtype="bfloat16")
    calls: dict[str, object] = {}

    class Processor:
        def __call__(self, text, *, add_special_tokens):
            assert text == "recognized text"
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[1, 2])

    class Model:
        def infer(self, processor, **kwargs):
            calls["processor"] = processor
            calls.update(kwargs)
            return "recognized text"

    processor = Processor()
    model = Model()
    monkeypatch.setattr(
        transformers_vlm,
        "_runtime_dependencies",
        lambda: (SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
    )

    def fake_load_runtime(arguments, *_args):
        assert arguments.dtype == "bfloat16"
        return processor, model, "cpu"

    monkeypatch.setattr(transformers_vlm, "_load_runtime", fake_load_runtime)

    transformers_vlm.run(arguments)

    assert calls["processor"] is processor
    assert calls["prompt"] == "<image>\nRead the image."
    assert calls["image_file"] == "/dataset/image.jpg"
    assert calls["save_results"] is False
    assert calls["eval_mode"] is True
    payload = json.loads(arguments.predictions.read_text(encoding="utf-8"))
    wall_ms = payload["responses"][0].pop("wall_ms")
    assert isinstance(wall_ms, float)
    assert payload == {
        "responses": [
            {
                "sample_id": "ocrbench_v2_000000",
                "output_text": "recognized text",
                "generated_tokens": 2,
                "generated_token_ids": [1, 2],
                "source": "hf",
            }
        ]
    }
