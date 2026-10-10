# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer checkpoint-to-family-DSO functional and semantic proof."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence


_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "text_generation", path
        assert manifest["precision"] == "fp16", path
        assert manifest["quantization"] == "nvfp4", path
        assert manifest["tensor_parallel_size"] == 1, path
        for case in manifest["testcases"]:
            assert case["name"] not in cases, case["name"]
            cases[case["name"]] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    selected |= _csv_values(config.getoption("--e2e-testcase", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected |= {
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    if os.environ.get("TRTMC_E2E") != "1" and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _checkpoint(manifest: dict) -> Path:
    override = os.environ.get("TRTMC_MUSE_GLIMMER_MODEL_DIR")
    if override:
        path = Path(override)
    else:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"]))
    assert (path / "config.json").is_file(), path
    assert (path / "hf_quant_config.json").is_file(), path
    return path


def _required_environment() -> tuple[Path, Path]:
    binary = Path(os.environ["TRTMC_BINARY"])
    runtime_root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), runtime_root
    assert (runtime_root / "libtrtmc_model_muse_glimmer.so").is_file(), runtime_root
    return binary, runtime_root


def _build_bundle(manifest: dict, model_dir: Path, bundle: Path) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
            quantization=manifest["quantization"],
        )
    )
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle


def _run_native(binary: Path, runtime_root: Path, bundle: Path, case: dict) -> dict:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        case["prompt"],
        "--system-prompt",
        case["system_prompt"],
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--temperature",
        str(case["temperature"]),
        "--top-k",
        str(case["top_k"]),
        "--top-p",
        str(case["top_p"]),
        "--use-chat-template",
        str(case["use_chat_template"]).lower(),
        "--enable-thinking",
        str(case["enable_thinking"]).lower(),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    record_evidence("commands", {"argv": command})
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    return json.loads(completed.stdout)


def _assert_correctness(payload: dict, case: dict) -> None:
    token_ids = payload["token_ids"]
    assert isinstance(token_ids, list) and token_ids
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert len(token_ids) <= case["max_new_tokens"]
    _assert_answer(str(payload["text"]), case)


def _assert_answer(output: str, case: dict) -> None:
    marker = "assistant to=user"
    assert marker in output, "Muse ATEM output never reached its public user channel"
    answer = output.rsplit(marker, 1)[1].casefold()
    assert answer.strip()
    for alternatives in case["expected_terms"]:
        assert any(term.casefold() in answer for term in alternatives), (alternatives, answer)


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    binary, runtime_root = _required_environment()
    model_dir = _checkpoint(manifest)
    bundle = tmp_path / manifest["bundle"]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest["hf_id"],
            "hf_revision": manifest["hf_revision"],
        },
    )
    with evidence_stage("build"):
        _build_bundle(manifest, model_dir, bundle)
    inspected = subprocess.run(
        [str(binary), "inspect", str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    inspection = json.loads(inspected.stdout)
    assert inspection["family"] == _FAMILY
    assert inspection["task"] == manifest["task"]
    with evidence_stage("native"):
        payload = _run_native(binary, runtime_root, bundle, case)
    with evidence_stage("compare"):
        _assert_correctness(payload, case)
