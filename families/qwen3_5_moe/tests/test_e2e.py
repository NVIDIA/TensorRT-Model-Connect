# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provisioned family-owned build/public/direct/independent-quality E2E; never a CPU placeholder."""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from tensorrt_model_connect import BuildRequest, build
from families.qwen3_5_moe.tests import edge_validation

TEST_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((TEST_DIR / "manifests/qwen35-moe-gptq-edge.json").read_text())


def _selection(config) -> set[str]:
    """Honor the same explicit family/case selection switches as other family tests."""
    selected = {
        item.strip()
        for flag in ("--e2e-model", "--e2e-testcase")
        for value in (config.getoption(flag, default=[]) or [])
        for item in value.split(",")
        if item.strip()
    }
    filename = config.getoption("--e2e-models-file", default=None)
    if filename:
        selected.update(
            line.strip()
            for line in Path(filename).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return selected


@pytest.mark.parametrize("case_name", [case["name"] for case in MANIFEST["testcases"]])
def test_official_checkpoint_e2e(case_name: str, tmp_path: Path, request) -> None:
    """Selected execution must complete a fresh build and unchanged strict HF comparisons."""
    selected = _selection(request.config)
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real family E2E requires explicit selection or TRTMC_E2E=1")
    if selected and not ({case_name, MANIFEST["name"], MANIFEST["family"]} & selected):
        pytest.skip("case not selected")
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    reference_python = Path(os.environ.get("TRTMC_QWEN35_MOE_REFERENCE_PYTHON", sys.executable))
    assert runtime.is_dir() and reference_python.is_file()
    assert (runtime / "libtrtmc_model_qwen3_5_moe.so").is_file()
    assert (runtime / "libtrtmc_backend_trt.so").is_file()
    native_build = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR") or runtime)
    helper = native_build / "families/qwen3_5_moe/qwen3_5_moe_edge_inference"
    assert helper.is_file() and os.access(helper, os.X_OK), helper
    from huggingface_hub import constants, snapshot_download

    source = Path(snapshot_download(
        repo_id=MANIFEST["hf_id"],
        revision=MANIFEST["hf_revision"],
        local_files_only=constants.HF_HUB_OFFLINE,
    ))
    # The family stages regular hardlinks itself and copies only writable metadata.
    # Reuse the immutable snapshot here; the independent reference reads it directly.
    checkpoint = source
    bundle = tmp_path / MANIFEST["bundle"]
    build(
        BuildRequest(
            checkpoint,
            bundle,
            MANIFEST["family"],
            MANIFEST["task"],
            MANIFEST["precision"],
            max_sequence_length=MANIFEST["max_sequence_length"],
            tensor_parallel_size=MANIFEST["tensor_parallel_size"],
        )
    )
    output = tmp_path / "proof"
    case = next(item for item in MANIFEST["testcases"] if item["name"] == case_name)
    image = TEST_DIR / case["test_image"]
    edge_validation.gpu(
        SimpleNamespace(
            bundle=bundle,
            runtime_root=runtime,
            native_build_dir=native_build,
            output=output,
            image=image,
            checkpoint_id=MANIFEST["hf_id"],
            checkpoint_revision=MANIFEST["hf_revision"],
        )
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    edge_validation.run_driver(
        [
            str(reference_python),
            "-m",
            "families.qwen3_5_moe.tests.hf_reference",
            "--device",
            "cuda",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--image",
            str(image),
        ],
        output / "hf.log",
        env=environment,
        timeout=3600,
    )
    evidence = json.loads((output / "evidence.json").read_text())
    assert evidence["passed"] and evidence["hf_quality"] == "passed"


def test_catalog_case_is_the_validated_image_request() -> None:
    """trtmc-bench, and AIPerf's bundle build, run the catalog case: it is the image request this family validates."""
    case = next(item for item in MANIFEST["testcases"] if "test_image" in item)
    image = TEST_DIR / case["test_image"]
    public, _ = edge_validation.requests(
        MANIFEST["max_sequence_length"], image, 16, 16, MANIFEST["task"], backend="native"
    )
    validated = next(row for row in public if row["id"] == "image")
    assert image.is_file()
    assert (case["prompt"], case["max_new_tokens"]) == (
        validated["prompt"],
        validated["config"]["max_new_tokens"],
    )
