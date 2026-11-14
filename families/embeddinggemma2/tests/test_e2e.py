# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit native Task qualification against the pinned official text reference."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "embeddinggemma2"
ROOT = Path(__file__).resolve().parent
CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    models, tests = set(), set()
    for value in metafunc.config.getoption("--e2e-model") or []:
        models.update(item.strip() for item in value.split(","))
    file = metafunc.config.getoption("--e2e-models-file")
    if file:
        models.update(
            line.strip()
            for line in Path(file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    for value in metafunc.config.getoption("--e2e-testcase") or []:
        tests.update(item.strip() for item in value.split(","))
    selected = []
    for name, (manifest, _) in CASES.items():
        if models and not models.intersection({FAMILY, name, manifest["name"]}):
            continue
        if tests and name not in tests:
            continue
        selected.append(
            name
            if models or tests
            else pytest.param(name, marks=pytest.mark.skip(reason="explicit E2E selector required"))
        )
    metafunc.parametrize("case_name", selected)


def required(name):
    value = os.environ.get(name)
    assert value, f"selected EmbeddingGemma2 E2E requires {name}"
    path = Path(value)
    assert path.exists(), f"{name} does not exist: {path}"
    return path


def test_embedding_e2e(case_name, tmp_path):
    manifest, case = CASES[case_name]
    import torch

    assert torch.cuda.is_available(), "selected EmbeddingGemma2 E2E requires CUDA"
    reference_python = Path(
        os.environ.get("TRTMC_EMBEDDINGGEMMA2_REFERENCE_PYTHON", sys.executable)
    )
    assert reference_python.is_file(), f"reference interpreter does not exist: {reference_python}"
    runtime = required("TRTMC_RUNTIME_ROOT")
    native_build = required("TRTMC_NATIVE_BUILD_DIR")
    explicit = os.environ.get("TRTMC_EMBEDDINGGEMMA2_MODEL_DIR")
    if explicit:
        model_dir = required("TRTMC_EMBEDDINGGEMMA2_MODEL_DIR")
    else:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download(
                manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
            )
        )
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task="embedding",
                precision=manifest["precision"],
                tensor_parallel_size=int(manifest["tensor_parallel_size"]),
                max_sequence_length=manifest["max_sequence_length"],
            )
        )
    subprocess.run(
        ["cmake", "--build", str(native_build), "--target", "embeddinggemma2_consumer"],
        check=True,
        timeout=600,
    )
    consumer = native_build / "families" / FAMILY / "embeddinggemma2_consumer"
    env = {
        **os.environ,
        "LD_LIBRARY_PATH": str(runtime) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    with evidence_stage("native"):
        actual_run = subprocess.run(
            [str(consumer), str(bundle), str(runtime), case["prompt"], case["role"]],
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=300,
        )
        actual = np.asarray(json.loads(actual_run.stdout), dtype=np.float32)
    with evidence_stage("reference"):
        reference_run = subprocess.run(
            [
                str(reference_python),
                str(ROOT / "reference.py"),
                str(model_dir),
                case["reference_precision"],
                case["prompt"],
                case["role"],
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=600,
        )
        reference = json.loads(reference_run.stdout)
        expected = np.asarray(reference["embedding"], dtype=np.float32)
    record_evidence("native", {"embedding": actual.tolist(), "interface": "text_to_embedding"})
    record_evidence("reference", {**reference, "hf_revision": manifest["hf_revision"]})
    record_evidence(
        "thresholds", {"cosine_similarity": 0.99, "l2_distance": 0.1, "norm_tolerance": 0.001}
    )
    with evidence_stage("compare"):
        assert actual.shape == expected.shape == (768,)
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        assert abs(float(np.linalg.norm(actual)) - 1) <= 0.001
        assert (
            float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
            >= 0.99
        )
        assert float(np.linalg.norm(actual - expected)) <= 0.1
