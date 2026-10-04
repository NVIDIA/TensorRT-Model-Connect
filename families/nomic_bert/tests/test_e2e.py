# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from families.nomic_bert.cli import BuildRequest, build_bundle
from tools.e2e_evidence import evidence_stage, record_evidence


MANIFEST = json.loads((Path(__file__).parent / "manifests/nomic-embed-text-v1.5.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    models = set()
    for item in metafunc.config.getoption("--e2e-model") or []:
        models.update(part.strip() for part in item.split(",") if part.strip())
    models_file = metafunc.config.getoption("--e2e-models-file")
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    names = set()
    for item in metafunc.config.getoption("--e2e-testcase") or []:
        names.update(part.strip() for part in item.split(",") if part.strip())
    parameters = []
    for name in CASES:
        enabled = (
            (models or names)
            and (not models or models.intersection({"nomic_bert", MANIFEST["name"], name}))
            and (not names or name in names)
        )
        parameters.append(
            name
            if enabled
            else pytest.param(
                name,
                marks=pytest.mark.skip(reason="Nomic E2E requires an explicit matching selector"),
            )
        )
    metafunc.parametrize("case_name", parameters)


@pytest.fixture(scope="session")
def runtime(tmp_path_factory):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    assert MANIFEST["tensor_parallel_size"] == 1
    model_dir = Path(
        snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        )
    )
    build_dir = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    binary = Path(os.environ["TRTMC_BINARY"])
    root = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    consumer = build_dir / "families/nomic_bert/nomic_embedding_consumer"
    assert consumer.is_file() and binary.is_file()
    assert (root / "libtrtmc_model_nomic_bert.so").is_file()
    bundle = tmp_path_factory.mktemp("nomic") / MANIFEST["bundle"]
    with evidence_stage("build"):
        build_bundle(
            BuildRequest(
                model_dir,
                task=MANIFEST["task"],
                precision=MANIFEST["precision"],
                max_sequence_length=MANIFEST["max_sequence_length"],
            ),
            bundle,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        code_revision=MANIFEST["code_revision"],
    )
    model = (
        AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            code_revision=MANIFEST["code_revision"],
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        .eval()
        .to("cuda")
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    return bundle, root, consumer, binary, tokenizer, model


def test_official_checkpoint_e2e(case_name, runtime):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    case = CASES[case_name]
    assert case["reference_precision"] == "fp32"
    bundle, root, consumer, binary, tokenizer, model = runtime
    record_evidence("inputs", {"manifest": MANIFEST, "case": case})
    with evidence_stage("native"):
        argv = [str(consumer), str(bundle), str(root), case["prompt"]]
        completed = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=300)
        actual = np.asarray(json.loads(completed.stdout), dtype=np.float32)
        record_evidence("commands", {"argv": argv})
        record_evidence("native", {"values": actual.tolist()})
    expected = []
    with evidence_stage("reference"):
        for prefix in ("", "search_query: ", "search_document: "):
            inputs = tokenizer(prefix + case["prompt"], return_tensors="pt", truncation=False)
            assert inputs["input_ids"].shape[1] <= MANIFEST["max_sequence_length"]
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                hidden = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                expected.append(torch.nn.functional.normalize(pooled, p=2, dim=-1)[0].cpu().numpy())
    expected = np.asarray(expected)
    record_evidence(
        "reference", {"values": expected.tolist(), "hf_revision": MANIFEST["hf_revision"]}
    )
    with evidence_stage("compare"):
        assert actual.shape == expected.shape == (3, 768)
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        norms = np.linalg.norm(actual, axis=1)
        cosine = (actual * expected).sum(1) / (norms * np.linalg.norm(expected, axis=1))
        distance = np.linalg.norm(actual - expected, axis=1)
        record_evidence("metrics", {"cosine": cosine.tolist(), "l2_distance": distance.tolist()})
        assert np.all(np.abs(norms - 1) <= 1e-5)
        assert np.all(cosine >= 0.9999)
        assert np.all(distance <= 0.01)
    with evidence_stage("native_cli"):
        argv = [
            str(binary),
            "embed",
            str(bundle),
            "--runtime-root",
            str(root),
            "--task",
            "text_to_embedding",
            "--role",
            "query",
            "--text",
            case["prompt"],
        ]
        completed = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=300)
        payload = json.loads(completed.stdout)
        np.testing.assert_allclose(payload["values"], actual[1], rtol=0, atol=1e-7)
    with evidence_stage("bounds"):
        argv[-1] = "hello " * MANIFEST["max_sequence_length"]
        rejected = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        assert rejected.returncode != 0
        assert "token limit" in rejected.stderr
