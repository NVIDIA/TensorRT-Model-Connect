# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned official-reference parity through the native text-embedding Task API."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "jina_embeddings_v3"
TASKS = frozenset({"text_to_embedding"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            (
                line.strip()
                for line in Path(models_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and (not line.lstrip().startswith("#"))
            )
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update((item.strip() for item in str(raw).split(",") if item.strip()))
    if not model_filters and (not testcase_filters):
        return (sorted(CASES), False)
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or (manifest["name"] in model_filters)
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return (sorted(selected), True)


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(
                    name,
                    marks=pytest.mark.skip(
                        reason="direct E2E requires one of the three explicit E2E selectors"
                    ),
                )
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
    return (binary, runtime_root)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def test_official_checkpoint_e2e(case_name, tmp_path):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    binary, runtime_root = _runtime()
    tokenizer = AutoTokenizer.from_pretrained(directory)
    for text in [
        "A small test.",
        "日本語の文章 中文 한국어",
        "café e\u0301 １２３\n test",
        "",
        "  trailing spaces  ",
    ]:
        completed = subprocess.run(
            [
                str(runtime_root / "test_jina_embeddings_v3_tokenizer"),
                str(directory / "tokenizer.json"),
                text,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == tokenizer.encode(text), repr(text)

    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = (
        AutoModel.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
        .eval()
        .to("cuda")
    )
    for name in ("retrieval_query", "retrieval_passage"):
        model.load_adapter(str(directory / name), adapter_name=name)
    model.eval()
    roles = {}
    for item in case["inputs"]:
        if item["role"] == "default":
            model.disable_adapters()
        else:
            model.enable_adapters()
            model.set_adapter(
                {"query": "retrieval_query", "document": "retrieval_passage"}[item["role"]]
            )
        inputs = tokenizer(item["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            valid = inputs["attention_mask"].unsqueeze(-1)
            mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1)
            expected = torch.nn.functional.normalize(mean, dim=-1)[0].cpu().numpy()
        payload = _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "embed",
            "--task",
            manifest["task"],
            "--text",
            item["text"],
            "--role",
            item["role"],
        )
        assert payload["embedding_space"] == "jina_embeddings_v3"
        assert payload["pooling"] == "mean" and payload["normalization"] == "l2"
        actual = np.asarray(payload["values"], dtype=np.float32)
        assert actual.shape == expected.shape == (1024,)
        assert np.isfinite(actual).all()
        assert abs(np.linalg.norm(actual) - 1) <= 1e-5
        cosine = np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
        assert cosine >= 0.99999
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
        roles.setdefault(item["role"], actual)
        print(
            f"Jina role={item['role']}, cosine={cosine:.8f}, maxabs={np.max(np.abs(actual - expected)):.6g}"
        )
    # The adapters must affect the embedding even with identical input text.
    assert np.linalg.norm(roles["default"] - roles["query"]) > 0.01
    assert np.linalg.norm(roles["query"] - roles["document"]) > 0.01
