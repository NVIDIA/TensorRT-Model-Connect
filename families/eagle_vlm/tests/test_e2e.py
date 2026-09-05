# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for eagle_vlm."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "eagle_vlm"
TASKS = frozenset({"embedding", "reranking"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


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


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    required_gpus = int(manifest["tensor_parallel_size"])
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
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
    manifest: dict,
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
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "TRTMC_NCCL_RENDEZVOUS",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
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


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _case_text(case: dict) -> str:
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    assert a.shape == b.shape and a.size > 0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    assert denominator > 0.0
    return float(np.dot(a, b) / denominator)


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    task = manifest["task"]
    if task == "embedding":
        return _run_json(
            binary, runtime_root, bundle, manifest, case, "embed", "--text", _case_text(case)
        )
    if task == "reranking":
        inputs = case.get("inputs") or {}
        documents = inputs.get("documents") or [inputs.get("document") or "Paris is in France."]
        scores = []
        for document in documents:
            payload = _run_json(
                binary,
                runtime_root,
                bundle,
                manifest,
                case,
                "rerank",
                "--query",
                str(inputs.get("query") or _case_text(case)),
                "--document",
                str(document),
            )
            scores.append(float(payload["score"]))
        return {"scores": scores}
    raise AssertionError(f"{FAMILY} has no direct native test for task={task}")


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    task = manifest["task"]
    import torch
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoProcessor,
        AutoTokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == "reranking":
        inputs = case.get("inputs") or {}
        documents = inputs.get("documents") or [inputs.get("document") or "Paris is in France."]
        query = str(inputs.get("query") or _case_text(case))
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            max_input_tiles=6,
            use_thumbnail=True,
            rerank_max_length=8192,
        )
        encoded = processor.process_queries_documents_crossencoder(
            [
                {"question": query, "doc_text": str(document), "doc_image": ""}
                for document in documents
            ]
        )
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=_torch_dtype(case["reference_precision"]),
            )
            .to(device)
            .eval()
        )
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.no_grad():
            logits = model(**encoded).logits.float().cpu().reshape(-1)
        if logits.numel() == len(documents):
            scores = logits
        else:
            scores = logits.reshape(len(documents), -1)[:, -1]
        return {"scores": scores.tolist()}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    encoded = tokenizer(_case_text(case), return_tensors="pt", truncation=True)
    model = (
        AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=_torch_dtype(case["reference_precision"])
        )
        .to(device)
        .eval()
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    values = (hidden * mask).sum(1)[0] / mask.sum(1)[0].clamp(min=1e-09)
    values = torch.nn.functional.normalize(values, p=2, dim=0)
    return {"values": values.float().cpu().numpy()}


def test_embedding_reference_requests_hidden_states(monkeypatch, tmp_path: Path) -> None:
    import torch
    import transformers

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class Model:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert kwargs.pop("output_hidden_states") is True
            device = kwargs["input_ids"].device
            return type(
                "Output",
                (),
                {
                    "hidden_states": (
                        torch.zeros(1, 2, 2, device=device),
                        torch.tensor([[[3.0, 4.0]] * 2], device=device),
                    )
                },
            )()

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: Model(),
    )
    result = _official_reference(
        tmp_path,
        {"task": "embedding"},
        {"prompt": "test", "reference_precision": "fp32"},
        tmp_path,
    )
    np.testing.assert_allclose(result["values"], [0.6, 0.8], rtol=0.0, atol=1e-6)


def test_reranking_reference_uses_checkpoint_processor(monkeypatch, tmp_path: Path) -> None:
    import torch
    import transformers

    class Processor:
        def process_queries_documents_crossencoder(self, examples):
            assert examples == [
                {"question": "query", "doc_text": "first", "doc_image": ""},
                {"question": "query", "doc_text": "second", "doc_image": ""},
            ]
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.int64),
                "metadata": "kept",
            }

    class Model:
        def to(self, device):
            del device
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert kwargs["input_ids"].shape == (2, 2)
            assert kwargs["metadata"] == "kept"
            return type("Output", (), {"logits": torch.tensor([[1.0], [2.0]])})()

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: Processor(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: Model(),
    )
    result = _official_reference(
        tmp_path,
        {"task": "reranking"},
        {
            "inputs": {"prompt": "query", "documents": ["first", "second"]},
            "reference_precision": "fp32",
        },
        tmp_path,
    )
    assert result == {"scores": [1.0, 2.0]}


def test_reranking_bundle_uses_checkpoint_pooling(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from families.eagle_vlm import model as family_model

    config = SimpleNamespace(model_type="llama_nemotron_vl_rerank", raw={"pooling": "avg"})
    model = SimpleNamespace(
        load_weights=lambda *args: {},
        build_engine=lambda *args, **kwargs: b"engine",
    )
    sections = {}
    monkeypatch.setattr(family_model.ModelConfig, "from_dir", lambda model_dir: config)
    monkeypatch.setattr(family_model, "_EagleModel", lambda: model)
    monkeypatch.setattr(family_model, "_tokenizer_runtime_contract", lambda model_dir: {})
    family_model.build(
        SimpleNamespace(
            model_dir=tmp_path,
            backend="trt",
            dynamic_kv_cache=False,
            image_height=None,
            image_width=None,
            video_num_frames=None,
            max_batch_size=1,
            context_parallel_size=1,
            task="reranking",
            quantization=None,
            fp32_layers=(),
            tensor_parallel_size=1,
            max_sequence_length=685,
            precision="fp16",
            verbose=False,
        ),
        SimpleNamespace(
            set_header=lambda **kwargs: None,
            add_bytes=lambda *args: None,
            add_json=lambda name, data: sections.update({name: data}),
        ),
    )
    assert sections["runtime.json"] == {"tensor_parallel_size": 1, "pooling": "avg"}


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    task = manifest["task"]
    if task in {"encoding", "embedding"}:
        actual_values = np.asarray(actual["values"]).reshape(-1)
        expected_values = np.asarray(expected["values"]).reshape(-1)
        configured = thresholds.get(
            "contract_cosine_threshold", thresholds.get("cls_embedding_cosine", 0.98)
        )
        assert _cosine(actual_values, expected_values) >= max(float(configured), 0.8)
        return
    if task == "reranking":
        left = np.asarray(actual["scores"], dtype=np.float64)
        right = np.asarray(expected["scores"], dtype=np.float64)
        assert left.shape == right.shape and left.size >= 2
        pairwise = []
        for first in range(left.size):
            for second in range(first + 1, left.size):
                pairwise.append((left[first] > left[second]) == (right[first] > right[second]))
        agreement = float(np.mean(pairwise))
        assert agreement >= float(thresholds.get("contract_ranking_agreement", 0.9))
        return
    raise AssertionError(f"{FAMILY} has no parity gate for task={task}")


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
