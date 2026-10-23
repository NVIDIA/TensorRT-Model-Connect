# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for nomic-embed-text.

**The gate is the pooled values.** Seven errors were built on purpose and
measured against this checkpoint:

=========================================  ==============  ============
rule broken                                pooled maxabs   pooled cosine
=========================================  ==============  ============
(correct build)                            3.7e-06         1.00000000
rotary base 10000 instead of 1000          0.194           0.99645567
token type embedding dropped               0.647           0.98069012
attention mask dropped (pads attend)       0.887           0.96837312
interleaved rotary instead of split-half   0.858           0.95412874
swiglu gate on fc11 instead of fc12        5.385           0.21539643
pre-norm block instead of post-norm        88.6            0.21521272
Wqkv split head-major instead of three     6.123           0.04051463
=========================================  ==============  ============

Unlike the vision families next door, a cosine floor would catch all seven
here - the rotary base, the weakest of them, still costs 0.0035 of cosine.
The values are gated anyway, because the margin is far larger in that space:
1e-3 sits 270x above a correct build's noise and 190x below the weakest
break, where a 0.999 cosine floor has only 1.7x of room against it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "nomic_bert"
TASKS = frozenset({"text_to_embedding"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        # The checkpoint's auto_map points at modules that live in a second
        # repository. Offline CI stages only what the manifest declares, so an
        # undeclared dependency fails as a missing import rather than as a
        # missing checkpoint. The declaration carries no revision on purpose:
        # auto_map names no revision either, so transformers resolves the
        # modules at "main", and staging a pinned commit leaves that alias
        # unresolvable offline.
        declared = {entry["repo_id"] for entry in manifest["hf_dependencies"]}
        assert declared == {"nomic-ai/nomic-bert-2048"}, (
            f"{path.name} must declare the repository its auto_map names"
        )
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
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(binary: Path, runtime_root: Path, bundle: Path, case: dict,
              command: str, *arguments: str) -> dict:
    invocation = [str(binary), command, str(bundle), "--runtime-root", str(runtime_root),
                  *arguments]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(invocation, check=True, capture_output=True, text=True, env=env,
                               timeout=int(case.get("runtime_timeout_s", 3600)))
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


# Measured on this case: a correct build is within 3.7e-06 of the reference
# and the weakest break the gate is meant to catch - the rotary base - is
# 0.194. This sits 270x above the noise and 190x below that break.
_EMBEDDING_TOLERANCE = 1e-3
# Kept only to fail early and loudly on a gross error.
_COSINE_FLOOR = 0.999

# The prefixes are the checkpoint's, not this test's: the runtime applies
# them from the role, and these are what it must apply.
_ROLE_PREFIXES = {"query": "search_query: ", "document": "search_document: "}


def _reference(model_dir: Path, text: str):
    """The reference's own pooled vector, pooled the way its module list says."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # trust_remote_code on both calls, not only the model: this checkpoint's
    # config carries an auto_map, so the tokenizer loader resolves it too and
    # would otherwise stop to ask on a terminal that is not there.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                      dtype=torch.float32).eval()
    batch = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state[0]
    # 1_Pooling/config.json sets pooling_mode_mean_tokens, and modules.json
    # lists no Normalize step, so the published vector is the plain mean.
    mask = batch["attention_mask"][0].unsqueeze(-1).to(hidden.dtype)
    return ((hidden * mask).sum(0) / mask.sum(0)).tolist()


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def _native(binary, runtime_root, bundle, case, text: str, role: str) -> dict:
    return _run_json(binary, runtime_root, bundle, case, "embed", "--task",
                     "text_to_embedding", "--text", text, "--role", role)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    vectors = {}
    worst = 0.0
    worst_cosine = 1.0
    for role, text in (("query", case["query"]), ("document", case["document"]),
                       ("document", case["negative_document"])):
        payload = _native(binary, runtime_root, bundle, case, str(text), role)

        # The runtime owns the prefix, so the reference is given the prefixed
        # text. If the runtime stopped applying it, this is what would notice.
        reference = _reference(model_dir, _ROLE_PREFIXES[role] + str(text))
        values = [float(value) for value in payload["values"]]
        assert len(values) == len(reference), (
            f"{FAMILY} returned {len(values)} values, reference {len(reference)}"
        )
        delta = max(abs(a - b) for a, b in zip(values, reference))
        cosine = _cosine(values, reference)
        assert cosine >= _COSINE_FLOOR, f"{FAMILY} embedding cosine {cosine} for {text!r}"
        assert delta <= _EMBEDDING_TOLERANCE, (
            f"{FAMILY} embedding for {text!r} differs from the reference by {delta}"
        )
        worst = max(worst, delta)
        worst_cosine = min(worst_cosine, cosine)

        # The published module list is Transformer then Pooling, with no
        # Normalize, so the vector keeps its length and says so.
        assert payload["pooling"] == "mean"
        assert payload["normalization"] == "none"
        # Two nomic-embed checkpoints share this family and do not share a
        # space, and neither names the one it is in.
        assert payload["embedding_space"] == ""
        norm = math.sqrt(sum(value * value for value in values))
        assert abs(norm - 1.0) > 1e-3, (
            f"{FAMILY} reports normalization=none but returned a unit vector: {norm}"
        )
        vectors.setdefault(role, []).append(values)

    # The role is not decoration. Measured, embedding the same text as a
    # query and as a document gives vectors 0.88 apart in cosine, so a
    # runtime that ignored the role would still answer, with the wrong
    # vector. An unprefixed embedding is refused for the same reason.
    query = vectors["query"][0]
    document = vectors["document"][0]
    as_document = _native(binary, runtime_root, bundle, case, str(case["query"]), "document")
    assert _cosine(query, [float(v) for v in as_document["values"]]) < 0.99, (
        f"{FAMILY} embedded a query and a document the same way"
    )
    with pytest.raises(subprocess.CalledProcessError):
        _native(binary, runtime_root, bundle, case, str(case["query"]), "default")

    # Retrieval is reported, not gated: it is a property of the checkpoint
    # rather than of this build, and one pair is not a benchmark.
    negative = vectors["document"][1]
    relevant = _cosine(query, document)
    irrelevant = _cosine(query, negative)
    assert relevant > irrelevant, (
        f"{FAMILY} scored the unrelated document higher: {relevant} <= {irrelevant}"
    )
    print(
        f"{FAMILY} parity: {len(document)} values, embedding maxabs {worst:.3e}, "
        f"cosine {worst_cosine:.8f}, query-vs-answer {relevant:.6f} "
        f"vs query-vs-other {irrelevant:.6f}"
    )
