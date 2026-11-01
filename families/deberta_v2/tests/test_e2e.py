# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for DeBERTa-v2/v3.

**The gate is the feature values.** Seven errors were built on purpose and
measured against this checkpoint, on the passage this case ships:

=========================================  ==========  ============
rule broken                                maxabs      cosine
=========================================  ==========  ============
(correct build)                            4.29e-05    1.00000012
linear relative positions, not log buckets 0.0685      0.99999666
absolute position embeddings added         4.801       0.99581569
scale sqrt(d) instead of sqrt(3d)          7.83        0.94018942
c2p term dropped                           13.37       0.85509247
p2c term dropped                           19.62       0.27363199
c2p and p2c indices swapped                19.91       0.44196913
relative embeddings not layer-normalised   20.32       0.41130102
=========================================  ==========  ============

**The passage is long on purpose.** Below ``position_buckets / 2`` - 128 rows
here - the log bucketing *is* the identity, so on a short input the bucketing
row above measures exactly the baseline and the gate tests nothing. At 148
tokens it costs 0.0685. A 0.999 cosine floor would still miss it at
0.99999666, which is why the values are gated and the cosine is only a
loud-failure guard.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "deberta_v2"
TASKS = frozenset({"text_to_token_features"})
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


# Measured on this case: a correct build is within 4.29e-05 of the reference
# and the weakest break the gate is meant to catch - the relative-position
# bucketing - is 0.0685. This sits 116x above the noise and 13x below that
# break.
_FEATURE_TOLERANCE = 5e-3
# Kept only to fail early and loudly on a gross error; measured, it cannot
# see the bucketing break at all.
_COSINE_FLOOR = 0.999


def _reference_ids(model_dir: Path, text: str) -> list[int]:
    """The framing the checkpoint was trained with, built explicitly.

    The installed transformers builds this tokenizer with a post-processor
    that adds no special tokens, so ``tokenizer(text)`` returns the body
    alone. The framing is applied here, and the family is required to produce
    the same ids, rather than letting a tokenizer-version quirk decide what
    the gate compares.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    body = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.cls_token_id, *body, tokenizer.sep_token_id]


def _reference_features(model_dir: Path, ids: list[int], window: int):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    padded = list(ids) + [0] * (window - len(ids))
    mask = [1] * len(ids) + [0] * (window - len(ids))
    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([padded]),
                       attention_mask=torch.tensor([mask])).last_hidden_state[0]
    return hidden[: len(ids)].tolist()


def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    prompt = str(case["prompt"])
    payload = _run_json(binary, runtime_root, bundle, case, "encode", "--task",
                        "text_to_token_features", "--text", prompt)

    # The tokenizer is gated before the features are. The runtime converts
    # spm.model at build time and frames the result itself, so an error in
    # either would otherwise show up only as a feature mismatch with no
    # indication of where it came from.
    expected_ids = _reference_ids(model_dir, prompt)
    tokens = payload["tokens"]
    assert [int(token["token_id"]) for token in tokens] == expected_ids, (
        f"{FAMILY} token ids differ from the reference tokenizer"
    )
    assert [int(token["token_index"]) for token in tokens] == list(range(len(expected_ids)))
    assert all(int(token["input_index"]) == 0 for token in tokens)

    # Long enough that the bucketing is not the identity; see the module
    # docstring for what a short passage would fail to test.
    assert len(expected_ids) > int(manifest["max_sequence_length"]) // 4, (
        "this case is meant to exercise the relative-position bucketing"
    )

    # The matrix is tokens by features, and says so: a transposed result
    # would otherwise have the right element count and the wrong meaning.
    assert payload["feature_kind"] == "token"
    assert payload["axes"] == ["token", "feature"]
    shape = [int(value) for value in payload["shape"]]
    assert shape[0] == len(expected_ids), (
        f"{FAMILY} returned {shape[0]} rows for {len(expected_ids)} tokens"
    )
    assert shape[1] == int(payload["dim"])

    values = [float(value) for value in payload["values"]]
    assert len(values) == shape[0] * shape[1]
    columns = shape[1]
    native = [values[row * columns:(row + 1) * columns] for row in range(shape[0])]
    reference = _reference_features(model_dir, expected_ids,
                                    int(manifest["max_sequence_length"]))

    worst = max(max(abs(a - b) for a, b in zip(row, want))
                for row, want in zip(native, reference))
    cosine = _cosine([v for row in native for v in row],
                     [v for row in reference for v in row])
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} feature cosine {cosine}"
    assert worst <= _FEATURE_TOLERANCE, (
        f"{FAMILY} features differ from the reference by {worst}"
    )
    print(
        f"{FAMILY} parity: {shape[0]} tokens x {shape[1]} features, "
        f"feature maxabs {worst:.3e}, cosine {cosine:.8f}"
    )
