# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for ESM-2.

**The gate is the logit values.** The breaks this is meant to catch are the
ones that load every tensor and still run, so each was built on purpose and
measured against this checkpoint on the sequence this case ships. The numbers
live in the PR that added the family; what matters here is that the tolerance
sits well above a correct build and well below the weakest break.

Five breaks were built on purpose and measured against this checkpoint on
the sequence this case ships:

=========================================  ===========  ==========
rule broken                                maxabs       cosine
=========================================  ===========  ==========
(correct build)                            6.438e-04    1.00000000
rotary schedule recomputed, not loaded     6.224e-03    1.00000000
query not scaled before the rotation       -            0.95814394
post-norm blocks instead of pre-norm       -            0.86183436
masked rows not zeroed (token_dropout off) 2.325e-01    -
=========================================  ===========  ==========

**Two rules here are deliberately not gated, because no input can gate
them**, and saying so is more useful than a control that proves nothing:

* ``token_dropout``'s compensation factor. It scales every row by
  ``0.88 / (1 - observed mask ratio)``, and the next thing the encoder does
  is a LayerNorm, which is scale-invariant - so the factor is erased.
  Measured both ways: dropping the input-dependent divisor costs 3.3e-05 at
  one mask and 9.7e-05 at eight, although the factor itself moves from 0.8%
  to 6.5% wrong. The gap does not grow with the error, which is the
  signature of scale-invariance rather than of a weak case. What
  ``token_dropout`` does observably is zero the masked rows, and that is the
  2.3e-01 row above.
* Zeroing the padded embedding rows. Byte-for-byte identical with and
  without: the additive attention mask already removes padded keys from
  every real row's softmax, and the result is read only at real positions.

Both are implemented because the reference does them, and both are fidelity
rather than behaviour. This file does not pretend to gate them.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "esm"
TASKS = frozenset({"masked_text_to_token_scores"})
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


# Logits, not features: the lm_head puts these on a wider scale than a hidden
# row, so the tolerance is stated against the reference logits directly.
_LOGIT_TOLERANCE = 5e-3
# Kept only to fail early and loudly on a gross error.
_COSINE_FLOOR = 0.999


def _reference_ids(model_dir: Path, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return list(tokenizer(text)["input_ids"])


def _reference_logits(model_dir: Path, ids: list[int], window: int):
    import torch
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    padded = list(ids) + [1] * (window - len(ids))
    mask = [1] * len(ids) + [0] * (window - len(ids))
    with torch.no_grad():
        scores = model(input_ids=torch.tensor([padded]),
                       attention_mask=torch.tensor([mask])).logits[0]
    return scores[: len(ids)].tolist()


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
                        "masked_text_to_token_scores", "--text", prompt)

    # The tokenizer is gated before the logits are. ESM's tokenizer is not a
    # fast one, so the family reads vocab.txt itself and matches longest-token
    # first; an error there would otherwise surface only as a value mismatch.
    expected_ids = _reference_ids(model_dir, prompt)
    positions = payload["positions"]
    assert payload["score_kind"] == "logit"
    assert payload["axes"] == ["selected_position", "vocabulary_id"]
    assert payload["vocabulary_id"] == "esm"

    # One <mask> in, one row out: the Task returns the masked positions when
    # the caller marked any. The mask is what makes token_dropout zero a row,
    # which is the part of it the tolerance below can see.
    masked = [index for index, value in enumerate(expected_ids) if value == 32]
    assert len(masked) == 1, "this case is meant to exercise the masked-position path"
    assert [int(token["token_index"]) for token in positions] == masked
    assert [int(token["token_id"]) for token in positions] == [expected_ids[index]
                                                               for index in masked]
    assert all(int(token["input_index"]) == 0 for token in positions)

    shape = [int(value) for value in payload["shape"]]
    assert shape[0] == len(masked)
    values = [float(value) for value in payload["logits"]]
    assert len(values) == shape[0] * shape[1]
    columns = shape[1]
    native = [values[row * columns:(row + 1) * columns] for row in range(shape[0])]

    reference_rows = _reference_logits(model_dir, expected_ids,
                                       int(manifest["max_sequence_length"]))
    reference = [reference_rows[index] for index in masked]
    assert columns == len(reference[0])

    worst = max(max(abs(a - b) for a, b in zip(row, want))
                for row, want in zip(native, reference))
    cosine = _cosine([v for row in native for v in row],
                     [v for row in reference for v in row])
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} logit cosine {cosine}"
    assert worst <= _LOGIT_TOLERANCE, (
        f"{FAMILY} logits differ from the reference by {worst}"
    )
    print(
        f"{FAMILY} parity: {shape[0]} masked positions x {shape[1]} vocabulary, "
        f"logit maxabs {worst:.3e}, cosine {cosine:.8f}"
    )
