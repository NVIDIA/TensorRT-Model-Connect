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

The fp16 case uses its own logit tolerance, set from the fp16 noise
measured on this case; see ``_FP16_LOGIT_TOLERANCE``.

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

from tools.e2e_evidence import evidence_stage, record_evidence, record_reference_comparison

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
    for fixture in ("case_name", "fp16_case_name"):
        if fixture not in metafunc.fixturenames:
            continue
        names, enabled = _selected_cases(metafunc.config)
        if fixture == "fp16_case_name":
            names = [name for name in names if CASES[name][1]["precision"] == "fp16"]
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
        metafunc.parametrize(fixture, parameters, ids=names)


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

# fp16 cannot use the tolerance above. Measured on this case (B200 and B300,
# TensorRT 11.1 and 11.4, 22 builds), a correct fp16 engine is 7.6e-03 to
# 2.5e-02 off the reference at the masked row, and an fp16 engine that leaves
# the masked rows (token_dropout off) 2.2e-01 to 2.4e-01. 7.5e-02 sits 3x above
# the first and 3x below the second. The two cosine breaks of the table stay
# below the shared floor (fp16 noise moves the cosine by under 1e-06).
#
# The rotary-schedule break (5.6e-03) is inside the fp16 noise and is not
# gated here: an fp16 build with it is 1.0e-02 to 1.7e-02 off, inside the range
# as a correct one, and the share of that defect a correct fp16 engine carries
# ranges from -1.6 to 0.9 on this row, so no second reference separates them
# the way the tanh-GELU one does for vit. The schedule is chosen by the same
# code at every precision, so the fp32 case gates it for both.
#
# test_fp16_gate_rejects_unzeroed_masked_rows builds both and checks this.
_FP16_LOGIT_TOLERANCE = 7.5e-2


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


def _check_logits(native, reference, precision: str) -> tuple[float, float]:
    """Assert the gate of ``precision`` on the selected rows; return max abs and cosine."""
    assert len(native) == len(reference)
    worst = max(max(abs(a - b) for a, b in zip(row, want))
                for row, want in zip(native, reference))
    cosine = _cosine([v for row in native for v in row],
                     [v for row in reference for v in row])
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} logit cosine {cosine}"
    tolerance = _FP16_LOGIT_TOLERANCE if precision == "fp16" else _LOGIT_TOLERANCE
    assert worst <= tolerance, (
        f"{FAMILY} {precision} logits differ from the reference by {worst}"
    )
    record_evidence("thresholds", {"cosine_floor": _COSINE_FLOOR, "max_abs_error": tolerance})
    record_reference_comparison(
        "Reference logits",
        {"logits": native},
        {"logits": reference},
        [
            {
                "name": "cosine",
                "scope": "independent_reference",
                "actual": cosine,
                "operator": ">=",
                "expected": _COSINE_FLOOR,
                "passed": cosine >= _COSINE_FLOOR,
            },
            {
                "name": "max_abs_error",
                "scope": "independent_reference",
                "actual": worst,
                "operator": "<=",
                "expected": tolerance,
                "passed": worst <= tolerance,
            },
        ],
    )
    return worst, cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "thresholds",
        {
            "cosine_floor": _COSINE_FLOOR,
            "max_abs_error": _FP16_LOGIT_TOLERANCE
            if manifest["precision"] == "fp16"
            else _LOGIT_TOLERANCE,
        },
    )
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)

    prompt = str(case["prompt"])
    record_evidence("inputs", {"prompt": prompt})
    with evidence_stage("native"):
        payload = _run_json(binary, runtime_root, bundle, case, "encode", "--task",
                            "masked_text_to_token_scores", "--text", prompt)
    record_evidence("native", payload)

    # The tokenizer is gated before the logits are. ESM's tokenizer is not a
    # fast one, so the family reads vocab.txt itself and matches longest-token
    # first; an error there would otherwise surface only as a value mismatch.
    with evidence_stage("reference"):
        expected_ids = _reference_ids(model_dir, prompt)
    record_evidence("reference", {"token_ids": expected_ids})
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

    with evidence_stage("reference"):
        reference_rows = _reference_logits(model_dir, expected_ids,
                                           int(manifest["max_sequence_length"]))
    reference = [reference_rows[index] for index in masked]
    record_evidence("reference", {"logits": reference, "positions": masked})
    assert columns == len(reference[0])

    with evidence_stage("compare"):
        worst, cosine = _check_logits(native, reference, manifest["precision"])
    record_evidence("metrics", {"max_abs_error": worst, "cosine": cosine})
    print(
        f"{FAMILY} {manifest['precision']} parity: {shape[0]} masked positions x "
        f"{shape[1]} vocabulary, "
        f"logit maxabs {worst:.3e}, cosine {cosine:.8f}"
    )


def _synthetic(seed: int, noise: float, rows: int = 1) -> tuple[list, list]:
    """Reference logits on the scale of this case, and scores with random noise of size ``noise``."""
    import random

    rng = random.Random(seed)
    reference = [[rng.gauss(0.0, 4.0) for _ in range(33)] for _ in range(rows)]
    scores = [[value + rng.uniform(-noise, noise) for value in row] for row in reference]
    return scores, reference


def test_fp16_gate_takes_fp16_noise_and_fails_the_masked_row_break() -> None:
    """fp16-sized noise passes the fp16 gate; an error of the unzeroed-mask size fails it."""
    scores, reference = _synthetic(0, 2.5e-2)
    _check_logits(scores, reference, "fp16")
    scores[0][3] = reference[0][3] + 0.2
    with pytest.raises(AssertionError, match="fp16 logits differ"):
        _check_logits(scores, reference, "fp16")


def test_fp32_gate_is_unchanged() -> None:
    """fp32 keeps its own tolerance: fp16-sized noise fails it."""
    scores, reference = _synthetic(0, 1e-5)
    _check_logits(scores, reference, "fp32")
    scores, reference = _synthetic(0, 2.5e-2)
    with pytest.raises(AssertionError, match="fp32 logits differ"):
        _check_logits(scores, reference, "fp32")


def test_fp16_gate_rejects_unzeroed_masked_rows(fp16_case_name: str) -> None:
    """A correct fp16 engine passes the fp16 gate; one that leaves the masked rows fails it.

    Both engines are built from the case checkpoint and run in this process on
    the case prompt, padded to the window the way the runtime pads it.
    """
    import tensorrt as trt
    import torch

    from .. import config as config_module, model

    _, manifest, case = CASES[fp16_case_name]
    assert manifest["precision"] == "fp16"
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    window = int(manifest["max_sequence_length"])
    ids = _reference_ids(model_dir, str(case["prompt"]))
    masked = [index for index, value in enumerate(ids) if value == 32]
    reference_rows = _reference_logits(model_dir, ids, window)
    reference = [reference_rows[index] for index in masked]
    record_evidence("reference", {"logits": reference, "positions": masked})
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights, model._prefix(weights))

    input_ids = torch.ones(window, dtype=torch.int32, device="cuda")
    input_ids[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
    attention_mask = torch.zeros(window, dtype=torch.int32, device="cuda")
    attention_mask[: len(ids)] = 1

    def run(build_cfg) -> list[list[float]]:
        plan = model.build_engine(weights, build_cfg, seq_len=window, precision="fp16")
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        logits = torch.empty((window, cfg["vocab_size"]), dtype=torch.float32, device="cuda")
        for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask),
                             ("logits", logits)):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        rows = logits.cpu().tolist()
        return [rows[index] for index in masked]

    worst, _ = _check_logits(run(cfg), reference, "fp16")
    broken = run(dict(cfg, token_dropout=False))
    broken_worst = max(max(abs(a - b) for a, b in zip(row, want))
                       for row, want in zip(broken, reference))
    with pytest.raises(AssertionError, match="fp16 logits differ"):
        _check_logits(broken, reference, "fp16")
    print(
        f"{FAMILY} fp16 gate: correct maxabs {worst:.3e}; masked rows not zeroed "
        f"maxabs {broken_worst:.3e}"
    )
