# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned MiniCPM-V 4.7 E2E: fresh BF16 bundle, native CLI, official Transformers oracle.

Each selected case checks, against the provider (``tests/hf_reference.py`` in the reference
environment named by ``TRTMC_MINICPMV47_REFERENCE_PYTHON``):

1. the host request contract, exactly: prompt token ids, crop grids, normalized pixel patches,
   vision position ids and canvas M-RoPE positions (``minicpmv4_7_preprocess_dump``);
2. the native CLI's greedy tokens under teacher forcing by the FP32 official model: every token is
   the reference argmax or within ``near_tie_logit`` of the reference maximum
   (``thresholds/<case>.json``);
3. ``expected_answers`` when the case lists them.

With ``TRTMC_E2E_ARTIFACT_DIR`` set, each case records its prompt and image, the native and the
reference text and token ids, and the checks of 1 and 2 as a reference comparison
(``tools.e2e_evidence``).

``near_tie_logit`` = 0.85 comes from a four-prompt calibration of the official BF16 model against the
official FP32 model on this image: three prompts gave identical tokens, and on the fourth the BF16 model
first left the FP32 greedy choice at an FP32 gap of 0.811, rounded up to 0.85. The value bounds that
first BF16 near-tie flip only; later in the same diverged sequence the official BF16 model chose a
token 1.92 below the FP32 maximum, which this check would reject. All cases use the same value.

The multilingual case holds Devanagari, Bengali and Thai text and a decomposed (NFD) sequence, so the
host contract also covers the tokenizer's NFC normalization and Unicode pre-tokenization.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

TEST_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((TEST_DIR / "manifests/minicpm-v-4.7-35b-a3b.json").read_text(encoding="utf-8"))
CASES = {case["name"]: case for case in MANIFEST["testcases"]}
_FAMILY = "minicpmv4_7"


def _selected(config, case_name: str) -> bool:
    selected = {
        item.strip()
        for flag in ("--e2e-model", "--e2e-testcase")
        for value in (config.getoption(flag, default=[]) or [])
        for item in value.split(",")
        if item.strip()
    }
    filename = config.getoption("--e2e-models-file", default=None)
    if filename:
        selected.update(line.strip() for line in Path(filename).read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#"))
    if not selected:
        return os.environ.get("TRTMC_E2E") == "1"
    return bool({case_name, MANIFEST["name"], _FAMILY} & selected)


def _environment() -> tuple[Path, Path, Path, Path]:
    binary = Path(os.environ["TRTMC_BINARY"])
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    native = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR") or runtime)
    reference = Path(os.environ.get("TRTMC_MINICPMV47_REFERENCE_PYTHON", sys.executable))
    assert binary.is_file() and os.access(binary, os.X_OK), binary
    assert (runtime / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime
    assert (native / "minicpmv4_7_preprocess_dump").is_file(), native
    assert reference.is_file(), reference
    return binary, runtime, native, reference


def _eos_ids(checkpoint: Path) -> set[int]:
    """End-of-sequence ids the builder writes to the bundle (``generation_config.json``)."""
    value = json.loads((checkpoint / "generation_config.json").read_text(encoding="utf-8"))["eos_token_id"]
    return {value} if isinstance(value, int) else set(value)


def _thresholds(case_name: str) -> dict:
    path = TEST_DIR / "thresholds" / f"{case_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"] if path.is_file() else {}


def _run(command: list[str], log: Path, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    log.write_text(f"$ {' '.join(command)}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}\n")
    assert result.returncode == 0, f"{command[0]} failed, see {log}"
    return result


def _request(case: dict, directory: Path) -> Path:
    """One user turn, image then text, as the native CLI sends it."""
    from PIL import Image

    image = Image.open(TEST_DIR / case["image"]).convert("RGB")
    (directory / "image.rgb").write_bytes(image.tobytes())
    (directory / "image.png").write_bytes((TEST_DIR / case["image"]).read_bytes())
    request = {
        "enable_thinking": False,
        "messages": [{"role": "user", "parts": [
            {"image": "image.png", "rgb": "image.rgb", "height": image.height, "width": image.width},
            {"text": case["prompt"]}]}],
    }
    path = directory / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _differing(ours, theirs) -> int:
    return sum(a != b for a, b in zip(ours, theirs)) + abs(len(ours) - len(theirs))


def _check(name: str, label: str, actual, operator: str, expected) -> dict:
    passed = actual == expected if operator == "==" else actual <= expected
    return {"name": name, "label": label, "scope": "independent_reference", "actual": actual,
            "operator": operator, "expected": expected, "passed": passed}


def _record_comparison(case: dict, work: Path, native_dump: dict, reference: dict, tolerance: float) -> None:
    """The checks the assertions below enforce, against the provider and the FP32 official model."""
    ours = np.fromfile(work / "native" / "patches.f32", dtype="<f4")
    theirs = np.fromfile((work / "reference.json").with_suffix(".patches.f32"), dtype="<f4")
    pixels = int(np.count_nonzero(ours != theirs)) if ours.shape == theirs.shape else max(ours.size, theirs.size)
    layout = _differing(
        [[crop["grid"] for crop in native_dump["crops"]],
         [crop["position_ids"] for crop in native_dump["crops"]],
         native_dump["positions"], native_dump["decode_position"]],
        [reference["crop_grids"], reference["crop_position_ids"], reference["positions"],
         reference["decode_position"]])
    gap = float(max(reference["teacher_forced"]["candidate_gap"], default=0.0))
    record_evidence("reference_comparison", {
        "label": case["prompt"], "scope": "independent_reference", "enforced": True,
        "native": work / "native.json", "reference": work / "reference.json",
        "checks": [
            _check("prompt_token_ids", "Prompt token ids unlike the provider's",
                   _differing(native_dump["input_ids"], reference["input_ids"]), "==", 0),
            _check("image_layout", "Crop grids, vision position ids and M-RoPE positions unlike the provider's",
                   layout, "==", 0),
            _check("normalized_pixels", "Normalized pixel values unlike the provider's", pixels, "==", 0),
            _check("max_teacher_forced_gap", "Largest FP32 logit gap below the argmax, native tokens forced",
                   gap, "<=", tolerance),
        ],
    })


def _check_host_contract(native: dict, reference: dict, patches: Path, reference_patches: Path) -> None:
    assert native["input_ids"] == reference["input_ids"], "prompt token ids differ from the provider"
    assert [crop["grid"] for crop in native["crops"]] == reference["crop_grids"], "crop grids differ"
    assert [crop["position_ids"] for crop in native["crops"]] == reference["crop_position_ids"], \
        "vision position ids differ"
    assert native["positions"] == reference["positions"], "canvas M-RoPE positions differ"
    assert native["decode_position"] == reference["decode_position"]
    ours = np.fromfile(patches, dtype="<f4")
    theirs = np.fromfile(reference_patches, dtype="<f4")
    assert ours.shape == theirs.shape and np.array_equal(ours, theirs), "normalized pixels differ"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, Path]:
    """One fresh bundle from the pinned checkpoint, shared by the selected cases.

    The 65.6 GiB checkpoint is read from the local Hugging Face cache only; the test never downloads it.
    """
    from huggingface_hub import snapshot_download

    try:
        checkpoint = Path(snapshot_download(repo_id=MANIFEST["hf_id"], revision=MANIFEST["hf_revision"],
                                            local_files_only=True))
    except FileNotFoundError as error:
        raise AssertionError(
            f"{MANIFEST['hf_id']} at revision {MANIFEST['hf_revision']} is not in the local Hugging Face cache; "
            f"fetch it first: hf download {MANIFEST['hf_id']} --revision {MANIFEST['hf_revision']}") from error
    bundle = tmp_path_factory.mktemp("bundle") / MANIFEST["bundle"]
    build(BuildRequest(checkpoint, bundle, MANIFEST["family"], MANIFEST["task"], MANIFEST["precision"],
                       max_sequence_length=MANIFEST["max_sequence_length"],
                       tensor_parallel_size=MANIFEST["tensor_parallel_size"]))
    return checkpoint, bundle


@pytest.mark.parametrize("case_name", list(CASES))
def test_official_checkpoint_e2e(case_name: str, tmp_path: Path, request) -> None:
    case = CASES[case_name]
    if not _selected(request.config, case_name):
        pytest.skip("real family E2E requires explicit selection or TRTMC_E2E=1")
    binary, runtime, native, reference_python = _environment()
    checkpoint, bundle = request.getfixturevalue("built")
    work = tmp_path / case_name
    work.mkdir()
    request_path = _request(case, work)
    root = TEST_DIR.parents[2]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}

    record_evidence("inputs", {"prompt": case["prompt"], "image": TEST_DIR / case["image"],
                               "max_new_tokens": case["max_new_tokens"]})
    with evidence_stage("native"):
        _run([str(native / "minicpmv4_7_preprocess_dump"), str(bundle), str(request_path), str(work / "native")],
             work / "preprocess.log")
        cli = _run([str(binary), "run", str(bundle), "--runtime-root", str(runtime), "--image",
                    str(TEST_DIR / case["image"]), "--prompt", case["prompt"], "--max-new-tokens",
                    str(case["max_new_tokens"])], work / "cli.log", timeout=3600)
    result = json.loads(cli.stdout)
    tokens = [int(token) for token in result["token_ids"]]
    output = {"text": result["text"], "token_ids": tokens}
    (work / "native.json").write_text(json.dumps(output), encoding="utf-8")
    (work / "candidate.json").write_text(json.dumps(tokens))
    record_evidence("native", output)
    with evidence_stage("reference"):
        _run([str(reference_python), "-m", "families.minicpmv4_7.tests.hf_reference", "--checkpoint",
              str(checkpoint), "--request", str(request_path), "--output", str(work / "reference.json"),
              "--dtype", MANIFEST["reference_precision"], "--candidate", str(work / "candidate.json"),
              "--max-new-tokens", str(case["max_new_tokens"])], work / "reference.log", env=environment,
             timeout=7200, cwd=root)
    reference = json.loads((work / "reference.json").read_text())
    record_evidence("reference", {"backend": "hf_transformers", "precision": MANIFEST["reference_precision"],
                                  "reference_text": reference["generated_text"],
                                  "reference_ids": reference["generated_ids"]})
    native_dump = json.loads((work / "native" / "dump.json").read_text())
    tolerance = float(_thresholds(case_name).get("near_tie_logit", 0.0))
    record_evidence("thresholds", {"near_tie_logit": tolerance})
    _record_comparison(case, work, native_dump, reference, tolerance)

    with evidence_stage("compare"):
        _check_host_contract(native_dump, reference, work / "native" / "patches.f32",
                             (work / "reference.json").with_suffix(".patches.f32"))
        teacher = reference["teacher_forced"]
        assert teacher["candidate"] == tokens and len(tokens) <= case["max_new_tokens"]
        # Generation stops at an end-of-sequence id (kept in the ids) or at the budget, never early without one.
        assert tokens and (tokens[-1] in _eos_ids(checkpoint) or len(tokens) == case["max_new_tokens"]), tokens
        for index, (token, gap) in enumerate(zip(teacher["candidate"], teacher["candidate_gap"])):
            assert gap <= tolerance, (
                f"token {index} ({token}) is {gap:.4f} below the FP32 reference maximum; "
                f"reference argmax {teacher['argmax'][index]}")
        for answer in case.get("expected_answers", []):
            assert answer.lower() in result["text"].lower(), result["text"]
