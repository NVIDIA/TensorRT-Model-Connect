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

TEST_DIR = Path(__file__).resolve().parent
MANIFEST = json.loads((TEST_DIR / "manifests/minicpm-v-4.7-35b-a3b.json").read_text(encoding="utf-8"))
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
    """One fresh bundle from the pinned checkpoint, shared by the selected cases."""
    from huggingface_hub import constants, snapshot_download

    checkpoint = Path(snapshot_download(repo_id=MANIFEST["hf_id"], revision=MANIFEST["hf_revision"],
                                        local_files_only=constants.HF_HUB_OFFLINE))
    bundle = tmp_path_factory.mktemp("bundle") / MANIFEST["bundle"]
    build(BuildRequest(checkpoint, bundle, MANIFEST["family"], MANIFEST["task"], MANIFEST["precision"],
                       max_sequence_length=MANIFEST["max_sequence_length"],
                       tensor_parallel_size=MANIFEST["tensor_parallel_size"]))
    return checkpoint, bundle


@pytest.mark.parametrize("case", MANIFEST["testcases"], ids=lambda case: case["name"])
def test_official_checkpoint_e2e(case: dict, tmp_path: Path, request) -> None:
    if not _selected(request.config, case["name"]):
        pytest.skip("real family E2E requires explicit selection or TRTMC_E2E=1")
    binary, runtime, native, reference_python = _environment()
    checkpoint, bundle = request.getfixturevalue("built")
    work = tmp_path / case["name"]
    work.mkdir()
    request_path = _request(case, work)
    root = TEST_DIR.parents[2]
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}

    _run([str(native / "minicpmv4_7_preprocess_dump"), str(bundle), str(request_path), str(work / "native")],
         work / "preprocess.log")
    cli = _run([str(binary), "run", str(bundle), "--runtime-root", str(runtime), "--image",
                str(TEST_DIR / case["image"]), "--prompt", case["prompt"], "--max-new-tokens",
                str(case["max_new_tokens"])], work / "cli.log", timeout=3600)
    result = json.loads(cli.stdout)
    tokens = [int(token) for token in result["token_ids"]]
    (work / "candidate.json").write_text(json.dumps(tokens))
    _run([str(reference_python), "-m", "families.minicpmv4_7.tests.hf_reference", "--checkpoint",
          str(checkpoint), "--request", str(request_path), "--output", str(work / "reference.json"),
          "--dtype", MANIFEST["reference_precision"], "--candidate", str(work / "candidate.json"),
          "--max-new-tokens", str(case["max_new_tokens"])], work / "reference.log", env=environment,
         timeout=7200, cwd=root)

    reference = json.loads((work / "reference.json").read_text())
    native_dump = json.loads((work / "native" / "dump.json").read_text())
    _check_host_contract(native_dump, reference, work / "native" / "patches.f32",
                         (work / "reference.json").with_suffix(".patches.f32"))

    teacher = reference["teacher_forced"]
    tolerance = float(_thresholds(case["name"]).get("near_tie_logit", 0.0))
    assert teacher["candidate"] == tokens and len(tokens) <= case["max_new_tokens"]
    for index, (token, gap) in enumerate(zip(teacher["candidate"], teacher["candidate_gap"])):
        assert gap <= tolerance, (
            f"token {index} ({token}) is {gap:.4f} below the FP32 reference maximum; "
            f"reference argmax {teacher['argmax'][index]}")
    for answer in case.get("expected_answers", []):
        assert answer.lower() in result["text"].lower(), result["text"]
    (work / "evidence.json").write_text(json.dumps({
        "case": case["name"], "native_tokens": tokens, "native_text": result["text"],
        "reference_tokens": reference["generated_ids"], "reference_text": reference["generated_text"],
        "max_candidate_gap": max(teacher["candidate_gap"]), "near_tie_logit": tolerance,
        "token_exact": tokens == reference["generated_ids"][:len(tokens)]}, indent=2))
