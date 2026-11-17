# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and C/C++ SDK, and official-reference E2E for Voxtral.

The oracle is the official Transformers model run in float32 on the export's
exactly dequantized weights (``reference.py``):

- float32 bundles must reproduce its greedy token ids and transcript exactly;
- fp16 / bf16 bundles are checked teacher-forced: every native token, and the
  end-of-sequence token where the native run stopped, must be the float32
  model's argmax or within ``NEAR_TIE_LOGIT_GAP`` of it given the native prefix.
  Greedy decoding flips at near-ties (on these clips the float32 model's two
  best tokens are 0.023 logit apart at a comma and 0.167 at a hyphen) and the
  texts then diverge, so free-running text equality is not a reduced-precision
  contract. The allowance is the largest such gap of the official Transformers
  pipeline itself run in bf16 on these inputs (0.167 logit), rounded up. Its fp16
  run matched every float32 decision there, but fp16 bundles get the same
  allowance: their last prompt logits sit 0.13-0.18 logit from float32 (the
  official fp16 model's 0.06), the same order as the near-ties above.

The CLI and both SDK consumers must return identical ids and text.
"""

from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "voxtral"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SAMPLE_RATE = 16000
JOIN_SILENCE_SECONDS = 1.5
NEAR_TIE_LOGIT_GAP = 0.2
# One bundle per manifest and one float32 reference per checkpoint for the session.
_BUNDLES: dict[str, Path] = {}
_REFERENCES: dict[str, object] = {}


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
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(name, marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"))
                for name in names
            ]
        metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_VOXTRAL_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_VOXTRAL_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"],
                                     local_files_only=True)
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, native


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(BuildRequest(
        model_dir=model_dir,
        output_path=bundle,
        family=FAMILY,
        task=manifest["task"],
        precision=manifest["precision"],
        max_sequence_length=manifest.get("max_sequence_length"),
        tensor_parallel_size=int(manifest["tensor_parallel_size"]),
    ))


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp(f"{FAMILY}-bundles")


def _bundle(manifest_path: Path, manifest: dict, model_dir: Path, directory: Path) -> Path:
    key = str(manifest_path)
    if key not in _BUNDLES:
        bundle = directory / str(manifest["bundle"])
        _build(model_dir, bundle, manifest)
        assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"
        _BUNDLES[key] = bundle
    return _BUNDLES[key]


def _reference(model_dir: Path):
    from families.voxtral.tests.reference import VoxtralReference

    key = str(model_dir)
    if key not in _REFERENCES:
        _REFERENCES[key] = VoxtralReference(model_dir)
    return _REFERENCES[key]


def _read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE, f"{path.name} is not {SAMPLE_RATE} Hz"
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")


def _input_audio(case: dict, directory: Path) -> Path:
    """The case's WAV, or its WAV list joined with silence, as 16-bit mono PCM."""
    assets = case["test_input_audio"]
    assets = [assets] if isinstance(assets, str) else list(assets)
    pieces = []
    for index, asset in enumerate(assets):
        path = TEST_ROOT / asset
        assert path.is_file(), f"{FAMILY} E2E asset does not exist: {path}"
        if index:
            pieces.append(np.zeros(int(JOIN_SILENCE_SECONDS * SAMPLE_RATE), dtype="<i2"))
        pieces.append(_read_pcm16(path))
    if len(assets) == 1:
        return TEST_ROOT / assets[0]
    joined = directory / f"{case['name']}.wav"
    with wave.open(str(joined), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.concatenate(pieces).tobytes())
    return joined


def _json_lines(stdout: str) -> dict:
    payloads = []
    for line in stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                payloads.append(json.loads(line[start:]))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native run returned no JSON: {stdout[-1000:]}"
    return payloads[-1]


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    return env


def _native_cli(binary: Path, runtime_root: Path, bundle: Path, wav: Path, case: dict) -> dict:
    language = ["--source-language", case["source_language"]] if case.get("source_language") else []
    completed = subprocess.run(
        [str(binary), "transcribe", str(bundle), "--runtime-root", str(runtime_root), "--input",
         str(wav), "--set", f"max_new_tokens={case['max_new_tokens']}", *language],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=1800)
    return _json_lines(completed.stdout)


def _native_sdk(native: Path, runtime_root: Path, bundle: Path, wav: Path, case: dict,
                directory: Path) -> dict[str, dict]:
    targets = [f"test_{FAMILY}_sdk_c", f"test_{FAMILY}_sdk_cpp"]
    subprocess.run(["cmake", "--build", str(native), "--parallel", "8", "--target", *targets],
                   check=True, timeout=1800)
    pcm = directory / f"{case['name']}.f32"
    (_read_pcm16(wav).astype(np.float32) / 32768.0).tofile(pcm)
    language = [case["source_language"]] if case.get("source_language") else []
    results = {}
    for target in targets:
        completed = subprocess.run(
            [str(native / target), str(bundle), str(runtime_root), str(pcm), str(SAMPLE_RATE), "1",
             str(case["max_new_tokens"]), *language],
            check=True, capture_output=True, text=True, env=_environment(runtime_root),
            timeout=1800)
        results[target] = _json_lines(completed.stdout)
    return results


def _assert_teacher_forced(reference, wav: Path, case: dict, native_ids: list[int],
                           precision: str) -> None:
    import torch

    allowance = NEAR_TIE_LOGIT_GAP
    logits = reference.teacher_forced_logits(wav, case.get("source_language"), native_ids)
    targets = list(native_ids)
    eos = reference.model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    stopped = len(native_ids) < int(case["max_new_tokens"])
    if stopped:
        targets.append(int(eos))
    rows = logits[: len(targets)]
    chosen = rows.gather(1, torch.tensor(targets, device=rows.device)[:, None])[:, 0]
    gaps = (rows.max(dim=1).values - chosen).tolist()
    worst = max(gaps) if gaps else 0.0
    exact = sum(1 for gap in gaps if gap == 0.0)
    print(f"{FAMILY} {precision} teacher-forced: {exact}/{len(gaps)} reference argmax, "
          f"largest gap {worst:.4f} (allowance {allowance})")
    assert worst <= allowance, (
        f"native token at step {gaps.index(worst)} trails the float32 argmax by {worst:.4f} logit")


def test_e2e(tmp_path: Path, bundle_dir: Path, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root, native = _runtime()
    model_dir = _model_dir(manifest)
    bundle = _bundle(manifest_path, manifest, model_dir, bundle_dir)

    wav = _input_audio(case, tmp_path)
    cli = _native_cli(binary, runtime_root, bundle, wav, case)
    native_ids = [int(value) for value in cli["token_ids"]]
    text = str(cli["text"])
    assert native_ids and text.strip(), "the native run produced an empty transcript"
    for target, result in _native_sdk(native, runtime_root, bundle, wav, case, tmp_path).items():
        assert result["token_ids"] == native_ids and result["text"] == text, (
            f"{target} differs from the CLI")

    reference = _reference(model_dir)
    reference_ids, reference_text, _ = reference.transcribe(
        wav, case.get("source_language"), int(case["max_new_tokens"]))
    print(f"{FAMILY} {manifest['precision']} native: {text!r}")
    print(f"{FAMILY} float32 reference: {reference_text!r}")
    if manifest["precision"] == "fp32":
        assert native_ids == reference_ids, (
            f"token ids differ: native {len(native_ids)}, reference {len(reference_ids)}")
        assert text == reference_text, f"transcript mismatch: {text!r} vs {reference_text!r}"
    else:
        _assert_teacher_forced(reference, wav, case, native_ids, manifest["precision"])
