# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and C/C++ SDK, and official-reference E2E for Voxtral.

The oracle is the official Transformers ``VoxtralForConditionalGeneration``,
loaded by Transformers from the same checkpoint and run in float32
(``reference.py``):

- float32 bundles must reproduce its greedy token ids and transcript exactly;
- fp16 / bf16 bundles are checked teacher-forced: every native token, and the
  end-of-sequence token where the native run stopped, must be the float32
  model's argmax or within ``NEAR_TIE_LOGIT_GAP`` of it given the native prefix.
  Greedy decoding flips at near-ties and the texts then diverge, so free-running
  text equality is not a reduced-precision contract. The allowance is the
  reduced-precision engines' own logit error against float32, rounded up: a token
  that trails the argmax by less is a rounding-level near tie, a larger gap is a
  defect (calibration and negative controls beside the constant).

The CLI and both SDK consumers must return identical ids and text. Every case
records its input audio, the native and reference transcripts and token ids,
and the enforced checks as evidence for the E2E report.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "voxtral"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SAMPLE_RATE = 16000
JOIN_SILENCE_SECONDS = 1.5
# Calibration on mistralai/Voxtral-Mini-3B-2507 @ 3060fe34, every output scored teacher-forced by the float32 model:
# - healthy: the official Transformers pipeline in bf16 and in fp16 on five inputs (the recording and the LibriSpeech
#   clip, each with `en` and automatic language, and the two-chunk clip; 336 scored steps), and this test's fp16 and
#   bf16 cases (249 scored steps), pick the float32 argmax at every step (largest gap 0.000); the engines' last
#   prompt-row logits differ from float32 by at most 0.18 (fp16) and 0.19 (bf16) on the LibriSpeech clip;
# - defective, in the Transformers model and in fp16 engines alike: on the three longer inputs the largest gaps are
#   10.0 to 17.5 with the rotary base 1e4 instead of 1e8, and 0.30 to 0.31 without the last decoder layer's MLP; on the
#   4 s recording both defective models still produce the correct transcript. Every defective output that differs
#   from float32 is rejected: the allowance sits 0.2 above the healthy runs and 0.1 below the closest defect.
NEAR_TIE_LOGIT_GAP = 0.2
# One bundle per manifest and one float32 reference per checkpoint for the session.
_BUNDLES: dict[str, Path] = {}
_REFERENCES: dict[str, object] = {}


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY, path
        assert manifest["task"] in TASKS, path
        assert manifest["tensor_parallel_size"] == 1, path
        assert isinstance(manifest["max_sequence_length"], int), path
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result, name
            assert isinstance(case.get("premerge", False), bool), name
            # A case that must fill its token budget (to reach a late cache position, for example).
            assert isinstance(case.get("stops_at_budget", False), bool), name
            result[name] = (path, manifest, case)
    assert result, f"{FAMILY} has no E2E cases"
    return result


CASES = _case_index()


def _csv_values(values) -> set[str]:
    return {item.strip() for value in values or [] for item in str(value).split(",") if item.strip()}


def _selection(config) -> tuple[set[str], set[str]]:
    """Names selected by --e2e-model or --e2e-models-file, and the cases selected by --e2e-testcase."""
    models = _csv_values(config.getoption("--e2e-model"))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        models |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return models, _csv_values(config.getoption("--e2e-testcase"))


def _collected_cases(items) -> set[str]:
    """The E2E cases of every family collected in this session: the ``case_name`` parameter of each
    ``test_*e2e`` item."""
    cases = set()
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        if re.fullmatch(r"test_.*e2e", getattr(item, "originalname", "") or "") and "case_name" in parameters:
            cases.add(str(parameters["case_name"]))
    return cases


def _require_selected(case_name: str, config, collected: set[str]) -> None:
    """Skip a case that is not selected; fail a selection that names this family but none of its cases."""
    models, testcases = _selection(config)
    if not (models or testcases):
        pytest.skip("direct E2E requires --e2e-model, --e2e-models-file or --e2e-testcase")
    owned = {name for name, (_, manifest, _) in CASES.items() if {FAMILY, manifest["name"], name} & models}
    if models and case_name not in owned:
        pytest.skip(f"{case_name} was not selected")
    if testcases and case_name not in testcases:
        # A selection that shares no case with this family's selected cases, or, without --e2e-model, with the E2E
        # cases collected in this session, is wrong and must not pass by running none.
        if models and not owned & testcases:
            pytest.fail(f"--e2e-testcase {sorted(testcases)} matches none of the selected {FAMILY} cases: "
                        f"{sorted(owned)}")
        if not models and not testcases & collected:
            pytest.fail(f"--e2e-testcase {sorted(testcases)} matches no E2E case collected in this session")
        pytest.skip(f"{case_name} was not selected")


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    """The manifest's pinned snapshot, from the local Hugging Face cache only (the pre-merge runs offline)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    hf_id, revision = manifest["hf_id"], manifest["hf_revision"]
    try:
        snapshot = Path(snapshot_download(repo_id=hf_id, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError as error:
        raise AssertionError(f"{hf_id} at {revision} is not in the Hugging Face cache; download it first "
                             f"(hf download {hf_id} --revision {revision})") from error
    assert (snapshot / "config.json").is_file(), snapshot
    return snapshot


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


def _teacher_forced_gaps(reference, wav: Path, case: dict, native_ids: list[int]) -> list[float]:
    """Float32 logit gap between the reference argmax and each native token (plus the end-of-sequence token
    where the native run stopped), every step given the native prefix."""
    import torch

    logits = reference.teacher_forced_logits(wav, case.get("source_language"), native_ids)
    targets = list(native_ids)
    eos = reference.model.generation_config.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    if len(native_ids) < int(case["max_new_tokens"]):
        targets.append(int(eos))
    assert logits.shape[0] >= len(targets), (tuple(logits.shape), len(targets))
    rows = logits[: len(targets)]
    assert bool(torch.isfinite(rows).all()), "the float32 reference produced non-finite logits"
    chosen = rows.gather(1, torch.tensor(targets, device=rows.device)[:, None])[:, 0]
    return (rows.max(dim=1).values - chosen).tolist()


def _largest_gap(gaps: list[float]) -> float:
    """The largest gap; a non-finite gap is an error, never a pass (``max`` would skip a NaN)."""
    if not all(math.isfinite(gap) for gap in gaps):
        raise ValueError(f"non-finite teacher-forced gap in {gaps}")
    return max(gaps, default=0.0)


def _token_differences(native: list[int], reference: list[int]) -> int:
    """Positions whose token differs, plus the length difference."""
    return sum(a != b for a, b in zip(native, reference)) + abs(len(native) - len(reference))


def _check(name: str, label: str, scope: str, actual, operator: str, expected) -> dict:
    passed = {"==": actual == expected, "<=": actual <= expected, ">=": actual >= expected}[operator]
    return {"name": name, "label": label, "scope": scope, "actual": actual, "operator": operator,
            "expected": expected, "passed": bool(passed)}


def _comparison_checks(precision: str, native_ids: list[int], native_text: str, reference_ids: list[int],
                       reference_text: str, gaps: list[float] | None, sdk_differences: int,
                       official_text: str) -> list[dict]:
    """The enforced checks; ``official_text`` is the official processor's decoding of the native ids."""
    reference = "independent_reference"
    checks = [
        _check("empty_transcript", "Native run produced no token or only whitespace (1 = empty)", "contract",
               int(not native_ids or not native_text.strip()), "==", 0),
        _check("sdk_differences", "C and C++ SDK results whose ids or text differ from the CLI", "contract",
               sdk_differences, "==", 0),
        _check("native_decoding_differs", "Native text differs from the official processor's decoding of the "
               "native ids (1 = differs)", reference, int(native_text != official_text), "==", 0),
    ]
    if precision == "fp32":
        checks += [
            _check("token_id_differences", "Token positions that differ from the reference, plus the length "
                   "difference", reference, _token_differences(native_ids, reference_ids), "==", 0),
            _check("transcript_differs", "Transcript differs from the reference transcript (1 = differs)",
                   reference, int(native_text != reference_text), "==", 0),
        ]
    else:
        assert gaps is not None
        checks.append(_check(
            "largest_teacher_forced_gap", "Largest float32 logit gap below the reference argmax of a native token, "
            "teacher-forced", reference, float(_largest_gap(gaps)), "<=", NEAR_TIE_LOGIT_GAP))
    return checks


def _record_comparison(case_name: str, precision: str, directory: Path, native: dict, reference: dict,
                       checks: list[dict]) -> None:
    """Record the case's enforced checks for the E2E report; never raises, so the test keeps its own result."""
    if not evidence_enabled():
        return
    try:
        native_file = directory / "native.json"
        reference_file = directory / "reference.json"
        native_file.write_text(json.dumps(native), encoding="utf-8")
        reference_file.write_text(json.dumps(reference), encoding="utf-8")
    except Exception as error:
        record_evidence("comparison_not_recorded", f"{type(error).__name__}: {error}")
        return
    record_evidence("reference_comparison", {
        "label": f"{case_name}: native {precision} bundle vs the Transformers float32 model",
        "scope": "independent_reference",
        "enforced": True,
        "native": native_file,
        "reference": reference_file,
        "checks": checks,
    })


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name: str, request, tmp_path: Path, bundle_dir: Path) -> None:
    manifest_path, manifest, case = CASES[case_name]
    _require_selected(case_name, request.config, _collected_cases(request.session.items))
    assert manifest_path.is_file()
    precision = manifest["precision"]
    binary, runtime_root, native = _runtime()
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest["hf_id"],
                                   "hf_revision": manifest["hf_revision"]})
    with evidence_stage("build"):
        bundle = _bundle(manifest_path, manifest, model_dir, bundle_dir)

    wav = _input_audio(case, tmp_path)
    record_evidence("inputs", {
        "manifest": manifest,
        "case": case,
        "audio": wav,
        "source_language": case.get("source_language") or "auto",
        "max_new_tokens": int(case["max_new_tokens"]),
    })
    record_evidence("thresholds", {"exact_token_ids": precision == "fp32",
                                   "near_tie_logit_gap": None if precision == "fp32" else NEAR_TIE_LOGIT_GAP})
    with evidence_stage("native"):
        cli = _native_cli(binary, runtime_root, bundle, wav, case)
        native_ids = [int(value) for value in cli["token_ids"]]
        native_text = str(cli["text"])
        sdk = _native_sdk(native, runtime_root, bundle, wav, case, tmp_path)
    sdk_differences = sum(result.get("token_ids") != native_ids or result.get("text") != native_text
                          for result in sdk.values())
    native_record = {"transcript": native_text, "token_ids": native_ids, "precision": precision,
                     "sdk": {target: {"text": result.get("text"), "token_ids": result.get("token_ids")}
                             for target, result in sdk.items()}}
    record_evidence("native", native_record)
    print(f"{FAMILY} {precision} native: {native_text!r}")

    with evidence_stage("reference"):
        reference = _reference(model_dir)
        reference_ids, reference_text, _ = reference.transcribe(
            wav, case.get("source_language"), int(case["max_new_tokens"]))
        gaps = None if precision == "fp32" else _teacher_forced_gaps(reference, wav, case, native_ids)
        official_text = reference.decode(native_ids)
    import transformers

    reference_record = {"reference_text": reference_text, "reference_ids": reference_ids,
                        "official_decoding_of_native_ids": official_text,
                        "backend": f"transformers {transformers.__version__} VoxtralForConditionalGeneration, float32",
                        "weights": f"{manifest['hf_id']}@{manifest['hf_revision']}"}
    if gaps is not None:
        reference_record["teacher_forced_gaps"] = [round(gap, 6) for gap in gaps]
    record_evidence("reference", reference_record)
    print(f"{FAMILY} float32 reference: {reference_text!r}")

    with evidence_stage("compare"):
        checks = _comparison_checks(precision, native_ids, native_text, reference_ids, reference_text, gaps,
                                    sdk_differences, official_text)
        budget = int(case["max_new_tokens"])
        if case.get("stops_at_budget", False):
            checks.append(_check("tokens_short_of_budget", "Tokens short of the budget this case must fill",
                                 "contract", budget - len(native_ids), "==", 0))
        try:
            if case.get("stops_at_budget", False):
                assert len(native_ids) == budget, f"the native run stopped after {len(native_ids)} of {budget} tokens"
            assert native_ids and native_text.strip(), "the native run produced an empty transcript"
            assert sdk_differences == 0, f"SDK consumers differ from the CLI: {sdk}"
            assert native_text == official_text, f"native decoding {native_text!r} vs official {official_text!r}"
            if precision == "fp32":
                assert native_ids == reference_ids, (
                    f"token ids differ: native {len(native_ids)}, reference {len(reference_ids)}")
                assert native_text == reference_text, f"transcript mismatch: {native_text!r} vs {reference_text!r}"
            else:
                largest = _largest_gap(gaps)
                exact = sum(1 for gap in gaps if gap == 0.0)
                print(f"{FAMILY} {precision} teacher-forced: {exact}/{len(gaps)} reference argmax, "
                      f"largest gap {largest:.4f} (allowance {NEAR_TIE_LOGIT_GAP})")
                assert largest <= NEAR_TIE_LOGIT_GAP, (
                    f"native token at step {gaps.index(largest)} trails the float32 argmax by {largest:.4f} logit")
            assert all(check["passed"] for check in checks), checks
        finally:
            _record_comparison(case_name, precision, tmp_path, native_record, reference_record, checks)
