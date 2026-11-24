# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and reference E2E for VibeVoice-ASR.

One bundle per manifest serves all of its cases. Each case reads one PCM16 file
(the committed clip, or a 24 kHz, repeated or padded version written at run time), runs
the native ``trtmc transcribe`` on it twice (the default transcript and
``raw_output``, which keeps the speaker ids) and the reference once: the
transformers implementation of the model on Microsoft's transformers-layout
release of the same weights, with the sigma-VAE mean instead of its sampling
noise (``transformers_reference.py``). Both native runs must generate the same
ids, and the default transcript must be the raw output's contents and
timestamps. Oracles:

* ``tokens``: used wherever both sides read the same 24 kHz samples, which then
  end on a whole acoustic frame (``_case_audio``). Either the
  ids equal the reference ones, with identical raw output and transcript; or the
  runs part at a near-tie of the reference: at the first differing step the
  native token's reference logit is at most ``NEAR_TIE_BF16_STEPS`` BF16 values
  below the reference best (the reference logits are BF16). Ids after that step
  are not comparable position by position, so the segments must then agree:
  same count, identical text and speaker, and timestamps within one acoustic
  frame (3200 samples at 24 kHz).
* ``transcript_words``: the transcripts have the same words, each with the same
  speaker. Used
  for 16 kHz input, which the reference resamples with torchaudio and the
  runtime with its own windowed-sinc filter, so the samples differ.
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

from families.vibevoice_asr.model import TOKENIZER_REPOSITORY, TOKENIZER_REVISION
from families.vibevoice_asr.tests.transformers_reference import (
    REFERENCE_FILES,
    REFERENCE_REPOSITORY,
    REFERENCE_REVISION,
    run_reference,
)
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "vibevoice_asr"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SILENCE_SECONDS = 1.0
FRAME_SECONDS = 3200 / 24000
# Teacher-forcing the model's original BF16 implementation on 20,000 tokens generated
# natively (a one-hour input, positions 27,052-47,052) gave 48 differing argmaxes: 30
# equal logits, 16 one BF16 value apart and 2 two values apart (logits 19.75-34.5).
# The native runtime's logit noise therefore reaches two BF16 values, and a first
# divergence within that distance is a near-tie, not a defect.
NEAR_TIE_BF16_STEPS = 2
SAMPLE_RATE = 24000
HOP_SAMPLES = 3200


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        # The build reads this tokenizer and the reference these weights; CI must
        # stage exactly these pins.
        assert manifest["hf_dependencies"] == [
            {"repo_id": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION},
            {"repo_id": REFERENCE_REPOSITORY, "revision": REFERENCE_REVISION},
        ]
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
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


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


def _snapshot(repo_id: str, revision: str, allow_patterns=None) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        )
    except Exception as error:
        raise AssertionError(f"selected {FAMILY} E2E requires the cached {repo_id}") from error


def _model_dir(manifest: dict) -> Path:
    # An explicit directory must hold the pinned checkpoint: the reference always
    # loads the pinned transformers-layout release of the same weights.
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    return _snapshot(manifest["hf_id"], manifest.get("hf_revision"))


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


_BUNDLES: dict[str, Path] = {}


def _bundle(manifest: dict, model_dir: Path, root: Path) -> Path:
    """Build each manifest's bundle once per session; every case reuses it."""
    if manifest["name"] not in _BUNDLES:
        bundle = root / str(manifest["bundle"])
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=manifest.get("max_sequence_length"),
                tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            )
        )
        assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"
        _BUNDLES[manifest["name"]] = bundle
    return _BUNDLES[manifest["name"]]


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1
        rate = handle.getframerate()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0, rate


def _write_pcm16(path: Path, samples: np.ndarray, rate: int) -> None:
    """PCM16, which the native reader and the reference both decode as
    ``value / 32768``, so the two sides see identical samples."""
    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Kaiser windowed-sinc rational resampling, only to derive a 24 kHz test input."""
    divisor = int(np.gcd(rate, target))
    up, down = target // divisor, rate // divisor
    spacing = max(up, down)
    half = 32 * spacing
    offsets = np.arange(-half, half + 1)
    taps = np.sinc(offsets / spacing) * np.kaiser(2 * half + 1, 8.6) * up / spacing
    stuffed = np.zeros(samples.size * up)
    stuffed[::up] = samples
    filtered = np.convolve(stuffed, taps)[half : half + stuffed.size]
    return filtered[::down][: -(-samples.size * up // down)].astype(np.float32)


def _case_audio(case: dict, directory: Path) -> tuple[Path, int]:
    """The PCM16 file both sides read and its sample count.

    For the ``tokens`` oracle the audio is padded with zeros to whole acoustic
    frames: the reference pads the waveform itself to a multiple of 3200 samples,
    while the runtime, like the model's original code, pads each strided layer, so
    a partial last frame would differ between them. ``transcript_words`` cases keep
    the partial frame.
    """
    source = TEST_ROOT / str(case["audio"])
    assert source.is_file(), f"selected {FAMILY} E2E audio does not exist: {source}"
    samples, rate = _read_pcm16(source)
    target = int(case["input_sample_rate"])
    repeat = int(case.get("repeat", 1))
    whole_frames = case["oracle"] == "tokens"
    if repeat == 1 and target == rate and not whole_frames:
        return source, samples.size
    if target != rate:
        samples = _resample(samples, rate, target)
    silence = np.zeros(int(SILENCE_SECONDS * target), dtype=np.float32)
    pieces = [samples]
    for _ in range(repeat - 1):
        pieces.extend((silence, samples))
    audio = np.concatenate(pieces)
    if whole_frames:
        assert target == SAMPLE_RATE, f"{case['name']}: the tokens oracle needs 24 kHz input"
        audio = np.pad(audio, (0, -audio.size % HOP_SAMPLES))
    path = directory / f"{case['name']}.wav"
    _write_pcm16(path, audio, target)
    return path, audio.size


def _native(
    binary: Path, runtime_root: Path, bundle: Path, case: dict, audio: Path, raw: bool
) -> dict:
    command = [
        str(binary),
        "transcribe",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--input",
        str(audio),
    ]
    if raw:
        command += ["--set", "raw_output=true"]
    if case.get("context_info"):
        command += ["--set", f"context_info={case['context_info']}"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    payloads = [
        json.loads(line[line.find("{") :])
        for line in completed.stdout.splitlines()
        if line.find("{") >= 0
    ]
    assert payloads, f"native transcribe returned no JSON: {completed.stdout[-1000:]}"
    return payloads[-1]


def _reference(case: dict, audio: Path, directory: Path) -> dict:
    weights = _snapshot(REFERENCE_REPOSITORY, REFERENCE_REVISION, list(REFERENCE_FILES))
    context = case.get("context_info")
    output = directory / f"{case['name']}.reference.json"
    return run_reference(weights, audio, str(context) if context else None, output)


def _segments(raw: str) -> list[dict] | None:
    # The runtime's rule: "assistant" is the role header only when whitespace or the end follows.
    text = raw.strip()
    if re.match(r"assistant(\s|$)", text):
        text = text[len("assistant") :].strip()
    try:
        segments = json.loads(text)
    except json.JSONDecodeError:
        return None
    return segments if isinstance(segments, list) else None


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _bf16_bits(value: float) -> int:
    """The BF16 bit pattern of a logit the reference produced in BF16."""
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    assert bits & 0xFFFF == 0, f"{value} is not a BF16 value"
    return bits >> 16


def _first_difference(native: list[int], reference: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(native, reference)):
        if left != right:
            return index
    return None if len(native) == len(reference) else min(len(native), len(reference))


def _assert_reference_tie(native_ids: list[int], reference: dict, step: int) -> float:
    """At ``step`` the native token's reference logit is a near-tie with the reference best."""
    choice = native_ids[step] if step < len(native_ids) else reference["eos_id"]
    top = reference["top_logits"][step]
    assert choice in top["ids"], f"step {step}: native token {choice} is not in the reference top 5"
    best = top["values"][0]
    value = top["values"][top["ids"].index(choice)]
    # With equal signs, consecutive BF16 values have consecutive bit patterns.
    assert (best >= 0) == (value >= 0), f"step {step}: logits {best} and {value} differ in sign"
    distance = abs(_bf16_bits(best) - _bf16_bits(value))
    assert distance <= NEAR_TIE_BF16_STEPS, (
        f"step {step}: native token {choice} is {distance} BF16 values behind"
    )
    return best - value


def _tie_distance(native_ids: list[int], reference: dict, step: int) -> int:
    """BF16 values from the reference best to the native token's logit at ``step``.

    No recorded logits at ``step``, a token outside the recorded top 5 or a logit
    of the other sign gives 65,536, more than any BF16 distance, so the value
    fails its limit exactly where ``_assert_reference_tie`` fails.
    """
    if step >= len(reference["top_logits"]):
        return 65536
    choice = native_ids[step] if step < len(native_ids) else reference["eos_id"]
    top = reference["top_logits"][step]
    if choice not in top["ids"]:
        return 65536
    best, value = top["values"][0], top["values"][top["ids"].index(choice)]
    if (best >= 0) != (value >= 0):
        return 65536
    return abs(_bf16_bits(best) - _bf16_bits(value))


def _largest_gap(differences) -> float:
    """The largest |difference|; infinite when one is not finite, which max() would skip."""
    gaps = [abs(value) for value in differences]
    return max(gaps, default=0.0) if all(math.isfinite(gap) for gap in gaps) else math.inf


def _check(name: str, label: str, scope: str, actual, operator: str, expected) -> dict:
    passed = {"==": actual == expected, "<=": actual <= expected, ">=": actual >= expected}[
        operator
    ]
    return {
        "name": name,
        "label": label,
        "scope": scope,
        "actual": actual,
        "operator": operator,
        "expected": expected,
        "passed": bool(passed),
    }


def _speaker_words(segments: list[dict]) -> list[tuple[str, object]]:
    """Each word of the segments with its segment's speaker, in order."""
    return [(word, s.get("Speaker")) for s in segments for word in _words(str(s.get("Content", "")))]


def _differences(left: list, right: list) -> int:
    """Positions whose items differ, plus the difference in length."""
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))


def _time(segment: dict, key: str, alias: str) -> float | None:
    """A timestamp under its key or the alias the runtime accepts; None when absent.

    As in the runtime, a timestamp must be a JSON number: a string or a boolean is a
    malformed segment, never converted.
    """
    for name in (key, alias):
        value = segment.get(name)
        if value is None:
            continue
        assert isinstance(value, (int, float)) and not isinstance(value, bool), (
            f"{name} is not a number: {value!r}"
        )
        return float(value)
    return None


def _timed(raw_segments: list[dict]) -> list[tuple[float, float, str]]:
    """The raw segments with both timestamps, the ones the runtime returns as segments."""
    times = [(_time(s, "Start", "Start time"), _time(s, "End", "End time"), s) for s in raw_segments]
    return [(start, end, str(s["Content"])) for start, end, s in times if start is not None and end is not None]


def _transcript_matches(default: dict, raw_segments: list[dict]) -> bool:
    """``_assert_transcript_consistent`` as a value."""
    try:
        _assert_transcript_consistent(default, raw_segments)
    except AssertionError:
        return False
    return True


def _comparison_checks(
    default: dict, raw: dict, reference: dict, case: dict, expected_samples: int | None = None
) -> list[dict]:
    """The assertions of a case after its runs, as check rows of the evidence report.

    ``contract`` rows: the reference encoded ``expected_samples`` (when given),
    the two native runs agree, the reference stopped at the end of text with a
    non-empty segment list, and the native raw output is a segment list whose
    contents and times are the transcript. ``independent_reference`` rows: the
    case's oracle against the transformers implementation. Without
    ``expected_samples``, every row passes exactly when ``_assert_parity``
    passes (``test_comparison_checks_agree_with_the_assertions``).
    """
    contract, independent = "contract", "independent_reference"
    native_ids, reference_ids = raw["token_ids"], reference["generated_ids"]
    segments = _segments(reference["text"]) or []
    expected_text = " ".join(str(item["Content"]) for item in segments).strip()
    parsed = _segments(raw["text"])
    native_segments = parsed or []
    checks = []
    if expected_samples is not None:
        checks.append(
            _check(
                "reference_speech_samples",
                "Samples the reference encoded at 24 kHz",
                contract,
                int(reference["speech_samples"]),
                "==",
                expected_samples,
            )
        )
    checks += [
        _check(
            "native_runs_differing_tokens",
            "Token IDs that differ between the default and the raw-output native run",
            contract,
            _differences(default["token_ids"], native_ids),
            "==",
            0,
        ),
        _check(
            "reference_stopped_at_eos",
            "The reference run ended at its end-of-text token",
            contract,
            bool(reference["stopped_at_eos"]),
            "==",
            True,
        ),
        _check(
            "reference_segments",
            "Segments in the reference output",
            contract,
            len(segments),
            ">=",
            1,
        ),
        _check(
            "reference_transcript_words",
            "Words in the reference transcript",
            contract,
            len(_words(expected_text)),
            ">=",
            1,
        ),
        _check(
            "native_output_is_segment_list",
            "The native raw output parses as a segment list",
            contract,
            parsed is not None,
            "==",
            True,
        ),
        _check(
            "transcript_matches_raw_output",
            "The native transcript and segments are the raw output's contents and times",
            contract,
            parsed is not None and _transcript_matches(default, parsed),
            "==",
            True,
        ),
    ]
    if case["oracle"] == "tokens":
        first = _first_difference(native_ids, reference_ids)
        if first is None:
            checks += [
                _check(
                    "differing_tokens",
                    "Generated token IDs that differ from the reference",
                    independent,
                    _differences(native_ids, reference_ids),
                    "==",
                    0,
                ),
                _check(
                    "differing_raw_output",
                    "Raw output (segment JSON) unlike the reference's: 1 = differs",
                    independent,
                    int(raw["text"] != reference["text"]),
                    "==",
                    0,
                ),
                _check(
                    "differing_transcript",
                    "Transcript unlike the reference's: 1 = differs",
                    independent,
                    int(default["text"] != expected_text),
                    "==",
                    0,
                ),
            ]
        else:
            pairs = list(zip(native_segments, segments))
            checks += [
                _check(
                    "first_divergence_bf16_values",
                    f"BF16 values from the reference best to the native token at step {first}"
                    " (65536: not a near-tie, outside the reference top 5)",
                    independent,
                    _tie_distance(native_ids, reference, first),
                    "<=",
                    NEAR_TIE_BF16_STEPS,
                ),
                _check(
                    "segment_count_difference",
                    "Segments, native minus reference",
                    independent,
                    len(native_segments) - len(segments),
                    "==",
                    0,
                ),
                _check(
                    "segment_content_mismatches",
                    "Segments whose content or speaker differs from the reference",
                    independent,
                    sum(
                        str(ours["Content"]) != str(theirs["Content"])
                        or ours["Speaker"] != theirs["Speaker"]
                        for ours, theirs in pairs
                    ),
                    "==",
                    0,
                ),
                _check(
                    "max_timestamp_difference_seconds",
                    "Largest start or end time difference to the reference, seconds",
                    independent,
                    _largest_gap(
                        float(ours[key]) - float(theirs[key])
                        for ours, theirs in pairs
                        for key in ("Start", "End")
                    ),
                    "<=",
                    FRAME_SECONDS,
                ),
            ]
    else:
        checks += [
            _check(
                "differing_words",
                "Word positions that differ from the reference transcript, plus the word-count"
                " difference",
                independent,
                _differences(_words(default["text"]), _words(expected_text)),
                "==",
                0,
            ),
            _check(
                "speaker_word_differences",
                "Word positions whose word or speaker differs from the reference, plus the"
                " word-count difference",
                independent,
                _differences(_speaker_words(native_segments), _speaker_words(segments)),
                "==",
                0,
            ),
        ]
    return checks


def _native_record(default: dict, raw: dict) -> dict:
    return {
        "transcript": default.get("text"),
        "raw_output": raw.get("text"),
        "token_ids": raw.get("token_ids"),
        "segments": default.get("segments", []),
    }


def _reference_record(reference: dict) -> dict:
    segments = _segments(str(reference.get("text", ""))) or []
    versions = reference.get("versions") or {}
    return {
        "backend": f"transformers {versions.get('transformers')} vibevoice_asr",
        "weights": f"{REFERENCE_REPOSITORY}@{REFERENCE_REVISION}",
        "source": (TEST_ROOT / "reference-source.json").read_text(encoding="utf-8"),
        "reference_text": " ".join(
            str(item.get("Content", "")) for item in segments if isinstance(item, dict)
        ).strip(),
        "raw_output": reference.get("text"),
        "reference_ids": reference.get("generated_ids"),
        "versions": versions,
    }


def _record_comparison(
    default: dict,
    raw: dict,
    reference: dict,
    case: dict,
    directory: Path,
    expected_samples: int,
) -> None:
    """Record the case's checks as a reference comparison for the evidence report.

    Runs after the assertions, also when one failed, and never raises: a record
    that cannot be made is noted instead, so the test keeps its own result.
    """
    if not evidence_enabled():
        return
    try:
        checks = _comparison_checks(default, raw, reference, case, expected_samples)
        native_file = directory / "native.json"
        reference_file = directory / "reference.json"
        native_file.write_text(
            json.dumps({"token_ids": raw.get("token_ids"), "raw_output": raw.get("text")}),
            encoding="utf-8",
        )
        reference_file.write_text(
            json.dumps(
                {"token_ids": reference.get("generated_ids"), "raw_output": reference.get("text")}
            ),
            encoding="utf-8",
        )
    except Exception as error:
        record_evidence("comparison_not_recorded", f"{type(error).__name__}: {error}")
        return
    record_evidence(
        "reference_comparison",
        {
            "label": f"{case['name']}: native runtime vs the transformers implementation",
            "scope": "independent_reference",
            "enforced": True,
            "native": native_file,
            "reference": reference_file,
            "checks": checks,
        },
    )


def _assert_segments_agree(native: list[dict], reference: list[dict]) -> None:
    assert len(native) == len(reference), f"{len(native)} native vs {len(reference)} segments"
    for ours, theirs in zip(native, reference):
        assert str(ours["Content"]) == str(theirs["Content"]), (ours, theirs)
        assert ours["Speaker"] == theirs["Speaker"], (ours, theirs)
        for key in ("Start", "End"):
            assert abs(float(ours[key]) - float(theirs[key])) <= FRAME_SECONDS, (ours, theirs)


def _assert_transcript_consistent(default: dict, raw_segments: list[dict]) -> None:
    """The default output is the raw output's contents and the times of its timed segments."""
    assert default["text"] == " ".join(str(s["Content"]) for s in raw_segments).strip()
    assert [
        (s["start_seconds"], s["end_seconds"], s["text"]) for s in default.get("segments", [])
    ] == _timed(raw_segments)


def _assert_parity(default: dict, raw: dict, reference: dict, case: dict) -> None:
    assert default["token_ids"] == raw["token_ids"], "the two native runs generated different ids"
    segments = _segments(reference["text"])
    assert reference["stopped_at_eos"], "the reference run hit its token limit"
    assert segments, f"the reference output is not a segment list: {reference['text']!r}"
    expected_text = " ".join(str(item["Content"]) for item in segments).strip()
    assert _words(expected_text), "the reference transcript is empty, so the gate proves nothing"
    native_segments = _segments(raw["text"])
    assert native_segments is not None, f"native raw output is not a list: {raw['text']!r}"
    _assert_transcript_consistent(default, native_segments)
    native_ids, reference_ids = raw["token_ids"], reference["generated_ids"]
    print(
        f"{case['name']}: {len(native_ids)} native / {len(reference_ids)} reference tokens; "
        f"transcript {default['text']!r}"
    )
    if case["oracle"] == "tokens":
        first = _first_difference(native_ids, reference_ids)
        if first is None:
            assert raw["text"] == reference["text"]
            assert default["text"] == expected_text
            return
        gap = _assert_reference_tie(native_ids, reference, first)
        print(f"{case['name']}: the runs part at step {first}, a reference near-tie (gap {gap})")
        _assert_segments_agree(native_segments, segments)
    else:
        assert case["oracle"] == "transcript_words"
        assert _words(default["text"]) == _words(expected_text), (
            f"transcript mismatch: native {default['text']!r} reference {expected_text!r}"
        )
        assert _speaker_words(native_segments) == _speaker_words(segments), (
            "a word has another speaker than in the reference"
        )


def test_near_ties_are_at_most_two_bf16_values_apart() -> None:
    def reference(best: float, value: float) -> dict:
        return {"eos_id": 2, "top_logits": [{"ids": [7, 9], "values": [best, value]}]}

    assert _assert_reference_tie([9], reference(20.125, 20.125), 0) == 0.0
    assert _assert_reference_tie([9], reference(20.125, 20.0), 0) == 0.125
    assert _assert_reference_tie([9], reference(20.125, 19.875), 0) == 0.25
    assert _assert_reference_tie([9], reference(32.0, 31.75), 0) == 0.25
    for best, value in ((20.125, 19.75), (32.0, 31.625), (0.5, -0.5)):
        with pytest.raises(AssertionError):
            _assert_reference_tie([9], reference(best, value), 0)
    with pytest.raises(AssertionError):
        _assert_reference_tie([5], reference(20.125, 20.125), 0)


def test_first_difference_counts_an_early_stop() -> None:
    assert _first_difference([1, 2, 3], [1, 2, 3]) is None
    assert _first_difference([1, 2, 4], [1, 2, 3]) == 2
    assert _first_difference([1, 2], [1, 2, 3]) == 2


_SEGMENT = {"Start": 0.0, "End": 1.5, "Speaker": 0, "Content": "Hello world."}
_OTHER_TEXT = {**_SEGMENT, "Content": "Hello there."}
_TIE = [{"ids": [5], "values": [20.0]}, {"ids": [6, 7], "values": [20.125, 20.0]}]
_FAR = [{"ids": [5], "values": [20.0]}, {"ids": [6, 7], "values": [20.125, 19.75]}]
_PARITY_SCENARIOS = {
    "identical": ({}, True),
    "near_tie": ({"native_ids": (5, 7, 2), "top": _TIE}, True),
    "tie_three_values_apart": ({"native_ids": (5, 7, 2), "top": _FAR}, False),
    "outside_top_5": ({"native_ids": (5, 8, 2), "top": _TIE}, False),
    "segments_differ_after_tie": (
        {"native_ids": (5, 7, 2), "top": _TIE, "native_segments": (_OTHER_TEXT,)},
        False,
    ),
    "late_end_after_tie": (
        {"native_ids": (5, 7, 2), "top": _TIE, "native_segments": ({**_SEGMENT, "End": 1.7},)},
        False,
    ),
    "same_ids_other_raw_output": ({"native_segments": (_OTHER_TEXT,)}, False),
    "reference_hit_token_limit": ({"stopped": False}, False),
    "empty_reference_transcript": (
        {
            "native_segments": ({**_SEGMENT, "Content": ""},),
            "reference_segments": ({**_SEGMENT, "Content": ""},),
        },
        False,
    ),
    "native_output_not_json": ({"raw_text": "not json"}, False),
    # Like the runtime, the oracle strips "assistant" only as a role header.
    "role_header_with_space": (
        {"oracle": "transcript_words", "raw_text": "assistant " + json.dumps([_SEGMENT])},
        True,
    ),
    "role_header_without_whitespace": (
        {"oracle": "transcript_words", "raw_text": "assistant" + json.dumps([_SEGMENT])},
        False,
    ),
    "transcript_not_raw_contents": ({"transcript": "Hello."}, False),
    "native_runs_differ": ({"default_ids": (5, 9, 2)}, False),
    "words_equal": ({"oracle": "transcript_words", "reference_ids": (5, 9, 9, 2)}, True),
    "words_differ": ({"oracle": "transcript_words", "native_segments": (_OTHER_TEXT,)}, False),
    "timestamp_as_text": (
        {"oracle": "transcript_words", "raw_text": json.dumps([{**_SEGMENT, "Start": "0.0"}])},
        False,
    ),
    "speakers_swapped": (
        {
            "oracle": "transcript_words",
            "reference_segments": (
                {**_SEGMENT, "Speaker": 0, "Content": "Hello"},
                {**_SEGMENT, "Speaker": 1, "Content": "Goodbye"},
            ),
            "native_segments": (
                {**_SEGMENT, "Speaker": 1, "Content": "Hello"},
                {**_SEGMENT, "Speaker": 0, "Content": "Goodbye"},
            ),
        },
        False,
    ),
    "speakers_differ": (
        {"oracle": "transcript_words", "native_segments": ({**_SEGMENT, "Speaker": 1},)},
        False,
    ),
}


def _parity_inputs(
    native_ids=(5, 6, 2),
    reference_ids=(5, 6, 2),
    native_segments=(_SEGMENT,),
    reference_segments=(_SEGMENT,),
    oracle="tokens",
    top=(),
    stopped=True,
    raw_text=None,
    transcript=None,
    default_ids=None,
) -> tuple[dict, dict, dict, dict]:
    """Native and reference results shaped like the runs' JSON, for the oracle tests."""
    native_segments, reference_segments = list(native_segments), list(reference_segments)
    raw = {
        "text": json.dumps(native_segments) if raw_text is None else raw_text,
        "token_ids": list(native_ids),
    }
    default = {
        "text": " ".join(s["Content"] for s in native_segments)
        if transcript is None
        else transcript,
        "token_ids": list(native_ids if default_ids is None else default_ids),
        "segments": [
            {"start_seconds": s["Start"], "end_seconds": s["End"], "text": s["Content"]}
            for s in native_segments
        ],
    }
    reference = {
        "text": json.dumps(reference_segments),
        "generated_ids": list(reference_ids),
        "top_logits": list(top),
        "stopped_at_eos": stopped,
        "eos_id": 2,
    }
    return default, raw, reference, {"name": "case", "oracle": oracle}


@pytest.mark.parametrize("scenario", sorted(_PARITY_SCENARIOS))
def test_comparison_checks_agree_with_the_assertions(scenario: str) -> None:
    changes, expected = _PARITY_SCENARIOS[scenario]
    default, raw, reference, case = _parity_inputs(**changes)
    try:
        _assert_parity(default, raw, reference, case)
        asserted = True
    except AssertionError:
        asserted = False
    checks = _comparison_checks(default, raw, reference, case)
    assert asserted is expected
    assert all(check["passed"] for check in checks) is expected


def test_a_timestamp_that_is_not_a_number_fails_its_check() -> None:
    # After a near-tie the segments are compared; a NaN time must fail the largest-gap row.
    nan_end = ({**_SEGMENT, "End": math.nan},)
    inputs = _parity_inputs(native_ids=(5, 7, 2), top=_TIE, native_segments=nan_end)
    rows = {row["name"]: row for row in _comparison_checks(*inputs)}
    assert rows["max_timestamp_difference_seconds"]["passed"] is False


@pytest.fixture(scope="session")
def bundle_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("vibevoice-asr-bundles")


def test_e2e(tmp_path: Path, bundle_root: Path, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    with evidence_stage("build"):
        bundle = _bundle(manifest, model_dir, bundle_root)
    audio, frames = _case_audio(case, tmp_path)
    rate = int(case["input_sample_rate"])
    record_evidence(
        "inputs",
        {
            "manifest": manifest,
            "case": case,
            "audio": audio,
            "context_info": str(case.get("context_info") or ""),
            "sample_rate": rate,
            "samples": frames,
        },
    )
    record_evidence(
        "thresholds",
        {"near_tie_bf16_values": NEAR_TIE_BF16_STEPS, "timestamp_seconds": FRAME_SECONDS},
    )
    with evidence_stage("native"):
        default = _native(binary, runtime_root, bundle, case, audio, raw=False)
        raw = _native(binary, runtime_root, bundle, case, audio, raw=True)
    record_evidence("native", _native_record(default, raw))
    with evidence_stage("reference"):
        reference = _reference(case, audio, tmp_path)
    record_evidence("reference", _reference_record(reference))
    print(f"{case['name']}: reference {reference['versions']}")
    # Resampling to 24 kHz gives ceil(n * 24000 / rate) samples on both sides.
    expected_samples = -(-frames * SAMPLE_RATE // rate)
    with evidence_stage("compare"):
        try:
            assert reference["speech_samples"] == expected_samples
            _assert_parity(default, raw, reference, case)
        finally:
            _record_comparison(default, raw, reference, case, tmp_path, expected_samples)
