# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for VibeVoice-ASR.

One bundle per manifest serves all of its cases. Each case reads one PCM16 file
(the committed clip, or a 24 kHz and repeated version written at run time), runs
the native ``trtmc transcribe`` on it twice (the default transcript and
``raw_output``, which keeps the speaker ids) and the official model once
(``official_asr.py`` under the reference interpreter of ``prepare_reference.py``,
with the sigma-VAE mean instead of its sampling noise). Both native runs must
generate the same ids, and the default transcript must be the raw output's
contents and timestamps. Oracles:

* ``tokens``: used wherever both sides read the same 24 kHz samples. Either the
  ids equal the official ones, with identical raw output and transcript; or the
  runs part at a near-tie of the official model: at the first differing step the
  native token's official logit is at most ``NEAR_TIE_BF16_STEPS`` BF16 values
  below the official best (the official logits are BF16). Ids after that step
  are not comparable position by position, so the segments must then agree:
  same count, identical text and speaker, and timestamps within one acoustic
  frame (3200 samples at 24 kHz).
* ``transcript_words``: the transcripts have the same words and speakers. Used
  for 16 kHz input, which the official processor resamples with librosa (soxr)
  and the runtime with its own windowed-sinc filter, so the samples differ.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from families.vibevoice_asr.model import TOKENIZER_FILES, TOKENIZER_REPOSITORY, TOKENIZER_REVISION
from tensorrt_model_connect import BuildRequest, build

FAMILY = "vibevoice_asr"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SILENCE_SECONDS = 1.0
FRAME_SECONDS = 3200 / 24000
# Teacher-forcing the official model on 20,000 tokens generated natively (a one-hour
# input, positions 27,052-47,052) gave 48 differing argmaxes: 30 equal official logits,
# 16 one BF16 value apart and 2 two values apart (logits 19.75-34.5). The native
# runtime's logit noise therefore reaches two BF16 values, and a first divergence
# within that distance is a near-tie, not a defect.
NEAR_TIE_BF16_STEPS = 2
SAMPLE_RATE = 24000


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        # The build downloads this tokenizer; CI must stage exactly this pin.
        assert manifest["hf_dependencies"] == [
            {"repo_id": TOKENIZER_REPOSITORY, "revision": TOKENIZER_REVISION}
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
    """PCM16, which the native reader and the official reader (ffmpeg or soundfile)
    both decode as ``value / 32768``, so the two sides see identical samples."""
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
    """The PCM16 file both sides read and its frame count."""
    source = TEST_ROOT / str(case["audio"])
    assert source.is_file(), f"selected {FAMILY} E2E audio does not exist: {source}"
    samples, rate = _read_pcm16(source)
    target = int(case["input_sample_rate"])
    repeat = int(case.get("repeat", 1))
    if repeat == 1 and target == rate:
        return source, samples.size
    if target != rate:
        samples = _resample(samples, rate, target)
    silence = np.zeros(int(SILENCE_SECONDS * target), dtype=np.float32)
    pieces = [samples]
    for _ in range(repeat - 1):
        pieces.extend((silence, samples))
    audio = np.concatenate(pieces)
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


def _official(model_dir: Path, case: dict, audio: Path, directory: Path) -> dict:
    python = _required_path(
        os.environ.get("TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON"),
        "TRTMC_VIBEVOICE_ASR_REFERENCE_PYTHON",
    )
    source = _required_path(
        os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"), "TRTMC_REFERENCE_SOURCE_DIR"
    )
    pin = json.loads((TEST_ROOT / "reference-source.json").read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert head == pin["revision"], f"official source is at {head}, not {pin['revision']}"
    tokenizer = _snapshot(TOKENIZER_REPOSITORY, TOKENIZER_REVISION, list(TOKENIZER_FILES))
    output = directory / f"{case['name']}.official.json"
    command = [
        str(python),
        "-I",
        str(TEST_ROOT / "official_asr.py"),
        "--source-dir",
        str(source),
        "--model-dir",
        str(model_dir),
        "--tokenizer-dir",
        str(tokenizer),
        "--audio",
        str(audio),
        "--acoustic",
        "mean",
        "--output",
        str(output),
    ]
    if case.get("context_info"):
        command += ["--context-info", str(case["context_info"])]
    subprocess.run(command, check=True, timeout=3600)
    return json.loads(output.read_text(encoding="utf-8"))


def _segments(raw: str) -> list[dict] | None:
    text = raw.strip()
    if text.startswith("assistant"):
        text = text[len("assistant") :].strip()
    try:
        segments = json.loads(text)
    except json.JSONDecodeError:
        return None
    return segments if isinstance(segments, list) else None


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _bf16_bits(value: float) -> int:
    """The BF16 bit pattern of a logit the official model produced in BF16."""
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    assert bits & 0xFFFF == 0, f"{value} is not a BF16 value"
    return bits >> 16


def _first_difference(native: list[int], official: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(native, official)):
        if left != right:
            return index
    return None if len(native) == len(official) else min(len(native), len(official))


def _assert_reference_tie(native_ids: list[int], official: dict, step: int) -> float:
    """At ``step`` the native token's official logit is a near-tie with the official best."""
    choice = native_ids[step] if step < len(native_ids) else official["eos_id"]
    top = official["top_logits"][step]
    assert choice in top["ids"], f"step {step}: native token {choice} is not in the official top 5"
    best = top["values"][0]
    value = top["values"][top["ids"].index(choice)]
    # With equal signs, consecutive BF16 values have consecutive bit patterns.
    assert (best >= 0) == (value >= 0), f"step {step}: logits {best} and {value} differ in sign"
    distance = abs(_bf16_bits(best) - _bf16_bits(value))
    assert distance <= NEAR_TIE_BF16_STEPS, (
        f"step {step}: native token {choice} is {distance} BF16 values behind"
    )
    return best - value


def _assert_segments_agree(native: list[dict], official: list[dict]) -> None:
    assert len(native) == len(official), f"{len(native)} native vs {len(official)} segments"
    for ours, theirs in zip(native, official):
        assert str(ours["Content"]) == str(theirs["Content"]), (ours, theirs)
        assert ours["Speaker"] == theirs["Speaker"], (ours, theirs)
        for key in ("Start", "End"):
            assert abs(float(ours[key]) - float(theirs[key])) <= FRAME_SECONDS, (ours, theirs)


def _assert_transcript_consistent(default: dict, raw_segments: list[dict]) -> None:
    """The default output is the raw output's contents and timestamps."""
    assert default["text"] == " ".join(str(s["Content"]) for s in raw_segments).strip()
    assert [
        (s["start_seconds"], s["end_seconds"], s["text"]) for s in default.get("segments", [])
    ] == [(float(s["Start"]), float(s["End"]), str(s["Content"])) for s in raw_segments]


def _assert_parity(default: dict, raw: dict, official: dict, case: dict) -> None:
    assert default["token_ids"] == raw["token_ids"], "the two native runs generated different ids"
    segments = _segments(official["text"])
    assert official["stopped_at_eos"], "the official run hit its token limit"
    assert segments, f"the official output is not a segment list: {official['text']!r}"
    expected_text = " ".join(str(item["Content"]) for item in segments).strip()
    assert _words(expected_text), "the official transcript is empty, so the gate proves nothing"
    native_segments = _segments(raw["text"])
    assert native_segments is not None, f"native raw output is not a list: {raw['text']!r}"
    _assert_transcript_consistent(default, native_segments)
    native_ids, official_ids = raw["token_ids"], official["generated_ids"]
    print(
        f"{case['name']}: {len(native_ids)} native / {len(official_ids)} official tokens; "
        f"transcript {default['text']!r}"
    )
    if case["oracle"] == "tokens":
        first = _first_difference(native_ids, official_ids)
        if first is None:
            assert raw["text"] == official["text"]
            assert default["text"] == expected_text
            return
        gap = _assert_reference_tie(native_ids, official, first)
        print(f"{case['name']}: the runs part at step {first}, an official near-tie (gap {gap})")
        _assert_segments_agree(native_segments, segments)
    else:
        assert case["oracle"] == "transcript_words"
        assert _words(default["text"]) == _words(expected_text), (
            f"transcript mismatch: native {default['text']!r} official {expected_text!r}"
        )
        assert {s["Speaker"] for s in native_segments} == {s["Speaker"] for s in segments}


def test_near_ties_are_at_most_two_bf16_values_apart() -> None:
    def official(best: float, value: float) -> dict:
        return {"eos_id": 2, "top_logits": [{"ids": [7, 9], "values": [best, value]}]}

    assert _assert_reference_tie([9], official(20.125, 20.125), 0) == 0.0
    assert _assert_reference_tie([9], official(20.125, 20.0), 0) == 0.125
    assert _assert_reference_tie([9], official(20.125, 19.875), 0) == 0.25
    assert _assert_reference_tie([9], official(32.0, 31.75), 0) == 0.25
    for best, value in ((20.125, 19.75), (32.0, 31.625), (0.5, -0.5)):
        with pytest.raises(AssertionError):
            _assert_reference_tie([9], official(best, value), 0)
    with pytest.raises(AssertionError):
        _assert_reference_tie([5], official(20.125, 20.125), 0)


def test_first_difference_counts_an_early_stop() -> None:
    assert _first_difference([1, 2, 3], [1, 2, 3]) is None
    assert _first_difference([1, 2, 4], [1, 2, 3]) == 2
    assert _first_difference([1, 2], [1, 2, 3]) == 2


@pytest.fixture(scope="session")
def bundle_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("vibevoice-asr-bundles")


def test_e2e(tmp_path: Path, bundle_root: Path, case_name: str) -> None:
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    bundle = _bundle(manifest, model_dir, bundle_root)
    audio, frames = _case_audio(case, tmp_path)
    default = _native(binary, runtime_root, bundle, case, audio, raw=False)
    raw = _native(binary, runtime_root, bundle, case, audio, raw=True)
    official = _official(model_dir, case, audio, tmp_path)
    rate = int(case["input_sample_rate"])
    # The official processor resamples to 24 kHz to ceil(n * 24000 / rate) samples.
    assert official["speech_samples"] == -(-frames * SAMPLE_RATE // rate)
    _assert_parity(default, raw, official, case)
