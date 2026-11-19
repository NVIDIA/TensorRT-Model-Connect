# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native CLI and official-reference E2E for moss_transcribe_diarize.

The reference is the checkpoint's own processor and model code (Transformers
remote code, loaded only from the manifest's pinned snapshot in the Hugging
Face cache), run greedily as the official inference utilities run it, in FP32
with TF32 disabled.

Oracles, chosen per testcase:

- ``token_exact``: the native generated ids equal the reference's.
- ``teacher_forced``: the FP32 reference scores the native transcript in one
  teacher-forced pass. Every native token must be the reference argmax or trail
  it by at most ``max_logit_gap_bf16_ulps`` units of the BF16 spacing at that
  step's largest logit: candidates that close are not ordered reliably once the
  logits are BF16, and the official BF16 pipeline itself picks such runners-up.
  The transcript must also end where the reference would end it: the reference
  argmax is EOS at a native EOS and at no earlier step.

Both oracles also check that the native text is the official tokenizer's
decoding of the native ids, that the segments are the transcript's
``[start][Sxx]text[end]`` spans as the official transcript parser finds them
(a bracketed time closes a segment only before the next ``[`` or the end of
the transcript), and that direct C and C++ SDK calls return
the CLI's result. Cases with ``input_contract`` also check that two different
channels are averaged, the default token limit and the refused requests.

A case names one clip as ``test_input_audio``, or clips that are joined in
order with 0.8 s of silence as ``input_audio_parts`` (an input longer than one
committed clip should be).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from tools.e2e_evidence import record_evidence

FAMILY = "moss_transcribe_diarize"
TASKS = frozenset({"speech_transcription"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
ORACLES = frozenset({"token_exact", "teacher_forced"})
# The official inference utilities' default instruction.
PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
STAMP_CHARS = frozenset("0123456789.")
SPEAKER_CHARS = frozenset("S0123456789")
SAMPLE_RATE = 16000
# The silence the clips already have between their utterances.
JOIN_SILENCE_SECONDS = 0.8


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            assert case["oracle"] in ORACLES
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
        parameters = names if enabled else [
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
    # Only the pinned snapshot: the reference runs the checkpoint's own Python
    # files, so no other directory may stand in for it.
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True))
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the cached checkpoint {manifest['hf_id']}"
        ) from error


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root


def _asset(raw: str) -> Path:
    path = TEST_ROOT / raw
    assert path.is_file(), f"{FAMILY} E2E asset does not exist: {path}"
    return path


def _read_pcm16_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
        assert stream.getframerate() == SAMPLE_RATE
        frames = stream.readframes(stream.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _input_clips(case: dict) -> list[str]:
    """One clip (``test_input_audio``) or clips to join in order (``input_audio_parts``)."""
    single, parts = case.get("test_input_audio"), case.get("input_audio_parts")
    assert (single is None) != (parts is None), (
        f"{case['name']} needs exactly one of test_input_audio and input_audio_parts")
    if single is not None:
        assert isinstance(single, str) and single, f"{case['name']}: test_input_audio"
        return [single]
    assert isinstance(parts, list) and len(parts) > 1, f"{case['name']}: input_audio_parts"
    assert all(isinstance(part, str) and part for part in parts), f"{case['name']}: input_audio_parts"
    return parts


def _input_samples(case: dict) -> np.ndarray:
    """The case's clips in order, joined with the clips' own silence between utterances."""
    parts = [_read_pcm16_wav(_asset(raw)) for raw in _input_clips(case)]
    silence = np.zeros(round(JOIN_SILENCE_SECONDS * SAMPLE_RATE), dtype=np.float32)
    joined = [parts[0]]
    for part in parts[1:]:
        joined += [silence, part]
    return np.concatenate(joined)


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(BuildRequest(
        model_dir=model_dir,
        output_path=bundle,
        family=FAMILY,
        task=manifest["task"],
        precision=manifest["precision"],
        max_sequence_length=int(manifest["max_sequence_length"]),
        tensor_parallel_size=int(manifest["tensor_parallel_size"]),
    ))


def _cli(binary: Path, runtime_root: Path, bundle: Path, audio: Path, *options: str):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    return subprocess.run(
        [str(binary), "transcribe", str(bundle), "--runtime-root", str(runtime_root),
         "--input", str(audio), *options],
        check=False, capture_output=True, text=True, env=env, timeout=1800,
    )


def _json_line(completed: subprocess.CompletedProcess) -> dict:
    assert completed.returncode == 0, completed.stderr[-2000:]
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            return json.loads(line[start:])
    raise AssertionError(f"native transcription returned no JSON: {completed.stdout[-1000:]}")


def _sdk_consumer(name: str) -> Path:
    """A direct SDK consumer of the native build, built by name when missing."""
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    target = f"test_{FAMILY}_sdk_{name}"
    subprocess.run(["cmake", "--build", str(native), "--parallel", "8", "--target", target],
                   check=True, capture_output=True, text=True, timeout=1800)
    return _required_path(str(native / target), target)


def _assert_sdk_results(native: dict, runtime_root: Path, bundle: Path, case: dict,
                        tmp_path: Path) -> None:
    """The C and C++ SDK return the CLI's text, ids and segments for the same request."""
    pcm = tmp_path / "input.f32"
    _input_samples(case).tofile(pcm)
    expected = {
        "text": native["text"],
        "token_ids": native["token_ids"],
        "segments": [{key: segment[key] for key in ("start_seconds", "end_seconds", "text")}
                     for segment in native["segments"]],
    }
    for language in ("c", "cpp"):
        completed = subprocess.run(
            [str(_sdk_consumer(language)), str(bundle), str(runtime_root), str(pcm),
             str(int(case["max_new_tokens"]))],
            check=False, capture_output=True, text=True, timeout=1800,
        )
        assert _json_line(completed) == expected, f"{language} SDK result differs from the CLI"


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
    pcm = np.round(np.clip(samples, -1.0, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())
    return path


def _assert_input_contract(binary: Path, runtime_root: Path, bundle: Path, case: dict,
                           audio_path: Path, tokens: list[int], tmp_path: Path) -> None:
    """Channel averaging, the default token limit and the refused requests."""
    audio = _input_samples(case)
    # The channels are the clip plus and minus its time-reversed copy, limited
    # per sample to the PCM16 headroom: either channel alone sounds different,
    # their average is the clip itself.
    mono = np.round(audio * 32768).astype(np.int32)
    headroom = np.maximum(32767 - np.abs(mono), 0)
    other = np.clip(mono[::-1], -headroom, headroom)
    stereo = tmp_path / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(np.stack([mono + other, mono - other], axis=1).astype("<i2").tobytes())
    limit = f"max_new_tokens={int(case['max_new_tokens'])}"
    assert _json_line(_cli(binary, runtime_root, bundle, stereo, "--set", limit))[
        "token_ids"] == tokens
    unlimited = _json_line(_cli(binary, runtime_root, bundle, audio_path))
    assert unlimited["token_ids"] == tokens
    refused = [
        (_write_wav(tmp_path / "rate.wav", audio, 8000), (), "16000 Hz"),
        (audio_path, ("--source-language", "en"), "source_language"),
        (audio_path, ("--set", "max_new_tokens=1000000"), "exceeds"),
    ]
    for path, options, message in refused:
        completed = _cli(binary, runtime_root, bundle, path, *options)
        assert completed.returncode != 0 and message in completed.stderr, completed.stderr[-500:]


@contextmanager
def _fp32_reference_precision():
    """TF32 off and the highest FP32 matmul precision while the reference runs; the
    process-wide settings are restored afterwards, also when the reference fails."""
    import torch

    saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cuda.matmul.allow_tf32 = saved[1]
        torch.backends.cudnn.allow_tf32 = saved[2]


@pytest.fixture
def fp32_reference():
    with _fp32_reference_precision():
        yield


class _Reference:
    """The checkpoint's own processor and model, run as the official utilities run them."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True, dtype="auto",
            attn_implementation="sdpa",
        ).to(dtype=torch.float32).to("cuda").eval()

    def inputs(self, audio: np.ndarray):
        messages = [{"role": "user", "content": [{"type": "audio", "audio": "input"},
                                                 {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self.processor(
            text=text, audio=[audio], audio_kwargs={"device": "cuda"}, return_tensors="pt"
        ).to("cuda")

    def generate(self, inputs, max_new_tokens: int) -> list[int]:
        config = self.model.generation_config
        config.max_new_tokens = max_new_tokens
        config.do_sample = False
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                generation_config=config,
            )
        return output[0, inputs["input_ids"].shape[1]:].tolist()

    def teacher_forced_gaps(
        self, inputs, tokens: list[int]
    ) -> tuple[list[float], list[float], list[int]]:
        """Per step: reference max logit minus the given token's logit, the max logit
        and the argmax."""
        torch = self.torch
        prompt = inputs["input_ids"]
        forced = torch.tensor([tokens[:-1]], dtype=prompt.dtype, device=prompt.device)
        ids = torch.cat([prompt, forced], dim=1)
        with torch.inference_mode():
            logits = self.model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
            ).logits[0, prompt.shape[1] - 1:].float()
        chosen = logits[torch.arange(len(tokens)), torch.tensor(tokens, device=logits.device)]
        peaks, argmax = logits.max(dim=-1)
        return (peaks - chosen).tolist(), peaks.tolist(), argmax.tolist()

    def decode(self, tokens: list[int]) -> str:
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _bf16_spacing(value: float) -> float:
    """Distance between adjacent BF16 numbers around ``value`` (8 significant bits)."""
    return math.ldexp(1.0, math.frexp(abs(value))[1] - 8) if value else 0.0


def _stamp(token: str) -> float | None:
    # The token holds only STAMP_CHARS: a time needs a digit and at most one dot.
    return float(token) if token.count(".") <= 1 and token.strip(".") else None


def _transcript_segments(text: str) -> list[tuple[float, float, str]]:
    """``(start, end, "[Sxx]text")`` per segment, as the official transcript
    parser (stripped text, empty segments skipped) reads ``[start][Sxx]text[end]``.

    Spaces may precede ``[Sxx]``. In the text, ``[end]`` at or after the start
    closes the segment only when the next character that is not a space is
    ``[`` or the transcript ends; every other bracket stays text. Times hold at
    most 32 characters and speaker labels at most 16.
    """
    segments: list[tuple[float, float, str]] = []
    state, token, body, pending, speaker, end_token = "seek", "", "", "", "", ""
    start = end = None

    def emit() -> None:
        if start is not None and end is not None and speaker and body.strip():
            segments.append((start, end, f"[{speaker}]{body.strip()}"))

    for char in text:
        if state == "seek":
            if char == "[":
                state, token = "start", ""
        elif state == "start":
            if char == "]":
                start, token = _stamp(token), ""
                state = "speaker_open" if start is not None else "seek"
            elif char in STAMP_CHARS and len(token) < 32:
                token += char
            else:
                state, token = ("start" if char == "[" else "seek"), ""
        elif state == "speaker_open":
            if char == "[":
                state, token = "speaker", ""
            elif not char.isspace():
                state = "seek"
        elif state == "speaker":
            if char == "]":
                valid = len(token) >= 2 and token[0] == "S" and token[1:].isdigit()
                state, speaker, body = ("text", token, "") if valid else ("seek", "", "")
            elif char in SPEAKER_CHARS and len(token) < 16:
                token += char
            else:
                state, token = ("start" if char == "[" else "seek"), ""
        elif state == "text":
            if char == "[":
                state, token = "end", ""
            else:
                body += char
        elif state == "end":
            if char == "]":
                value = _stamp(token)
                if value is not None and value >= start:
                    end, end_token, pending, state = value, token, "", "after"
                else:
                    body, state = body + f"[{token}]", "text"
                token = ""
            elif char in STAMP_CHARS and len(token) < 32:
                token += char
            else:
                # Like the official parser, a 33rd time character is kept twice.
                body += "[" + token + (char * 2 if char in STAMP_CHARS else char)
                state, token = "text", ""
        else:  # after an end
            if char == "[":
                emit()
                state, token = "start", ""
            elif char.isspace():
                pending += char
            else:
                body += f"[{end_token}]{pending}{char}"
                state, pending, end = "text", "", None
    if state == "after":
        emit()
    return segments


def _assert_segments(native: dict) -> None:
    actual = [(s["start_seconds"], s["end_seconds"], s["text"]) for s in native["segments"]]
    assert actual == _transcript_segments(native["text"])
    assert actual, "the transcript has no [start][Sxx]text[end] segment"


def _check(name: str, label: str, scope: str, actual: int | float, operator: str,
           expected: int | float) -> dict:
    """One evidence check row (``tools.e2e_evidence``); its verdict follows from its values."""
    holds = {"==": actual == expected, "<=": actual <= expected, ">=": actual >= expected}
    return {"name": name, "label": label, "scope": scope, "actual": actual,
            "operator": operator, "expected": expected, "passed": holds[operator]}


def _termination_checks(tokens: list[int], argmax: list[int], eos: set[int],
                        max_new_tokens: int) -> list[dict]:
    """How the native ids end, against the teacher-forced reference argmax at every step."""
    return [
        _check("native_eos_before_end", "Native EOS IDs before the last position", "contract",
               sum(1 for token in tokens[:-1] if token in eos), "==", 0),
        _check("stops_at_eos_or_limit", "Native IDs end with EOS or at max_new_tokens (1 = yes)",
               "contract", int(tokens[-1] in eos or len(tokens) == max_new_tokens), "==", 1),
        _check("reference_eos_before_native_end",
               "Steps before the native end at which the FP32 argmax is EOS",
               "independent_reference", sum(1 for token in argmax[:-1] if token in eos), "==", 0),
        _check("native_eos_without_reference_eos",
               "Native EOS at a step whose FP32 argmax is not EOS (1 = yes)",
               "independent_reference", int(tokens[-1] in eos and argmax[-1] not in eos), "==", 0),
    ]


def _assert_termination(tokens: list[int], argmax: list[int], eos: set[int],
                        max_new_tokens: int) -> None:
    """The native ids end where the teacher-forced reference would end them.

    ``argmax`` is the reference argmax at every native step. The ids stop at an
    EOS that is also the reference argmax there, or at the token limit; the
    reference argmax is no EOS at any earlier step. The evidence record uses the
    same ``_termination_checks``.
    """
    assert len(argmax) == len(tokens)
    failed = [check["label"] for check in _termination_checks(tokens, argmax, eos, max_new_tokens)
              if not check["passed"]]
    assert not failed, f"termination checks failed: {failed}"


def _steps_beyond_tolerance(gaps: list[float], peaks: list[float], ulps: float) -> list[int]:
    """Steps whose native token trails the FP32 argmax by more than ``ulps`` BF16
    spacings at that step's largest logit; a step with a non-finite score never passes."""
    return [step for step, (gap, peak) in enumerate(zip(gaps, peaks))
            if not (math.isfinite(gap) and math.isfinite(peak)
                    and gap <= ulps * _bf16_spacing(peak))]


def _ids_file(path: Path, ids: list[int]) -> Path:
    np.save(path, np.asarray(ids, dtype=np.int64))
    return path


def _token_exact_checks(native_ids: list[int], reference_ids: list[int]) -> list[dict]:
    """The ``token_exact`` comparison as an evidence record (``tools.e2e_evidence``)."""
    mismatches = sum(native != reference for native, reference in zip(native_ids, reference_ids))
    mismatches += abs(len(native_ids) - len(reference_ids))
    return [{
        "name": "token_mismatches",
        "label": "Native token IDs that differ from the FP32 reference's (with the length difference)",
        "scope": "independent_reference", "actual": mismatches, "operator": "==", "expected": 0,
        "passed": mismatches == 0,
    }]


def _teacher_forced_checks(tokens: list[int], gaps: list[float], peaks: list[float],
                           argmax: list[int], eos: set[int], max_new_tokens: int,
                           ulps: float) -> list[dict]:
    """The ``teacher_forced`` oracle's checks as an evidence record, from the same values
    as its assertions."""
    widest = max((gap / _bf16_spacing(peak) for gap, peak in zip(gaps, peaks)
                  if math.isfinite(gap) and math.isfinite(peak) and _bf16_spacing(peak) > 0),
                 default=0.0)
    beyond = _steps_beyond_tolerance(gaps, peaks, ulps)
    return [_check("steps_beyond_bf16_tolerance",
                   f"Steps whose native token trails the FP32 argmax by more than {ulps:g} BF16 "
                   f"spacings, or with a non-finite score (largest finite gap: {widest:.4g})",
                   "independent_reference", len(beyond), "==", 0),
            *_termination_checks(tokens, argmax, eos, max_new_tokens)]


@pytest.mark.usefixtures("fp32_reference")
def test_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    audio_path = _write_wav(tmp_path / "input.wav", _input_samples(case), SAMPLE_RATE)
    max_new_tokens = int(case["max_new_tokens"])
    record_evidence("inputs", {"prompt": PROMPT, "audio": audio_path, "oracle": case["oracle"],
                               "max_new_tokens": max_new_tokens})
    native = _json_line(_cli(binary, runtime_root, bundle, audio_path,
                             "--set", f"max_new_tokens={max_new_tokens}"))
    tokens = [int(token) for token in native["token_ids"]]
    record_evidence("native", {"transcript": native["text"], "token_ids": tokens,
                               "segments": native["segments"]})
    assert 0 < len(tokens) <= max_new_tokens
    _assert_sdk_results(native, runtime_root, bundle, case, tmp_path)
    if case.get("input_contract"):
        _assert_input_contract(binary, runtime_root, bundle, case, audio_path, tokens, tmp_path)

    reference = _Reference(model_dir)
    inputs = reference.inputs(_input_samples(case))
    reference_ids = reference.generate(inputs, max_new_tokens)
    record_evidence("reference", {"reference_text": reference.decode(reference_ids),
                                  "reference_ids": reference_ids})
    assert native["text"] == reference.decode(tokens)
    _assert_segments(native)
    native_file = _ids_file(tmp_path / "native_ids.npy", tokens)
    if case["oracle"] == "token_exact":
        record_evidence("reference_comparison", {
            "label": "FP32 reference token IDs", "scope": "independent_reference", "enforced": True,
            "native": native_file,
            "reference": _ids_file(tmp_path / "reference_ids.npy", reference_ids),
            "checks": _token_exact_checks(tokens, reference_ids),
        })
        assert tokens == reference_ids
        return
    eos = set(reference.model.generation_config.eos_token_id
              if isinstance(reference.model.generation_config.eos_token_id, list)
              else [reference.model.generation_config.eos_token_id])
    ulps = float(_thresholds(case_name)["max_logit_gap_bf16_ulps"])
    record_evidence("thresholds", {"max_logit_gap_bf16_ulps": ulps})
    gaps, peaks, argmax = reference.teacher_forced_gaps(inputs, tokens)
    record_evidence("reference_comparison", {
        "label": "FP32 teacher-forced scores of the native tokens", "scope": "independent_reference",
        "enforced": True, "native": native_file,
        "reference": _ids_file(tmp_path / "reference_argmax_ids.npy", argmax),
        "checks": _teacher_forced_checks(tokens, gaps, peaks, argmax, eos, max_new_tokens, ulps),
    })
    _assert_termination(tokens, argmax, eos, max_new_tokens)
    beyond = _steps_beyond_tolerance(gaps, peaks, ulps)
    assert not beyond, "native tokens trail the reference argmax beyond the tolerance: " + ", ".join(
        f"step {step} ({tokens[step]}) by {gaps[step]:.4f} at largest logit {peaks[step]:.4f}"
        for step in beyond[:5])
