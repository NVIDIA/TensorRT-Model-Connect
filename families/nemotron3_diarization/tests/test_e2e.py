# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, public-SDK and official-reference E2E for Nemotron 3 Diarization."""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from fnmatch import fnmatch
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import record_evidence

FAMILY = "nemotron3_diarization"
TASKS = frozenset({"audio_to_timed_labels"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
SAMPLE_RATE = 16000
# Two people from two public recordings: a LibriSpeech test-clean reader and
# the project maintainer. Both are 16 kHz mono PCM16.
FIRST_SPEAKER = "data/librispeech-test-clean-6930-75918-0003.wav"
SECOND_SPEAKER = "data/maintainer-recording-16k-mono.wav"
# The one case of a testcase-only selection that names none of this family's
# cases: another collected family may own it, or it is a typo.
UNMATCHED = "unmatched-testcase-selection"


def _manifest_value(container: dict, key: str, kinds: tuple[type, ...]):
    """A manifest value of its own JSON type, never coerced (a boolean is not a number)."""
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, kinds):
        names = " or ".join(kind.__name__ for kind in kinds)
        raise TypeError(f"{FAMILY} manifest {key} must be a JSON {names}, not {value!r}")
    return value


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] in TASKS
        for case in manifest["testcases"]:
            name = _manifest_value(case, "name", (str,))
            assert name not in result
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _filters(config) -> tuple[set[str], set[str]]:
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
    return model_filters, testcase_filters


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters, testcase_filters = _filters(config)
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
    named = FAMILY in model_filters or any(
        name in model_filters or manifest["name"] in model_filters
        for name, (_, manifest, _) in CASES.items()
    )
    if named and not selected:
        # Pytest skips an empty parametrization, so a selection that names this
        # family but matches none of its cases would pass without testing.
        raise pytest.UsageError(
            f"the E2E selection names {FAMILY}, but --e2e-testcase "
            f"{', '.join(sorted(testcase_filters))} matches none of its cases: "
            f"{', '.join(sorted(CASES))}"
        )
    return sorted(selected), True


def _resolve_unmatched(request) -> None:
    """Skip when another collected family's E2E owns the selected testcase; fail otherwise."""
    _, wanted = _filters(request.config)
    owned = set()
    for item in request.session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        name = getattr(item, "originalname", None) or item.name
        if fnmatch(name, "test_*e2e*") and params.get("case_name") not in (None, UNMATCHED):
            owned.add(params["case_name"])
    if wanted & owned:
        pytest.skip(f"another collected family owns --e2e-testcase {', '.join(sorted(wanted))}")
    pytest.fail(
        f"--e2e-testcase {', '.join(sorted(wanted))} matches no collected E2E case; "
        f"{FAMILY} has {', '.join(sorted(CASES))}"
    )


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        model_filters, testcase_filters = _filters(metafunc.config)
        parameters = names
        if enabled and not names and testcase_filters and not model_filters:
            # Pytest would skip the empty parametrization; whether the testcase
            # is a typo is known only once the whole session is collected.
            parameters = names = [UNMATCHED]
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


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, native_build


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            tensor_parallel_size=_manifest_value(manifest, "tensor_parallel_size", (int,)),
        )
    )


def _asset(raw: str) -> Path:
    path = TEST_ROOT / raw
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _read_pcm16(path: Path):
    import numpy as np

    with wave.open(str(path)) as handle:
        assert handle.getsampwidth() == 2 and handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / np.float32(32768.0)


def _conversation():
    """About 56 s of two people taking turns, with pauses and one overlap.

    Long enough for three encoder chunks of 27.2 s, so the later chunks run
    behind a speaker cache that has already been compressed; the second
    speaker returns after the first has spoken at length, which is what the
    arrival-order cache exists to remember. The length is not a whole number
    of 10 ms frames, so the last encoder frame carries padding.
    """
    import numpy as np

    first, second = _read_pcm16(_asset(FIRST_SPEAKER)), _read_pcm16(_asset(SECOND_SPEAKER))

    def pause(seconds: float):
        return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)

    second_quiet = second * np.float32(0.8)
    turns = [pause(0.5), first[: 10 * SAMPLE_RATE], pause(0.3), second,
             first[10 * SAMPLE_RATE: 18 * SAMPLE_RATE], pause(0.4), second_quiet,
             first[18 * SAMPLE_RATE:], pause(0.25), second, first[: 8 * SAMPLE_RATE],
             pause(0.6), second_quiet, first[8 * SAMPLE_RATE: 14 * SAMPLE_RATE],
             np.zeros(77, dtype=np.float32)]
    audio = np.concatenate(turns)
    overlap = int(40.0 * SAMPLE_RATE)
    audio[overlap: overlap + second.size] += np.float32(0.6) * second
    return audio


def _input_audio(case: dict, directory: Path):
    """The case's waveform, as the PCM16 WAV the binary reads and its float32 decoding.

    Every reader decodes PCM16 as value / 32768, so the binary, the SDK
    consumers and the reference all see the same samples.
    """
    import numpy as np

    layouts = {
        "conversation": _conversation,
        "single_speaker": lambda: _read_pcm16(_asset(FIRST_SPEAKER)),
    }
    audio = layouts[case["test_input"]]()
    pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype("<i2")
    path = directory / f"{case['name']}.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return path, pcm.astype(np.float32) / np.float32(32768.0)


def _environment(runtime_root: Path) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    return env


def _native(binary: Path, runtime_root: Path, bundle: Path, wav: Path) -> dict:
    """The family's own command, `trtmc nemotron3_diarization diarize` (cli.json)."""
    completed = subprocess.run(
        [str(binary), FAMILY, "diarize", str(bundle), "--runtime-root", str(runtime_root),
         "--audio", str(wav)],
        check=True, capture_output=True, text=True, env=_environment(runtime_root), timeout=3600,
    )
    payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 1, f"native diarize returned no single JSON: {completed.stdout[-1000:]}"
    result = json.loads(payloads[0])
    assert result["task"] == "audio_to_timed_labels"
    return result


def _sdk_results(native_build: Path, runtime_root: Path, bundle: Path, samples,
                 directory: Path) -> list[dict]:
    """The public C and C++ consumers on the same float32 samples the binary decoded."""
    raw = directory / "sdk-input.f32"
    samples.tofile(raw)
    outputs = []
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [str(consumer), str(bundle), str(runtime_root), str(raw), str(SAMPLE_RATE)],
            check=True, capture_output=True, text=True, env=_environment(runtime_root),
            timeout=3600,
        )
        payloads = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(payloads) == 1, f"the {language} consumer must print one result"
        outputs.append(json.loads(payloads[0]))
    return outputs


def _official_reference(model_dir: Path, samples):
    """Transformers' own Nemotron3Diarization in float32 on the GPU, TF32 off.

    The speaker cache keeps the frames its scores rank highest. The native
    runtime breaks an exact tie deterministically, by the first index seen
    (`top_k_first_seen`). PyTorch does not guarantee the order of tied
    indices in torch.topk; on CUDA the reference's selections agree with that
    rule on the E2E inputs, so the reference runs on CUDA, not the CPU. A tie
    broken differently changes the frames the cache keeps: the first chunk,
    which runs before any cache state, is gated on its own, and the
    later-chunk gate then fails with a message that points at the cache.
    """
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    # TF32 is process-wide: off for this run only, and restored even when it
    # fails. Matmul goes through the float32 matmul precision alone: setting
    # it after `matmul.allow_tf32` mixes PyTorch's two APIs, and the next
    # reference run in the process then fails to read the precision.
    saved = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        processor = AutoProcessor.from_pretrained(str(model_dir))
        model = AutoModelForAudioFrameClassification.from_pretrained(
            str(model_dir), dtype=torch.float32).to("cuda").eval()
        inputs = processor(samples, sampling_rate=SAMPLE_RATE).to("cuda")
        with torch.inference_mode():
            logits = model(**inputs).logits
        mask = inputs["attention_mask"]
        valid = int(mask[0].sum())
        segments = processor.extract_speaker_dict(logits, mask)[0]
        reference = logits[0, :valid].double().cpu().numpy()
        del model
        torch.cuda.empty_cache()
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cudnn.allow_tf32 = saved[1]
    return reference, [(item["Start"], item["End"], item["Speaker"]) for item in segments]


def _active(logits):
    """The processor's rule: the float32 logistic of the logit above 0.5."""
    import numpy as np

    values = logits.astype(np.float32)
    return np.float32(1.0) / (np.float32(1.0) + np.exp(-values)) > np.float32(0.5)


def _runs(active) -> list[tuple[int, int, int]]:
    runs = []
    for speaker in range(active.shape[1]):
        start = None
        for frame in range(active.shape[0] + 1):
            on = frame < active.shape[0] and bool(active[frame, speaker])
            if on and start is None:
                start = frame
            if not on and start is not None:
                runs.append((start, frame, speaker))
                start = None
    return sorted(runs)


def _frames(intervals) -> list[tuple[int, int, int]]:
    return sorted((round(start * 100), round(end * 100), int(label))
                  for start, end, label in intervals)


# A native endpoint is a whole frame times the 10 ms hop, divided once in
# double precision, so it lies on the grid to about 1e-12 of a frame. Rounding
# to frames alone would hide an endpoint that is off by less than 5 ms.
GRID_TOLERANCE_FRAMES = 1e-6


def _native_frames(intervals) -> list[tuple[int, int, int]]:
    """Native intervals as frames, after checking every endpoint against the 10 ms grid."""
    intervals = list(intervals)
    for start, end, label in intervals:
        for seconds in (start, end):
            offset = abs(seconds * 100 - round(seconds * 100))
            assert offset <= GRID_TOLERANCE_FRAMES, (
                f"native interval endpoint {seconds!r} s of label {label} is {offset:.3g} frames "
                "off the 10 ms frame grid")
    return _frames(intervals)


def _check(name: str, label: str, actual, operator: str, expected) -> dict:
    """One row of a reference_comparison record, with the verdict its values state."""
    passed = actual <= expected if operator == "<=" else actual == expected
    return {"name": name, "label": label, "scope": "independent_reference", "actual": actual,
            "operator": operator, "expected": expected, "passed": bool(passed)}


def _timeline(path: Path, native_active, reference_active) -> tuple[Path, int]:
    """A PNG of who is active when: per speaker, the native row above the reference row.

    Returns the file and the frames one pixel column covers. Written with zlib,
    so the report needs no plotting package.
    """
    import struct
    import zlib

    import numpy as np

    frames = native_active.shape[0]
    speakers = [speaker for speaker in range(native_active.shape[1])
                if native_active[:, speaker].any() or reference_active[:, speaker].any()]
    step = max(1, -(-frames // 1200))
    columns = max(1, -(-frames // step))
    lane, gap = 10, 6
    image = np.full((max(1, len(speakers)) * (2 * lane + gap), columns, 3), 255, dtype=np.uint8)
    for row, speaker in enumerate(speakers):
        for offset, active, color in ((0, native_active, (31, 119, 180)),
                                      (lane, reference_active, (255, 127, 14))):
            padded = np.zeros(columns * step, dtype=bool)
            padded[:frames] = active[:, speaker]
            on = padded.reshape(columns, step).any(axis=1)
            top = row * (2 * lane + gap) + offset
            band = image[top:top + lane - 1]
            band[:, ~on] = (230, 230, 230)
            band[:, on] = color

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + line.tobytes() for line in image)
    header = struct.pack(">IIBBBBB", columns, image.shape[0], 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return path, step


def _first_chunk_frames(model_dir: Path) -> int:
    """Spectrogram frames of the first offline chunk, which runs before any cache state."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return int(config["chunk_length"]) * int(config["audio_config"]["subsampling_factor"])


def _assert_parity(native: dict, reference, segments, case: dict, first_chunk: int,
                   evidence: tuple[Path, Path] | None = None) -> None:
    """The native result against the reference; `evidence` names the retained native and
    reference logits files, next to which the comparison records keep their interval tables."""
    import numpy as np

    assert native["labels"] == [f"speaker_{index}" for index in range(8)]
    assert native["vocabulary_id"] == "", "speakers are result-local, not a vocabulary"
    assert native["score_kind"] == "logit" and native["frame_seconds"] == 0.01
    logits = np.asarray(native["frame_scores"], dtype=np.float64)
    assert logits.shape == reference.shape, (
        f"frame count differs: native={logits.shape} reference={reference.shape}")

    # The gate is on every logit of every frame and speaker, not on the
    # segments: a segment boundary moves only when a logit crosses zero.
    # Calibration of the manifests' 1e-3 on one B300, float32 against the
    # float32 reference: healthy builds stay at 1.6e-4 (conversation, 3 runs of
    # 44,800 logits) and 5.2e-5 (single speaker, 3 runs of 18,648 logits);
    # builds made defective on purpose miss it on the conversation by orders:
    # 0.10 (tanh GELU), 0.45 (periodic Hann window), 3.7 (no latest-frames
    # boost in the cache), 39 (interleaved RoPE pairs).
    tolerance = _manifest_value(case, "logit_tolerance", (int, float))
    assert math.isfinite(tolerance) and tolerance > 0, f"unusable logit_tolerance {tolerance!r}"
    gap = np.abs(logits - reference)
    worst = np.unravel_index(int(np.argmax(gap)), gap.shape)
    first = float(gap[:first_chunk].max())
    print(f"{case['name']}: {logits.shape[0]} frames, worst logit gap {gap[worst]:.3e} at "
          f"frame {worst[0]} speaker {worst[1]}, mean gap {gap.mean():.3e}, first chunk "
          f"{first:.3e}")
    if evidence is not None:
        record_evidence("reference_comparison", {
            "label": "Frame logits against the Transformers reference",
            "scope": "independent_reference", "enforced": True,
            "native": evidence[0], "reference": evidence[1],
            "checks": [
                _check("first_chunk_logit_gap", "First chunk: max |native - reference| logit",
                       first, "<=", float(tolerance)),
                _check("logit_gap", "All frames and speakers: max |native - reference| logit",
                       float(gap[worst]), "<=", float(tolerance)),
            ],
        })
    # The first chunk sees no speaker-cache state, so a gap there is the front
    # end, the engines or the step input, never the cache.
    assert first <= tolerance, f"first-chunk logit gap {first:.3e} exceeds {tolerance}"
    # Later chunks add what the first never reaches: the cached and queued rows
    # ahead of the chunk, positions past the first step, the padding mask at
    # the end, and which frames the cache kept. Its selections break float32
    # near-ties among nearly saturated frames identically in the native and
    # reference runs, so a gap here is not rounding: a float64 reference, which
    # keeps other frames, moves these logits by about 0.4 without changing a
    # decision.
    assert gap[worst] <= tolerance, (
        f"logit gap {gap[worst]:.3e} at frame {worst[0]} (first chunk {first:.3e}) exceeds "
        f"{tolerance}: look at the later steps' cached and queued rows, positions past the "
        "first step, the padding mask and the frames the speaker cache kept")

    native_active, reference_active = _active(logits), _active(reference)
    native_frames = _native_frames((item["start_seconds"], item["end_seconds"], item["label"])
                                   for item in native["intervals"])
    reference_frames = _frames(segments)
    assert native_frames == _runs(native_active), (
        "native intervals are not the runs of its own frame decisions")
    assert reference_frames == _runs(reference_active), (
        "the reference segments are not the runs of its frame decisions")
    # A decision may differ only where the reference itself sits within the
    # logit tolerance of the threshold; anywhere else it is a defect.
    flipped = np.argwhere(native_active != reference_active)
    if evidence is not None:
        directory = evidence[0].parent
        tables = (directory / "native_intervals.npy", directory / "reference_intervals.npy")
        for table, frames in zip(tables, (native_frames, reference_frames)):
            np.save(table, np.asarray(frames, dtype=np.int64).reshape(-1, 3))
        beyond = sum(1 for frame, speaker in flipped if abs(reference[frame, speaker]) > tolerance)
        checks = [_check("decisions_flipped_beyond_tolerance",
                         "Frame decisions that differ where |reference logit| exceeds the tolerance",
                         beyond, "==", 0)]
        if not len(flipped):
            differing = len(set(native_frames) ^ set(reference_frames))
            checks.append(_check("intervals_differing",
                                 "Intervals, as [start frame, end frame, speaker], not in both results",
                                 differing, "==", 0))
        record_evidence("reference_comparison", {
            "label": "Speaker intervals against the Transformers reference segments",
            "scope": "independent_reference", "enforced": True,
            "native": tables[0], "reference": tables[1], "checks": checks,
        })
    for frame, speaker in flipped:
        assert abs(reference[frame, speaker]) <= tolerance, (
            f"frame {frame} speaker {speaker} flipped with reference logit "
            f"{reference[frame, speaker]:.4f}")
    if not len(flipped):
        assert native_frames == reference_frames
    speakers = sorted({speaker for _, _, speaker in segments})
    print(f"{case['name']}: {len(native['intervals'])} intervals, reference speakers {speakers}, "
          f"{len(flipped)} near-threshold decisions differ; setup {native['setup_ms']:.1f} ms, "
          f"inference {native['inference_ms']:.1f} ms")
    if case["test_input"] == "conversation":
        assert speakers[:2] == [0, 1], "the conversation must exercise two speakers"


def test_e2e(tmp_path: Path, case_name: str, request) -> None:
    if case_name == UNMATCHED:
        _resolve_unmatched(request)
    manifest_path, manifest, case = CASES[case_name]
    assert manifest_path.is_file()
    binary, runtime_root, native_build = _runtime()
    model_dir = _model_dir(manifest)
    bundle = tmp_path / _manifest_value(manifest, "bundle", (str,))
    _build(model_dir, bundle, manifest)
    assert bundle.is_file(), f"{FAMILY} bundle was not produced: {bundle}"

    import numpy as np

    wav, samples = _input_audio(case, tmp_path)
    record_evidence("inputs", {
        "case": case, "audio": wav, "sample_rate": SAMPLE_RATE,
        "seconds": round(len(samples) / SAMPLE_RATE, 3),
        "checkpoint": f"{manifest['hf_id']}@{manifest['hf_revision']}",
    })
    native = _native(binary, runtime_root, bundle, wav)
    # The consumers print float32 scores with nine digits, the CLI with the
    # double that holds them; both name the same float32 values exactly.
    native_scores = np.asarray(native["frame_scores"], dtype=np.float32)
    for sdk in _sdk_results(native_build, runtime_root, bundle, samples, tmp_path):
        for key in ("labels", "vocabulary_id", "intervals", "frame_seconds", "score_kind"):
            assert sdk[key] == native[key], f"the public SDK result differs from the CLI's {key}"
        assert np.array_equal(np.asarray(sdk["frame_scores"], dtype=np.float32), native_scores), (
            "the public SDK frame scores differ from the CLI's")
    reference, segments = _official_reference(model_dir, samples)
    native_file, reference_file = tmp_path / "native_logits.npy", tmp_path / "reference_logits.npy"
    np.save(native_file, native_scores)
    np.save(reference_file, reference)
    timeline, step = _timeline(tmp_path / "speaker_timeline.png", _active(native_scores),
                               _active(reference))
    record_evidence("native", {
        "intervals": native["intervals"], "labels": native["labels"],
        "frame_seconds": native["frame_seconds"], "score_kind": native["score_kind"],
        "frame_scores": native_file, "timeline": timeline,
        "setup_ms": native["setup_ms"], "inference_ms": native["inference_ms"],
    })
    record_evidence("reference", {
        "mode": "hf_transformers", "logits": reference_file,
        "segments": [{"start_seconds": float(start), "end_seconds": float(end),
                      "speaker": int(speaker)} for start, end, speaker in segments],
    })
    record_evidence("thresholds", {
        "logit_tolerance": _manifest_value(case, "logit_tolerance", (int, float))})
    record_evidence("views", [{
        "title": "Speaker timeline",
        "caption": "One row pair per active speaker: native above (blue), reference below "
                   f"(orange); a pixel column covers {step * 10} ms.",
    }])
    _assert_parity(native, reference, segments, case, _first_chunk_frames(model_dir),
                   evidence=(native_file, reference_file))


def test_parity_refuses_native_endpoints_off_the_frame_grid() -> None:
    """An endpoint off the 10 ms grid fails even when rounding to frames would hide it."""
    import numpy as np

    logits = np.full((6, 8), -5.0)
    logits[1:4, 0] = 5.0
    case = {"name": "frame-grid", "logit_tolerance": 1e-3, "test_input": "single_speaker"}
    segments = [(0.01, 0.04, 0)]

    def native(start: float, end: float) -> dict:
        return {"labels": [f"speaker_{index}" for index in range(8)], "vocabulary_id": "",
                "score_kind": "logit", "frame_seconds": 0.01, "frame_scores": logits.tolist(),
                "intervals": [{"start_seconds": start, "end_seconds": end, "label": 0}],
                "setup_ms": 0.0, "inference_ms": 0.0}

    _assert_parity(native(0.01, 0.04), logits, segments, case, first_chunk=6)
    # 2 ms and 4.9 ms off the grid: both still round to frames 1 and 4.
    for start, end in ((0.012, 0.04), (0.01, 0.0449)):
        with pytest.raises(AssertionError, match="off the 10 ms frame grid"):
            _assert_parity(native(start, end), logits, segments, case, first_chunk=6)


def test_the_reference_run_restores_the_tf32_settings(monkeypatch) -> None:
    """The reference turns TF32 off for its own run only, also when that run fails,
    and the next reference run in the same process works too."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    def fail(*args, **kwargs):
        raise RuntimeError("reference failed")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fail)
    precision, cudnn = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    try:
        for incoming in ("highest", "medium"):
            torch.set_float32_matmul_precision(incoming)
            torch.backends.cudnn.allow_tf32 = True
            for _ in range(2):
                with pytest.raises(RuntimeError, match="reference failed"):
                    _official_reference(Path("unused"), None)
                assert torch.get_float32_matmul_precision() == incoming
                assert torch.backends.cuda.matmul.allow_tf32 == (incoming != "highest")
                assert torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn
