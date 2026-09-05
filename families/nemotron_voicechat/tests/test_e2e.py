# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for nemotron_voicechat."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "nemotron_voicechat"
TASKS = frozenset({"speech_session"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
_TRANSCRIPT_MIN_SIMILARITY = 0.35


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


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    required_gpus = int(manifest["tensor_parallel_size"])
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected {FAMILY} E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
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
            image_height=manifest.get("image_height"),
            image_width=manifest.get("image_width"),
            video_num_frames=manifest.get("video_num_frames"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            quantization=manifest.get("quantization"),
            fp32_layers=tuple((int(layer) for layer in manifest.get("fp32_layers", ()))),
        )
    )


def _run_json(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    command: str,
    *arguments: str,
    timeout_s: int | None = None,
) -> dict:
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    if int(manifest["tensor_parallel_size"]) > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-np",
            str(manifest["tensor_parallel_size"]),
            *invocation,
        ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    )
    completed = subprocess.run(
        invocation,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s or int(case.get("runtime_timeout_s", 3600)),
    )
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


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def _speech_source(case: dict, field: str) -> Path:
    root = _required_path(
        os.environ.get("TRTMC_REFERENCE_SOURCE_DIR"), "TRTMC_REFERENCE_SOURCE_DIR"
    )
    relative = str((case.get("inputs") or {})[field])
    path = root / relative
    assert path.is_file(), f"selected {FAMILY} E2E source asset does not exist: {path}"
    return path


def _wav_stats(path: Path) -> dict:
    import soundfile as sf

    info = sf.info(path)
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert samples.shape[1] == 1
    values = np.asarray(samples[:, 0], dtype=np.float32)
    return {
        "channels": int(info.channels),
        "sample_rate": int(rate),
        "num_samples": int(values.size),
        "subtype": str(info.subtype),
        "all_finite": bool(np.isfinite(values).all()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "peak": float(np.max(np.abs(values))),
    }


def _complete_agent_text(events: object) -> str:
    assert isinstance(events, list), "speech-session events must be a list"
    last_sequence_by_epoch: dict[int, int] = {}
    agent_text_epochs: list[int] = []
    final_text_by_epoch: dict[int, str] = {}
    for event in events:
        assert isinstance(event, dict), "speech-session events must be objects"
        epoch = event.get("epoch")
        sequence = event.get("sequence")
        assert type(epoch) is int and epoch > 0, "speech-session event epoch must be positive"
        assert type(sequence) is int and sequence >= 0, (
            "speech-session event sequence must be non-negative"
        )
        previous = last_sequence_by_epoch.get(epoch)
        assert previous is None or sequence > previous, (
            f"speech-session event sequence is not increasing for epoch {epoch}"
        )
        last_sequence_by_epoch[epoch] = sequence

        if event.get("kind") != "agent_text":
            continue
        is_final = event.get("is_final")
        text = event.get("text")
        assert type(is_final) is bool, "agent_text is_final must be a boolean"
        assert isinstance(text, str), "agent_text text must be a string"
        if epoch not in agent_text_epochs:
            agent_text_epochs.append(epoch)
        assert epoch not in final_text_by_epoch, f"agent_text epoch {epoch} continued after final"
        if is_final:
            assert text.strip(), f"agent_text epoch {epoch} final text must not be empty"
            final_text_by_epoch[epoch] = text

    assert agent_text_epochs, "speech-session produced no agent_text events"
    missing = [epoch for epoch in agent_text_epochs if epoch not in final_text_by_epoch]
    assert not missing, f"agent_text epochs missing a final event: {missing}"
    return " ".join(final_text_by_epoch[epoch] for epoch in agent_text_epochs)


def _edit_distance(left: str, right: str) -> float:
    a = " ".join(left.lower().split())
    b = " ".join(right.lower().split())
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, start=1):
        current = [index]
        for offset, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b)
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b), 1)


def _native_model_card(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    inputs = case.get("inputs") or {}
    source = _speech_source(case, "speech_source_relative_path")
    output = tmp_path / "native.wav"
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "speech-session",
        "--input",
        str(source),
        "--output",
        str(output),
        "--timeout-ms",
        str(int(inputs.get("runtime_timeout_s", 1800)) * 1000),
        timeout_s=int(inputs.get("runtime_timeout_s", 1800)),
    )
    assert output.is_file()
    payload["audio"] = str(output)
    payload["text"] = _complete_agent_text(payload.get("events"))
    payload["source_stats"] = _wav_stats(source)
    payload["output_stats"] = _wav_stats(output)
    payload["event_audio_samples"] = sum(
        int(event.get("audio_samples", 0)) for event in payload["events"]
    )
    transcription = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "transcribe",
        "--input",
        str(output),
        "--max-output-tokens",
        str(int(case["max_new_tokens"])),
        timeout_s=int(inputs.get("transcribe_timeout_s", 1800)),
    )
    payload["transcript"] = str(transcription["text"])
    return payload


@cache
def _lifecycle_binary(timeout_s: int) -> Path:
    build_value = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    assert build_value, "selected VoiceChat lifecycle E2E requires TRTMC_NATIVE_BUILD_DIR"
    build_dir = Path(build_value)
    assert build_dir.is_dir()
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            "8",
            "--target",
            "test_nemotron_voicechat_lifecycle_probe_host",
        ],
        check=True,
        timeout=timeout_s,
    )
    probe = build_dir / "test_nemotron_voicechat_lifecycle_probe_host"
    assert probe.is_file()
    return probe


def _native_lifecycle(runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> dict:
    inputs = case.get("inputs") or {}
    source = _speech_source(case, "speech_source_relative_path")
    _speech_source(case, "function_speech_source_relative_path")
    output = tmp_path / "lifecycle.wav"
    receipt_path = tmp_path / "lifecycle.json"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        [
            str(_lifecycle_binary(int(inputs["lifecycle_build_timeout_s"]))),
            str(bundle),
            str(source),
            str(runtime_root),
            str(output),
            str(receipt_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(inputs["lifecycle_runtime_timeout_s"]),
    )
    assert completed.returncode in {0, 1}, completed.stderr[-2000:]
    assert receipt_path.is_file(), "VoiceChat lifecycle probe did not write its receipt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert not receipt.get("error"), receipt.get("error")
    assert output.is_file(), "VoiceChat lifecycle probe did not write its audio artifact"
    return {"receipt": receipt, "probe_returncode": completed.returncode}


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    del model_dir
    if case["name"] == "nemotron-voicechat-11b-full-duplex-lifecycle":
        return _native_lifecycle(runtime_root, bundle, case, tmp_path)
    return _native_model_card(binary, runtime_root, bundle, manifest, case, tmp_path)


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    del model_dir, tmp_path
    import soundfile as sf

    reference = _asset((case.get("inputs") or {})["reference_audio"])
    samples, rate = sf.read(reference, dtype="float32")
    return {
        "samples": np.asarray(samples).reshape(-1),
        "sample_rate": int(rate),
        "text": str(case.get("expected_response_text", "")),
        "speech_source_sample_rate": case.get("speech_source_sample_rate"),
        "speech_source_num_samples": case.get("speech_source_num_samples"),
        "expected_output_sample_rate": case.get("expected_output_sample_rate"),
        "expected_output_num_samples": case.get("expected_output_num_samples"),
        "expected_output_samples_per_frame": case.get("expected_output_samples_per_frame"),
        "expected_output_codec_frames": case.get("expected_output_codec_frames"),
        "required_response_terms": case.get("required_response_terms", []),
    }


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    if case["name"] == "nemotron-voicechat-11b-full-duplex-lifecycle":
        from families.nemotron_voicechat.tests.lifecycle_oracle import assert_lifecycle_receipt

        assert_lifecycle_receipt(actual["receipt"], str(case["expected_response_text"]))
        assert actual["probe_returncode"] == 0
        return

    source = actual["source_stats"]
    output = actual["output_stats"]
    assert source["channels"] == 1
    assert source["sample_rate"] == int(expected["speech_source_sample_rate"])
    assert source["num_samples"] == int(expected["speech_source_num_samples"])
    assert output["channels"] == 1 and output["subtype"] == "FLOAT"
    assert output["all_finite"] is True
    assert output["sample_rate"] == int(expected["expected_output_sample_rate"])
    assert output["num_samples"] == int(expected["expected_output_num_samples"])
    assert actual["event_audio_samples"] == output["num_samples"]
    frame_samples = int(expected["expected_output_samples_per_frame"])
    assert output["num_samples"] % frame_samples == 0
    codec_frames = output["num_samples"] // frame_samples
    assert codec_frames == int(expected["expected_output_codec_frames"])
    input_frames = (source["num_samples"] + 1279) // 1280 + int(
        (case.get("inputs") or {}).get("tail_frames", 0)
    )
    assert codec_frames == input_frames
    assert output["rms"] >= float(thresholds["audio_min_rms"])
    assert output["peak"] >= float(thresholds["audio_min_peak"])
    actual_text = str(actual["text"])
    assert len(actual_text.split()) >= int(thresholds["transcript_min_words"])
    assert 1.0 - _edit_distance(actual_text, expected["text"]) >= float(
        thresholds["agent_text_min_similarity"]
    )
    normalized_text = actual_text.casefold()
    assert all(
        str(term).casefold() in normalized_text for term in expected["required_response_terms"]
    )
    transcript = str(actual["transcript"])
    assert len(transcript.split()) >= int(thresholds["transcript_min_words"])
    assert 1.0 - _edit_distance(transcript, expected["text"]) >= _TRANSCRIPT_MIN_SIMILARITY
    assert int(expected["sample_rate"]) == int(expected["expected_output_sample_rate"])
    assert expected["samples"].size == int(expected["expected_output_num_samples"])
    return


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))


def test_manifest_declares_text_tokenizer_dependency() -> None:
    for _, manifest, _ in CASES.values():
        assert manifest["hf_dependencies"] == [{"repo_id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2"}]


def test_complete_agent_text_uses_the_final_event_for_each_epoch() -> None:
    def agent_text(epoch: int, sequence: int, text: str, is_final: bool) -> dict:
        return {
            "kind": "agent_text",
            "epoch": epoch,
            "sequence": sequence,
            "text": text,
            "is_final": is_final,
        }

    events = [
        {"kind": "turn_started", "epoch": 2, "sequence": 0},
        agent_text(2, 1, "Hi ", False),
        agent_text(2, 2, "there !", False),
        agent_text(2, 3, "Hi there!", True),
        {"kind": "turn_finished", "epoch": 2, "sequence": 4},
        {"kind": "turn_started", "epoch": 4, "sequence": 0},
        agent_text(4, 1, "The sky ", False),
        agent_text(4, 2, "is blue.", False),
        agent_text(4, 3, "The sky is blue.", True),
        {"kind": "turn_finished", "epoch": 4, "sequence": 4},
    ]
    original_events = json.loads(json.dumps(events))

    assert _complete_agent_text(events) == "Hi there! The sky is blue."
    assert events == original_events


def test_complete_agent_text_rejects_an_incomplete_epoch() -> None:
    events = [
        {
            "kind": "agent_text",
            "epoch": 2,
            "sequence": 0,
            "text": "incomplete",
            "is_final": False,
        }
    ]

    with pytest.raises(AssertionError, match="missing a final event"):
        _complete_agent_text(events)
