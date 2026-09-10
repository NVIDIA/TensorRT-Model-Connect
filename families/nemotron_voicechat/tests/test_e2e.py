# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for nemotron_voicechat."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence
import json
import os
import re
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
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": getattr(completed, "stdout", None), "stderr": getattr(completed, "stderr", None)})
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
    record_evidence("reference_assets", {"reference_audio": path})
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


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _edit_distance(left: str, right: str) -> float:
    a = " ".join(_normalized_words(left))
    b = " ".join(_normalized_words(right))
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


def test_text_similarity_ignores_punctuation_only_differences() -> None:
    assert _edit_distance("Hello, world!", "hello world") == 0.0


def test_transcript_word_count_ignores_punctuation_only_tokens() -> None:
    assert _normalized_words("hello ... !!! world") == ["hello", "world"]


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
    probe = _run_lifecycle_probe(runtime_root, bundle, case, tmp_path, baseline_only=True)
    assert probe["probe_returncode"] == 0
    output = probe["audio"]
    receipt = probe["receipt"]
    baseline = receipt["baseline"]
    payload = {
        "receipt": receipt,
        "probe_returncode": probe["probe_returncode"],
        "audio": str(output),
        "text": str(baseline["agent_text"]),
    }
    payload["source_stats"] = _wav_stats(source)
    payload["output_stats"] = _wav_stats(output)
    payload["reported_audio_samples"] = int(baseline["output_samples"])
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


def _run_lifecycle_probe(
    runtime_root: Path,
    bundle: Path,
    case: dict,
    tmp_path: Path,
    *,
    baseline_only: bool,
) -> dict:
    inputs = case.get("inputs") or {}
    source = _speech_source(case, "speech_source_relative_path")
    mode = "baseline" if baseline_only else "lifecycle"
    output = tmp_path / f"{mode}.wav"
    receipt_path = tmp_path / f"{mode}.json"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value
    )
    invocation = [
        str(_lifecycle_binary(int(inputs.get("lifecycle_build_timeout_s", 600)))),
        str(bundle),
        str(source),
        str(runtime_root),
        str(output),
        str(receipt_path),
    ]
    if baseline_only:
        invocation.append("--baseline-only")
    completed = subprocess.run(
        invocation,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=int(
            inputs.get("lifecycle_runtime_timeout_s", inputs.get("runtime_timeout_s", 1800))
        ),
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": getattr(completed, "stdout", None), "stderr": getattr(completed, "stderr", None)})
    assert completed.returncode in {0, 1}, completed.stderr[-2000:]
    assert receipt_path.is_file(), "VoiceChat lifecycle probe did not write its receipt"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert not receipt.get("error"), receipt.get("error")
    assert output.is_file(), "VoiceChat lifecycle probe did not write its audio artifact"
    return {"receipt": receipt, "probe_returncode": completed.returncode, "audio": output}


def _native_lifecycle(runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> dict:
    _speech_source(case, "function_speech_source_relative_path")
    probe = _run_lifecycle_probe(runtime_root, bundle, case, tmp_path, baseline_only=False)
    return {"receipt": probe["receipt"], "probe_returncode": probe["probe_returncode"]}


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
    assert actual["reported_audio_samples"] == output["num_samples"]
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
    assert 1.0 - _edit_distance(actual_text, expected["text"]) >= float(
        thresholds["agent_text_min_similarity"]
    )
    normalized_text = actual_text.casefold()
    assert all(
        str(term).casefold() in normalized_text for term in expected["required_response_terms"]
    )
    transcript = str(actual["transcript"])
    assert len(_normalized_words(transcript)) >= int(thresholds["transcript_min_words"])
    assert 1.0 - _edit_distance(transcript, expected["text"]) >= _TRANSCRIPT_MIN_SIMILARITY
    assert int(expected["sample_rate"]) == int(expected["expected_output_sample_rate"])
    assert expected["samples"].size == int(expected["expected_output_num_samples"])
    return


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": CASES[case_name][-1]})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")})
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    with evidence_stage("native"):
        actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    record_evidence("native", actual)
    with evidence_stage("reference"):
        expected = _official_reference(model_dir, manifest, case, tmp_path)
    from families.nemotron_voicechat.tests.reporting import record_audio_views

    record_audio_views(actual, expected, tmp_path / "report-views")
    record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_parity(actual, expected, manifest, case, record_evidence("thresholds", _thresholds(case_name)))


def test_manifest_declares_text_tokenizer_dependency() -> None:
    for _, manifest, _ in CASES.values():
        assert manifest["hf_dependencies"] == [{"repo_id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2"}]
