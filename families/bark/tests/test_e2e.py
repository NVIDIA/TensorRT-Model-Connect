# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for bark."""

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

FAMILY = "bark"
TASKS = frozenset({"audio_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


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
        timeout=int(case.get("runtime_timeout_s", 3600)),
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


@cache
def _token_trace_binary() -> tuple[Path, Path]:
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            "8",
            "--target",
            "bark_token_trace",
            "trtmc_backend_trt",
        ],
        check=True,
        timeout=600,
    )
    binary = build_dir / "families" / FAMILY / "bark_token_trace"
    assert binary.is_file(), f"selected {FAMILY} E2E token trace binary is missing: {binary}"
    assert (build_dir / "libtrtmc_backend_trt.so").is_file()
    return binary, build_dir


def _token_trace(
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
) -> tuple[list[int], list[int]]:
    prefix = tmp_path / "bark-token-trace"
    binary, trace_runtime_root = _token_trace_binary()
    invocation = [
        str(binary),
        str(bundle),
        str(trace_runtime_root),
        str(prefix),
        _case_text(case),
        str(int(case["max_new_tokens"])),
        str(int(case["seed"])),
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
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (
            str(trace_runtime_root),
            str(runtime_root),
            environment.get("LD_LIBRARY_PATH", ""),
        )
        if value
    )
    subprocess.run(invocation, check=True, env=environment, timeout=1800)

    def read(suffix: str) -> list[int]:
        path = Path(f"{prefix}.rank0{suffix}")
        assert path.is_file(), f"selected {FAMILY} E2E token trace is missing: {path}"
        return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    return read(".sem_tokens"), read(".coarse_tokens")


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


def _case_text(case: dict) -> str:
    inputs = case.get("inputs") or {}
    value = str(case.get("prompt") or case.get("test_prompt") or inputs.get("prompt") or "")
    assert value, f"selected {FAMILY} E2E requires a direct prompt"
    return value


def _torch_dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def _normalized_edit_distance(actual: str, expected: str) -> float:
    actual = " ".join(actual.split()).strip().lower()
    expected = " ".join(expected.split()).strip().lower()
    if not actual and not expected:
        return 0.0
    if len(actual) < len(expected):
        actual, expected = expected, actual
    previous = list(range(len(expected) + 1))
    for row, actual_character in enumerate(actual, start=1):
        current = [row]
        for column, expected_character in enumerate(expected, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (actual_character != expected_character),
                )
            )
        previous = current
    return previous[-1] / max(len(actual), len(expected))


def _asr_dependency(manifest: dict) -> tuple[str, str | None]:
    dependencies = manifest["hf_dependencies"]
    assert isinstance(dependencies, list)
    matches = [
        dependency
        for dependency in dependencies
        if dependency.get("repo_id") == "openai/whisper-large-v3-turbo"
    ]
    assert len(matches) == 1, (
        f"selected {FAMILY} E2E requires one explicit Whisper ASR checkpoint dependency"
    )
    revision = matches[0].get("revision")
    return str(matches[0]["repo_id"]), str(revision) if revision else None


def _asr_model_dir(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    repo_id, revision = _asr_dependency(manifest)
    try:
        return Path(snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True))
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the cached ASR checkpoint {repo_id}"
        ) from error


def _transcribe_wavs(paths: list[Path], manifest: dict) -> list[str]:
    import librosa
    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    assert torch.cuda.is_available(), f"selected {FAMILY} ASR round-trip requires CUDA"
    model_dir = _asr_model_dir(manifest)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
        .to("cuda:0")
        .eval()
    )
    transcripts = []
    try:
        target_rate = int(processor.feature_extractor.sampling_rate)
        for path in paths:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim == 2:
                samples = np.mean(samples, axis=1)
            assert samples.ndim == 1 and samples.size > 0
            if int(sample_rate) != target_rate:
                samples = librosa.resample(
                    samples, orig_sr=int(sample_rate), target_sr=target_rate
                ).astype(np.float32)
            inputs = processor(samples, sampling_rate=target_rate, return_tensors="pt")
            inputs = {
                key: (
                    value.to("cuda:0", dtype=torch.float16)
                    if value.is_floating_point()
                    else value.to("cuda:0")
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=256)
            transcript = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            assert transcript, f"selected {FAMILY} ASR round-trip produced an empty transcript"
            transcripts.append(transcript)
    finally:
        del model
        torch.cuda.empty_cache()
    return transcripts


def _assert_audio_health(samples: np.ndarray, sample_rate: int, thresholds: dict) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    assert samples.ndim == 1 and samples.size > 0
    assert np.isfinite(samples).all()
    duration = samples.size / sample_rate
    assert duration >= float(thresholds["duration_s_min"])
    assert duration <= float(thresholds["duration_s_max"])
    rms = float(np.sqrt(np.mean(samples**2)))
    assert rms >= float(thresholds["rms_min"])
    assert rms <= float(thresholds["rms_max"])


def _read_healthy_wav(path: Path, thresholds: dict) -> tuple[np.ndarray, int]:
    import soundfile as sf

    info = sf.info(path)
    assert info.format == "WAV"
    assert int(info.channels) == 1
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    _assert_audio_health(samples, int(sample_rate), thresholds)
    return samples, int(sample_rate)


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    model_dir: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
):
    manifest["task"]
    output = tmp_path / "native.wav"
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "generate-audio",
        "--prompt",
        _case_text(case),
        "--output",
        str(output),
        "--max-new-tokens",
        str(int(case["max_new_tokens"])),
        "--seed",
        str(int(case["seed"])),
    )
    assert output.is_file()
    payload["audio"] = str(output)
    thresholds = _thresholds(str(case["name"]))
    if {"min_semantic_tokens", "golden_semantic_tokens", "golden_coarse_tokens"} & set(thresholds):
        payload["semantic_tokens"], payload["coarse_tokens"] = _token_trace(
            runtime_root, bundle, manifest, case, tmp_path
        )
    return payload


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import torch
    from transformers import AutoProcessor, BarkModel

    processor = AutoProcessor.from_pretrained(model_dir)
    model = BarkModel.from_pretrained(model_dir).to("cuda").eval()
    torch.manual_seed(int(case["seed"]))
    encoded = processor(_case_text(case), return_tensors="pt")
    encoded = {
        key: value.to("cuda") if hasattr(value, "to") else value for key, value in encoded.items()
    }
    with torch.no_grad():
        audio = model.generate(**encoded)
    return {
        "samples": audio.float().cpu().numpy().reshape(-1),
        "sample_rate": int(model.generation_config.sample_rate),
    }


def _assert_token_trace(actual: dict, thresholds: dict) -> None:
    if "min_semantic_tokens" in thresholds:
        semantic_tokens = actual["semantic_tokens"]
        assert len(semantic_tokens) >= int(thresholds["min_semantic_tokens"])
    if "golden_semantic_tokens" in thresholds:
        semantic_tokens = actual["semantic_tokens"]
        golden = [int(value) for value in thresholds["golden_semantic_tokens"]]
        assert semantic_tokens[: len(golden)] == golden
    if "golden_coarse_tokens" in thresholds:
        golden = [int(value) for value in thresholds["golden_coarse_tokens"]]
        assert actual["coarse_tokens"][: len(golden)] == golden


def test_token_trace_keeps_the_old_minimum_and_golden_gates() -> None:
    actual = {"semantic_tokens": [1, 2, 3], "coarse_tokens": [4, 5]}
    _assert_token_trace(
        actual,
        {
            "min_semantic_tokens": 3,
            "golden_semantic_tokens": [1, 2],
            "golden_coarse_tokens": [4, 5],
        },
    )


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    import librosa

    actual_samples, actual_rate = _read_healthy_wav(Path(actual["audio"]), thresholds)
    expected_samples = np.asarray(expected["samples"]).reshape(-1)
    assert expected_samples.size > 0 and np.isfinite(expected_samples).all()
    assert int(actual_rate) == int(expected["sample_rate"])
    common = min(actual_samples.size, expected_samples.size)
    assert common > 0
    duration_ratio = actual_samples.size / max(expected_samples.size, 1)
    assert duration_ratio >= float(thresholds["duration_ratio_min"])
    assert duration_ratio <= float(thresholds["duration_ratio_max"])
    actual_mel = librosa.feature.melspectrogram(
        y=actual_samples[:common], sr=int(actual_rate), n_mels=80
    )
    expected_mel = librosa.feature.melspectrogram(
        y=expected_samples[:common], sr=int(actual_rate), n_mels=80
    )
    frames = min(actual_mel.shape[1], expected_mel.shape[1])
    actual_db = librosa.power_to_db(actual_mel[:, :frames] + 1e-12)
    expected_db = librosa.power_to_db(expected_mel[:, :frames] + 1e-12)
    mel_distance = float(np.mean(np.abs(actual_db - expected_db)))
    spectral_distance = float(np.sqrt(np.mean((actual_db - expected_db) ** 2)))
    assert mel_distance <= float(thresholds["mel_spectrogram_distance"])
    assert spectral_distance <= float(thresholds["log_spectral_distance"])
    _assert_token_trace(actual, thresholds)
    transcript = _transcribe_wavs([Path(actual["audio"])], manifest)[0]
    assert _normalized_edit_distance(transcript, _case_text(case)) <= float(
        thresholds["asr_ned_max"]
    )
    return


def test_audio_contract_helpers_are_strict() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate // 5, dtype=np.float32) / sample_rate
    samples = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    thresholds = {
        "duration_s_min": 0.1,
        "duration_s_max": 30.0,
        "rms_min": 0.005,
        "rms_max": 1.0,
    }
    _assert_audio_health(samples, sample_rate, thresholds)
    with pytest.raises(AssertionError):
        _assert_audio_health(samples[:10], sample_rate, thresholds)
    assert _normalized_edit_distance("  Hello WORLD ", "hello world") == 0.0
    assert _normalized_edit_distance("unrelated", "hello world") > 0.15
    for _, manifest, _ in CASES.values():
        assert _asr_dependency(manifest)[0] == "openai/whisper-large-v3-turbo"


def test_asr_checkpoint_lookup_is_offline_and_fail_closed(monkeypatch) -> None:
    call = {}

    def unavailable(**kwargs):
        call.update(kwargs)
        raise FileNotFoundError("not cached")

    monkeypatch.setattr("huggingface_hub.snapshot_download", unavailable)
    manifest = next(iter(CASES.values()))[1]
    with pytest.raises(AssertionError, match="requires the cached ASR checkpoint"):
        _asr_model_dir(manifest)
    assert call["local_files_only"] is True


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path.parent / manifest["bundle"]
    if not bundle.is_file():
        _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
