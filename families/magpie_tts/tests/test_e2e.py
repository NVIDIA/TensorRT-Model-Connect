# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for magpie_tts."""

from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import pytest
import numpy as np
from tensorrt_model_connect import BuildRequest, build

FAMILY = "magpie_tts"
TASKS = frozenset({"audio_generation"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
_MPI_RANK_ZERO = re.compile(r"^\[[^,]+,0\]<stdout>:(.*)$")


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
    rank_output_root: Path | None = None,
) -> dict:
    tp_size = int(manifest["tensor_parallel_size"])
    invocation = [
        str(binary),
        command,
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        *arguments,
    ]
    if tp_size > 1:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected multi-GPU E2E requires mpirun"
        if rank_output_root is not None:
            invocation = [
                "bash",
                "-c",
                'rank="${OMPI_COMM_WORLD_RANK:?missing rank}"; '
                'out="$1/rank_${rank}"; mkdir -p "$out"; shift; '
                'exec "$@" --output "$out/native.wav"',
                "trtmc_rank_audio",
                str(rank_output_root),
                *invocation,
            ]
        invocation = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-x",
            "TRTMC_NCCL_RENDEZVOUS",
            "-np",
            str(tp_size),
            *invocation,
        ]
    env = os.environ.copy()
    env["TRTMC_NCCL_RENDEZVOUS"] = str(bundle.with_suffix(".nccl-rendezvous"))
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
        if tp_size > 1:
            match = _MPI_RANK_ZERO.fullmatch(line)
            candidate = match.group(1) if match else ""
        else:
            start = line.find("{")
            candidate = line[start:] if start >= 0 else ""
        if candidate.lstrip().startswith("{"):
            try:
                payloads.append(json.loads(candidate))
            except json.JSONDecodeError:
                pass
    assert payloads, f"native {command} returned no JSON: {completed.stdout[-1000:]}"
    assert all((payload == payloads[0] for payload in payloads))
    return payloads[0]


def test_tp_audio_launcher_uses_rank_local_output_and_rank_zero_json(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        stdout = "\n".join(('[1,1]<stdout>:{"rank":1}', '[1,0]<stdout>:{"rank":0}'))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/mpirun")
    monkeypatch.setattr(subprocess, "run", fake_run)
    output_root = tmp_path / "rank-outputs"

    assert _run_json(
        Path("/trtmc"),
        Path("/runtime"),
        tmp_path / "model.bundle",
        {"tensor_parallel_size": 4},
        {},
        "generate-audio",
        "--prompt",
        "hello",
        rank_output_root=output_root,
    ) == {"rank": 0}
    command = captured["command"]
    wrapper = command[command.index("bash") + 2]
    assert str(output_root) in command
    assert "OMPI_COMM_WORLD_RANK" in wrapper
    assert "rank_${rank}" in wrapper and '--output "$out/native.wav"' in wrapper
    assert "PMI" not in wrapper and "RANK:-" not in wrapper


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    if not path.is_file():
        return {}
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
    assert duration >= float(thresholds.get("contract_min_duration_s", 0.1))
    assert duration <= float(thresholds.get("contract_max_duration_s", 30.0))
    rms = float(np.sqrt(np.mean(samples**2)))
    assert rms >= float(thresholds.get("contract_min_rms", 0.001))


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
    tp_size = int(manifest["tensor_parallel_size"])
    rank_output_root = tmp_path / "rank-outputs" if tp_size > 1 else None
    output = (
        rank_output_root / "rank_0/native.wav"
        if rank_output_root is not None
        else tmp_path / "native.wav"
    )
    arguments = [
        "--prompt",
        _case_text(case),
        "--max-new-tokens",
        str(int(case["max_new_tokens"])),
        "--seed",
        str(int(case["seed"])),
    ]
    if rank_output_root is None:
        arguments.extend(("--output", str(output)))
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "generate-audio",
        *arguments,
        rank_output_root=rank_output_root,
    )
    assert output.is_file()
    payload["audio"] = str(output)
    return payload


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]
    import fsspec
    import soundfile as sf
    import torch
    from unittest.mock import patch
    from nemo.collections.tts.models import MagpieTTSModel
    from huggingface_hub import hf_hub_download

    archive = model_dir if model_dir.is_file() else model_dir / "magpie_tts_multilingual_357m.nemo"
    assert archive.is_file(), f"selected {FAMILY} E2E requires the materialized NeMo archive"
    speaker_checkpoint = hf_hub_download(
        repo_id="Edresson/Speaker_Encoder_H_ASP",
        filename="pytorch_model.bin",
        local_files_only=True,
    )
    speaker_checkpoint_url = (
        "https://huggingface.co/Edresson/Speaker_Encoder_H_ASP/resolve/main/pytorch_model.bin"
    )
    original_fsspec_open = fsspec.open

    def offline_fsspec_open(path, *args, **kwargs):
        if str(path).split("?", 1)[0] == speaker_checkpoint_url:
            path = speaker_checkpoint
        return original_fsspec_open(path, *args, **kwargs)

    with patch.object(fsspec, "open", offline_fsspec_open):
        model = MagpieTTSModel.restore_from(restore_path=archive).to("cuda").eval()
        torch.manual_seed(int(case["seed"]))
        with torch.no_grad():
            audio, length = model.do_tts(transcript=_case_text(case), language="en", use_cfg=True)
    samples = audio[0, : int(length.item())].float().cpu().numpy()
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        samples = samples / peak
    path = tmp_path / "official-nemo.wav"
    sf.write(path, samples, 22050, subtype="PCM_16")
    return {"samples": samples, "sample_rate": 22050, "audio": str(path)}


def _assert_contract(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    _read_healthy_wav(Path(actual["audio"]), thresholds)
    _read_healthy_wav(Path(expected["audio"]), thresholds)
    native_transcript, reference_transcript = _transcribe_wavs(
        [Path(actual["audio"]), Path(expected["audio"])], manifest
    )
    prompt = _case_text(case)
    limit = float(thresholds.get("contract_asr_ned_threshold", 0.15))
    assert _normalized_edit_distance(native_transcript, prompt) <= limit
    assert _normalized_edit_distance(reference_transcript, prompt) <= float(limit)


def test_audio_contract_helpers_are_strict() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate // 5, dtype=np.float32) / sample_rate
    samples = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    thresholds = {
        "contract_min_duration_s": 0.1,
        "contract_max_duration_s": 30.0,
        "contract_min_rms": 0.005,
    }
    _assert_audio_health(samples, sample_rate, thresholds)
    with pytest.raises(AssertionError):
        _assert_audio_health(samples[:10], sample_rate, thresholds)
    assert _normalized_edit_distance("  Hello WORLD ", "hello world") == 0.0
    assert _normalized_edit_distance("unrelated", "hello world") > 0.15
    expected_dependencies = {
        "Edresson/Speaker_Encoder_H_ASP",
        "google/byt5-small",
        "microsoft/wavlm-base-plus",
        "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
        "openai/whisper-large-v3-turbo",
    }
    for _, manifest, _ in CASES.values():
        assert {
            dependency["repo_id"] for dependency in manifest["hf_dependencies"]
        } == expected_dependencies
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
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_contract(actual, expected, manifest, case, _thresholds(case_name))
