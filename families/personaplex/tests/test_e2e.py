# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for personaplex."""

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

FAMILY = "personaplex"
TASKS = frozenset({"speech_to_speech"})
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


def test_manifests_declare_mimi_dependency() -> None:
    expected = [{"repo_id": "kyutai/mimi"}]
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["hf_dependencies"] == expected, path


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
    payload = payloads[0]
    payload["_stderr"] = completed.stderr
    return payload


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

    payload = _run_json(
        Path("/trtmc"),
        Path("/runtime"),
        tmp_path / "model.bundle",
        {"tensor_parallel_size": 4},
        {},
        "speak",
        "--input",
        "/input.wav",
        rank_output_root=output_root,
    )
    assert payload["rank"] == 0
    command = captured["command"]
    wrapper = command[command.index("bash") + 2]
    assert str(output_root) in command
    assert "OMPI_COMM_WORLD_RANK" in wrapper
    assert "rank_${rank}" in wrapper and '--output "$out/native.wav"' in wrapper
    assert "PMI" not in wrapper and "RANK:-" not in wrapper


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
    source = _asset(case["test_input_audio"])
    tp_size = int(manifest["tensor_parallel_size"])
    rank_output_root = tmp_path / "rank-outputs" if tp_size > 1 else None
    output = (
        rank_output_root / "rank_0/native.wav"
        if rank_output_root is not None
        else tmp_path / "native.wav"
    )
    arguments = [
        "--input",
        str(source),
        "--max-new-tokens",
        str(int(case["max_new_tokens"])),
        "--seed",
        str(int(case["seed"])),
        "--tail-frames",
        str(int(case["speech_test_max_frames"])),
    ]
    if rank_output_root is None:
        arguments.extend(("--output", str(output)))
    payload = _run_json(
        binary,
        runtime_root,
        bundle,
        manifest,
        case,
        "speak",
        *arguments,
        rank_output_root=rank_output_root,
    )
    assert output.is_file()
    payload["audio"] = str(output)
    frames = []
    for line in payload["_stderr"].splitlines():
        if "<stderr>:" in line and (not line.startswith("[1,0]<stderr>:")):
            continue
        match = re.search("\\[speech\\] Output frame [0-9]+:(.*)$", line)
        if match:
            frames.append([int(token) for token in match.group(1).split()])
    assert frames, "PersonaPlex native runtime emitted no output speech tokens"
    payload["speech_tokens"] = np.asarray(frames, dtype=np.int32)
    return payload


def _official_reference(model_dir: Path, manifest: dict, case: dict, tmp_path: Path):
    manifest["task"]

    reference_tokens = _asset(case["speech_reference_tokens"])
    return {"speech_tokens": np.load(reference_tokens, allow_pickle=False)}


def _assert_token_parity(
    actual_tokens: np.ndarray, expected_tokens: np.ndarray, thresholds: dict
) -> None:
    actual_tokens = np.asarray(actual_tokens, dtype=np.int32)
    expected_tokens = np.asarray(expected_tokens, dtype=np.int32)
    assert actual_tokens.ndim == 2
    assert expected_tokens.ndim == 2
    assert actual_tokens.shape[0] > 0
    assert actual_tokens.shape[0] == expected_tokens.shape[0]
    assert actual_tokens.shape == expected_tokens.shape
    assert actual_tokens.shape[1] > 1
    depth_agreement = float(np.mean(actual_tokens[:, 0] == expected_tokens[:, 0]))
    audio_agreement = float(np.mean(actual_tokens[:, 1:] == expected_tokens[:, 1:]))
    token_agreement = float(np.mean(actual_tokens == expected_tokens))
    frame_agreement = float(np.mean(np.all(actual_tokens == expected_tokens, axis=1)))
    assert depth_agreement >= float(thresholds["depth_token_match_rate"])
    assert audio_agreement >= float(thresholds["audio_token_match_rate"])
    assert token_agreement >= float(thresholds["speech_min_token_match"])
    assert frame_agreement >= float(thresholds["speech_min_frame_exact"])


def _assert_audio_health(samples: np.ndarray, sample_rate: int, thresholds: dict) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    assert samples.size > 0 and np.isfinite(samples).all()
    assert samples.shape[0] / sample_rate >= float(thresholds["speech_min_duration_s"])
    assert float(np.sqrt(np.mean(samples**2))) >= float(thresholds["speech_min_rms"])


def _assert_parity(actual, expected, manifest: dict, case: dict, thresholds: dict) -> None:
    manifest["task"]
    import soundfile as sf

    _assert_token_parity(actual["speech_tokens"], expected["speech_tokens"], thresholds)
    samples, sample_rate = sf.read(actual["audio"], dtype="float32", always_2d=False)
    _assert_audio_health(samples, int(sample_rate), thresholds)
    return


def test_personaplex_contract_requires_exact_untruncated_frames() -> None:
    expected_tokens = np.arange(30, dtype=np.int32).reshape(10, 3)
    thresholds = {
        "depth_token_match_rate": 0.7,
        "audio_token_match_rate": 0.7,
        "speech_min_token_match": 0.8,
        "speech_min_frame_exact": 0.7,
        "speech_min_duration_s": 0.1,
        "speech_min_rms": 0.001,
    }
    _assert_token_parity(expected_tokens.copy(), expected_tokens, thresholds)
    _assert_audio_health(np.full(1_600, 0.1, dtype=np.float32), 16_000, thresholds)
    with pytest.raises(AssertionError):
        _assert_token_parity(expected_tokens[:-1], expected_tokens, thresholds)
    exact_frame_failure = expected_tokens.copy()
    exact_frame_failure[:4, 1] += 1
    with pytest.raises(AssertionError):
        _assert_token_parity(exact_frame_failure, expected_tokens, thresholds)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    actual = _native(binary, runtime_root, bundle, model_dir, manifest, case, tmp_path)
    expected = _official_reference(model_dir, manifest, case, tmp_path)
    _assert_parity(actual, expected, manifest, case, _thresholds(case_name))
