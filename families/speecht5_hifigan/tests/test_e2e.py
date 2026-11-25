# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned SpeechT5 HiFiGAN spectral-feature to PCM parity through the public Task API."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

from ._checkpoint import checkpoint_record

FAMILY = "speecht5_hifigan"
TASKS = frozenset({"spectrogram_to_audio"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


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


@cache
def _native_test_binary(name: str) -> Path:
    """Build source-only helpers in the configured native test tree."""
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


def _record_comparison(tmp_path, label, native, reference, checks):
    """Retain the output pair and the result of already enforced comparisons."""
    if not evidence_enabled():
        return
    import numpy as np

    paths = []
    for role, value in (("native", native), ("reference", reference)):
        if isinstance(value, np.ndarray):
            path = tmp_path / f"{label}-{role}.npz"
            np.savez_compressed(path, output=value)
        else:
            path = tmp_path / f"{label}-{role}.json"
            path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    record_evidence(
        "reference_comparison",
        {
            "label": label,
            "scope": "independent_reference",
            "enforced": True,
            "native": paths[0],
            "reference": paths[1],
            "checks": [dict(check, scope="independent_reference") for check in checks],
        },
    )


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


def _runtime() -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    assert torch.cuda.device_count() >= 1, f"selected {FAMILY} E2E requires one GPU"
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
    case: dict,
    command: str,
    *arguments: str,
) -> dict:
    with evidence_stage("runtime"):
        invocation = [
            str(binary),
            command,
            str(bundle),
            "--runtime-root",
            str(runtime_root),
            *arguments,
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join(
            (value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
        )
        completed = subprocess.run(
            invocation,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=int(case.get("runtime_timeout_s", 3600)),
        )
        record_evidence(
            "native_process",
            {
                "argv": completed.args,
                "stdout": (completed.stdout or "")[-4096:],
                "stderr": (completed.stderr or "")[-16384:],
                "returncode": completed.returncode,
            },
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
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


def test_official_checkpoint_e2e(case_name, tmp_path):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "thresholds",
        {
            key: value
            for key, value in case.items()
            if any(
                word in key
                for word in ("error", "cosine", "atol", "rtol", "snr", "rmse", "max_abs")
            )
        },
    )
    for asset_key in ("image", "test_image", "audio"):
        if asset_key in case:
            for asset_path in (TEST_ROOT / case[asset_key], TEST_ROOT / "data" / case[asset_key]):
                if asset_path.is_file():
                    record_evidence("inputs", {asset_key: asset_path})
                    break
    import wave
    import numpy as np
    import torch
    from transformers import SpeechT5FeatureExtractor, SpeechT5HifiGan

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
        record_evidence(
            "checkpoint",
            checkpoint_record(
                directory,
                manifest,
                local_override=bool(os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")),
            ),
        )
    _, runtime_root = _runtime()
    with wave.open(str(TEST_ROOT / "data" / case["audio"]), "rb") as audio:
        with evidence_stage("comparison"):
            assert (
                audio.getframerate() == 16000
                and audio.getnchannels() == 1
                and audio.getsampwidth() == 2
            )
        waveform = (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with evidence_stage("reference"):
        reference = (
            SpeechT5HifiGan.from_pretrained(directory, dtype=torch.float32).eval().to("cuda")
        )
    extractor = SpeechT5FeatureExtractor()
    for gain in (1.0, 0.5):
        features = extractor(
            audio_target=waveform * gain, sampling_rate=16000, return_tensors="pt"
        ).input_values
        features = features[:, : manifest["max_sequence_length"], :].contiguous()
        with evidence_stage("comparison"):
            assert features.shape == (1, 64, 80)
        source = tmp_path / f"mel-{gain}.f32"
        target = tmp_path / f"audio-{gain}.f32"
        source.write_bytes(features.numpy().astype("<f4").tobytes())
        with evidence_stage("reference"), torch.no_grad():
            expected = reference(features.to("cuda")).cpu().numpy().reshape(-1)
        with evidence_stage("runtime"):
            completed = subprocess.run(
                [
                    str(_native_test_binary("test_speecht5_hifigan_runtime")),
                    str(bundle),
                    str(runtime_root),
                    str(source),
                    str(target),
                    str(features.shape[1]),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=int(case.get("runtime_timeout_s", 3600)),
            )
        record_evidence(
            "runtime_process",
            {
                "argv": completed.args,
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[-4096:],
                "stderr": (completed.stderr or "")[-16384:],
            },
        )
        with evidence_stage("comparison"):
            assert completed.returncode == 0, completed.stderr + completed.stdout
        payload = json.loads(completed.stdout.splitlines()[-1])
        with evidence_stage("comparison"):
            assert payload == dict(samples=16384, rate=16000, channels=1)
        actual = np.fromfile(target, dtype="<f4")
        with evidence_stage("comparison"):
            assert actual.shape == expected.shape and np.isfinite(actual).all()
        record_evidence("inputs", {"gain": gain, "mel": features.numpy()})
        record_evidence("native", {"gain": gain, "waveform": actual, "sample_rate": 16000})
        record_evidence(
            "reference",
            {"mode": "hf_transformers", "gain": gain, "waveform": expected, "sample_rate": 16000},
        )
        if evidence_enabled():
            views = []
            for role, samples in (("native", actual), ("reference", expected)):
                output = tmp_path / f"{role}-{gain}.wav"
                with wave.open(str(output), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    stream.writeframes(
                        np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
                    )
                views.append(
                    {
                        "title": f"Gain {gain} / {role}",
                        "audio": output,
                        "caption": "16 kHz mono waveform; PCM16 listening preview",
                    }
                )
            record_evidence("views", views)
        error = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = np.abs(error).max()
        snr = 10 * np.log10(np.sum(expected.astype(np.float64) ** 2) / max(np.sum(error**2), 1e-30))
        with evidence_stage("comparison"):
            assert np.max(np.abs(expected)) > 0.01
        record_evidence(
            "metrics", {"gain": gain, "max_abs_error": float(maximum), "snr_db": float(snr)}
        )
        with evidence_stage("comparison"):
            assert maximum <= case["max_abs_error"]
        with evidence_stage("comparison"):
            assert snr >= case["min_snr_db"]
        _record_comparison(
            tmp_path,
            f"waveform-{gain}",
            actual,
            expected,
            [
                {
                    "name": "max_abs_error",
                    "actual": float(maximum),
                    "operator": "<=",
                    "expected": case["max_abs_error"],
                    "passed": bool(maximum <= case["max_abs_error"]),
                },
                {
                    "name": "snr_below_limit",
                    "actual": int(snr < case["min_snr_db"]),
                    "operator": "==",
                    "expected": 0,
                    "passed": bool(snr >= case["min_snr_db"]),
                },
            ],
        )
        print(f"HiFiGAN gain={gain}: max_abs={maximum:.8g}, SNR={snr:.3f} dB")
