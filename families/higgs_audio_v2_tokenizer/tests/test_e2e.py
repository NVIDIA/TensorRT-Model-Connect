# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Higgs Audio V2 codec token to PCM parity through the public Task API."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "higgs_audio_v2_tokenizer"
TASKS = frozenset({"audio_codebook_tokens_to_audio"})
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
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


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
    if evidence_enabled():
        for key in ("image", "test_image", "audio"):
            if key in case:
                for source in (TEST_ROOT / case[key], TEST_ROOT / "data" / case[key]):
                    if source.is_file():
                        record_evidence("inputs", {key: source})
                        break
    record_evidence(
        "checkpoint",
        {
            "repo_id": manifest["hf_id"],
            "revision": manifest["hf_revision"],
            "precision": manifest["precision"],
        },
    )
    record_evidence(
        "thresholds",
        {
            key: value
            for key, value in case.items()
            if any(word in key for word in ("error", "cosine", "atol", "rtol", "snr", "rmse"))
        },
    )
    import wave
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from transformers import HiggsAudioV2TokenizerModel

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    with wave.open(str(TEST_ROOT / "data" / case["audio"]), "rb") as audio:
        assert (
            audio.getframerate() == 16000
            and audio.getnchannels() == 1
            and audio.getsampwidth() == 2
        )
        waveform = np.frombuffer(audio.readframes(32000), dtype="<i2").astype(np.float32) / 32768
    waveform = resample_poly(waveform, 3, 2).astype(np.float32)
    assert waveform.shape == (48000,)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        HiggsAudioV2TokenizerModel.from_pretrained(directory, dtype=torch.float32).eval().to("cuda")
    )
    views = []
    for sample_index, gain in enumerate((1.0, 0.25)):
        # The encoder supplies real speech tokens to the decode-only native Task.
        with torch.no_grad():
            codes = reference.encode(
                torch.from_numpy(waveform * gain)[None, None].to("cuda"), return_dict=True
            ).audio_codes
            expected = (
                reference.decode(codes, return_dict=True).audio_values.cpu().numpy().reshape(-1)
            )
        assert codes.shape == (1, 8, 50)
        source = tmp_path / f"codes-{gain}.i32"
        target = tmp_path / f"audio-{gain}.f32"
        source.write_bytes(codes.cpu().numpy().astype("<i4").tobytes())
        completed = subprocess.run(
            [
                str(_native_test_binary("test_higgs_audio_v2_tokenizer_runtime")),
                str(bundle),
                str(runtime_root),
                str(source),
                str(target),
                "50",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        payload = json.loads(completed.stdout.splitlines()[-1])
        assert payload == dict(samples=48000, rate=24000, channels=1)
        actual = np.fromfile(target, dtype="<f4")
        record_evidence(
            "inputs", {"gain": gain, "audio_codes": codes.cpu().numpy(), "sample_rate": 24000}
        )
        record_evidence("native", {"gain": gain, "waveform": actual, "metadata": payload})
        record_evidence(
            "reference", {"mode": "hf_transformers", "gain": gain, "waveform": expected}
        )
        record_evidence(
            "native_process",
            {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
        )
        if evidence_enabled():
            import struct

            paired = {}
            for role, samples in (("native", actual), ("reference", expected)):
                audio_path = tmp_path / f"{role}-{sample_index}.wav"
                paired[role] = audio_path
                pcm = samples.astype("<f4").tobytes()
                audio_path.write_bytes(
                    struct.pack(
                        "<4sI4s4sIHHIIHH4sI",
                        b"RIFF",
                        36 + len(pcm),
                        b"WAVE",
                        b"fmt ",
                        16,
                        3,
                        1,
                        24000,
                        96000,
                        4,
                        32,
                        b"data",
                        len(pcm),
                    )
                    + pcm
                )
                views.append(
                    {
                        "title": f"Sample {sample_index} / {role}",
                        "audio": audio_path,
                        "caption": f"Input speech gain {gain}",
                    }
                )
            record_evidence("views", views[-2:])
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        error = actual.astype(np.float64) - expected.astype(np.float64)
        maximum = np.abs(error).max()
        snr = 10 * np.log10(np.sum(expected.astype(np.float64) ** 2) / max(np.sum(error**2), 1e-30))
        assert np.max(np.abs(expected)) > 0.01
        record_evidence(
            "metrics", {"gain": gain, "max_abs_error": float(maximum), "snr_db": float(snr)}
        )
        assert maximum <= case["max_abs_error"]
        assert snr >= case["min_snr_db"]
        if evidence_enabled():
            record_evidence(
                "reference_comparison",
                {
                    "label": f"Decoded speech at gain {gain}",
                    "scope": "independent_reference",
                    "enforced": True,
                    **paired,
                    "checks": [
                        {
                            "name": name,
                            "scope": "independent_reference",
                            "actual": value,
                            "operator": operator,
                            "expected": limit,
                            "passed": value <= limit if operator == "<=" else value >= limit,
                        }
                        for name, value, operator, limit in [
                            ("waveform_max_abs_error", float(maximum), "<=", case["max_abs_error"]),
                            ("waveform_snr_db", float(snr), ">=", case["min_snr_db"]),
                        ]
                    ],
                },
            )
        print(f"HiggsAudioV2Tokenizer gain={gain}: max_abs={maximum:.8g}, SNR={snr:.3f} dB")
