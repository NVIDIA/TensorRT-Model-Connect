# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned W2v-BERT spectral frame embedding parity through the public Task API."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "wav2vec2_bert"
TASKS = frozenset({"spectrogram_to_frame_features"})
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


def _model_dir(manifest: dict) -> tuple[Path, str | None]:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR"), None
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot), manifest["hf_revision"]


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


def test_official_checkpoint_e2e(case_name, tmp_path, request):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "reproduction_requirements",
        {
            "checkpoint": "Cache the exact manifest revision or set TRTMC_WAV2VEC2_BERT_MODEL_DIR (unverified local checkpoint).",
            "environment": ["TRTMC_BINARY", "TRTMC_RUNTIME_ROOT", "TRTMC_NATIVE_BUILD_DIR"],
            "native_build": "Configure CMake with TRTMC_BUILD_TESTS=ON; the test builds test_wav2vec2_bert_runtime in TRTMC_NATIVE_BUILD_DIR.",
        },
    )
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
            "hf_id": manifest["hf_id"],
            "expected_hf_revision": manifest["hf_revision"],
            "hf_revision": None,
            "revision_status": "unresolved",
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
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory, checkpoint_revision = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(directory),
            "hf_revision": checkpoint_revision,
            "revision_status": "unverified"
            if checkpoint_revision is None
            else "pinned_cache_lookup",
        },
    )
    _, runtime_root = _runtime()
    with wave.open(str(TEST_ROOT / "data" / case["audio"]), "rb") as audio:
        assert (
            audio.getframerate() == 16000
            and audio.getnchannels() == 1
            and audio.getsampwidth() == 2
        )
        waveform = np.frombuffer(audio.readframes(32000), dtype="<i2").astype(np.float32) / 32768
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(matmul_precision))
    request.addfinalizer(lambda: setattr(torch.backends.cudnn, "allow_tf32", cudnn_tf32))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        Wav2Vec2BertModel.from_pretrained(
            directory, torch_dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(directory)
    for label, samples in [("speech", waveform), ("reversed", waveform[::-1].copy())]:
        inputs = extractor(samples, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features.contiguous()
        assert features.shape == (1, 99, 160) and inputs.attention_mask.all()
        source = tmp_path / f"spectral-{label}.f32"
        target = tmp_path / f"features-{label}.f32"
        source.write_bytes(features.numpy().astype("<f4").tobytes())
        with torch.no_grad():
            expected = reference(**inputs.to("cuda")).last_hidden_state[0].cpu().numpy()
        completed = subprocess.run(
            [
                str(_native_test_binary("test_wav2vec2_bert_runtime")),
                str(bundle),
                str(runtime_root),
                str(source),
                str(target),
                "99",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        payload = json.loads(completed.stdout.splitlines()[-1])
        assert payload == dict(
            rows=99, columns=1024, source_frames=99, input_frame_indices=list(range(99))
        )
        actual = np.fromfile(target, dtype="<f4").reshape(99, 1024)
        record_evidence(
            "inputs", {"sample": label, "features": features.numpy(), "waveform": samples}
        )
        record_evidence("native", {"sample": label, "features": actual, "metadata": payload})
        record_evidence(
            "reference", {"mode": "hf_transformers", "sample": label, "features": expected}
        )
        record_evidence(
            "native_process",
            {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
        )
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        maximum = float(np.max(np.abs(actual - expected)))
        cosine = np.sum(actual.astype(np.float64) * expected, axis=1) / (
            np.linalg.norm(actual.astype(np.float64), axis=1)
            * np.linalg.norm(expected.astype(np.float64), axis=1)
        )
        record_evidence(
            "metrics",
            {"sample": label, "max_abs_error": maximum, "min_cosine": float(cosine.min())},
        )
        pair = {}
        for role, values in (("native", actual), ("reference", expected)):
            path = tmp_path / f"{role}-{label}.npy"
            np.save(path, values, allow_pickle=False)
            pair[role] = path
        record_evidence(
            "reference_comparison",
            {
                "label": f"W2v-BERT {label} frame features",
                "scope": "independent_reference",
                "enforced": True,
                **pair,
                "checks": [
                    {
                        "name": "max_abs_error",
                        "scope": "independent_reference",
                        "actual": maximum,
                        "operator": "<=",
                        "expected": case["max_abs_error"],
                        "passed": maximum <= case["max_abs_error"],
                    },
                    {
                        "name": "min_cosine",
                        "scope": "independent_reference",
                        "actual": float(cosine.min()),
                        "operator": ">=",
                        "expected": case["min_cosine"],
                        "passed": float(cosine.min()) >= case["min_cosine"],
                    },
                ],
            },
        )
        assert maximum <= case["max_abs_error"]
        assert float(cosine.min()) >= case["min_cosine"]
        print(f"W2v-BERT {label}: max_abs={maximum:.8g}, min_cosine={cosine.min():.9f}")
