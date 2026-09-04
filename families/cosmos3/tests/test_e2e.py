# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build and native-runtime E2E for Cosmos3-Nano."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "cosmos3"
TASK = "image_generation"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
MPI_RANK_ZERO = re.compile(r"^\[[^,]+,0\]<stdout>:(.*)$")


def _case_index() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == TASK
        assert manifest["tensor_parallel_size"] == 1
        assert manifest["context_parallel_size"] in (1, 2)
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
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
    for name, (manifest, _) in CASES.items():
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
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = names
    if not enabled:
        parameters = [
            pytest.param(
                name,
                marks=pytest.mark.skip(
                    reason="direct E2E requires one of the explicit E2E selectors"
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
    explicit = os.environ.get("TRTMC_COSMOS3_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_COSMOS3_MODEL_DIR")
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    except Exception as error:
        raise AssertionError(
            "selected cosmos3 E2E requires the exact cached nvidia/Cosmos3-Nano checkpoint"
        ) from error
    return Path(snapshot)


def _runtime(manifest: dict) -> tuple[Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_cosmos3.so").is_file()

    import torch

    required_gpus = int(manifest["context_parallel_size"])
    assert torch.cuda.is_available(), "selected cosmos3 E2E requires CUDA"
    assert torch.cuda.device_count() >= required_gpus, (
        f"selected cosmos3 E2E requires {required_gpus} GPUs, found {torch.cuda.device_count()}"
    )
    return binary, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=TASK,
            precision=manifest["precision"],
            max_sequence_length=int(manifest["max_sequence_length"]),
            image_height=int(manifest["image_height"]),
            image_width=int(manifest["image_width"]),
            video_num_frames=int(manifest["video_num_frames"]),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
            context_parallel_size=int(manifest["context_parallel_size"]),
        )
    )


def _native(
    binary: Path,
    runtime_root: Path,
    bundle: Path,
    manifest: dict,
    case: dict,
    tmp_path: Path,
) -> Path:
    output = tmp_path / "native-frames"
    command = [
        str(binary),
        "generate-video",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        str(case["test_prompt"]),
        "--output",
        str(output),
        "--height",
        str(manifest["image_height"]),
        "--width",
        str(manifest["image_width"]),
        "--num-steps",
        str(case["num_inference_steps"]),
        "--guidance-scale",
        str(case["guidance_scale"]),
        "--seed",
        str(case["seed"]),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    cp_size = int(manifest["context_parallel_size"])
    if cp_size == 2:
        mpirun = shutil.which("mpirun")
        assert mpirun, "selected Cosmos3 CP2 E2E requires mpirun"
        rendezvous = tmp_path / "cosmos3.nccl"
        env["TRTMC_NCCL_RENDEZVOUS"] = str(rendezvous)
        command = [
            mpirun,
            "--tag-output",
            "-x",
            "LD_LIBRARY_PATH",
            "-np",
            "2",
            "-x",
            f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}",
            "-x",
            f"TRTMC_NCCL_RENDEZVOUS={rendezvous}",
            *command,
        ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case["runtime_timeout_s"]),
    )
    if cp_size == 1:
        payloads = [
            json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")
        ]
    else:
        payloads = [
            json.loads(match.group(1))
            for line in completed.stdout.splitlines()
            if (match := MPI_RANK_ZERO.fullmatch(line))
        ]
    assert len(payloads) == 1, completed.stdout[-2000:]
    assert Path(payloads[0]["output"]) == output
    return output


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _assert_video(output: Path, thresholds: dict) -> None:
    from PIL import Image

    frames = sorted(output.glob("frame-*.png"))
    assert len(frames) == int(thresholds["exact_num_frames"])
    total = 0.0
    total_squared = 0.0
    count = 0
    for path in frames:
        pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        assert pixels.shape == (
            int(thresholds["exact_video_height"]),
            int(thresholds["exact_video_width"]),
            3,
        )
        total += float(pixels.sum(dtype=np.float64))
        total_squared += float(np.square(pixels, dtype=np.float64).sum(dtype=np.float64))
        count += int(pixels.size)
    mean = total / count
    standard_deviation = max(total_squared / count - mean * mean, 0.0) ** 0.5
    assert mean >= float(thresholds["min_pixel_mean"])
    assert mean <= float(thresholds["max_pixel_mean"])
    assert standard_deviation >= float(thresholds["min_pixel_std"])


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime(manifest)
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    output = _native(binary, runtime_root, bundle, manifest, case, tmp_path)
    _assert_video(output, _thresholds(case_name))
