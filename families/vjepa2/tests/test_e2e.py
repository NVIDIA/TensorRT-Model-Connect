# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned V-JEPA2 native video preprocessing, patch features and frame-span parity."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "vjepa2"
TASKS = frozenset({"video_to_token_features"})
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
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as vision
    from torchvision.transforms import InterpolationMode
    from transformers import AutoVideoProcessor, VJEPA2Model
    import sys
    import transformers

    record_evidence(
        "reference_environment",
        {
            "requested": manifest.get("reference_environment", "default"),
            "python": sys.executable,
            "transformers": transformers.__version__,
        },
    )
    assert transformers.__version__ == "5.20.0", (
        "V-JEPA2 parity requires the original pinned Transformers5.20.0 reference"
    )

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
    processor = AutoVideoProcessor.from_pretrained(directory)
    image = Image.open(TEST_ROOT / "data" / case["image"]).convert("RGB")
    record_evidence(
        "inputs",
        {
            "image": TEST_ROOT / "data" / case["image"],
            "video_recipe": "64 frames; frame i rolls the image right by 2*i pixels; then forward/reversed order",
        },
    )
    geometry = None
    for index, sample in enumerate(
        [image, image.transpose(Image.Transpose.ROTATE_90), image.resize((302, 300))]
    ):
        frame = np.asarray(sample, dtype=np.float32) / 255
        source = tmp_path / f"frame-{index}.f32"
        target = tmp_path / f"pixels-{index}.f32"
        source.write_bytes(frame.astype("<f4").tobytes())
        completed = subprocess.run(
            [str(_native_test_binary("test_vjepa2_preprocess"))],
            input=json.dumps(
                dict(
                    source=str(source), output=str(target), height=sample.height, width=sample.width
                )
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        native_geometry = json.loads(completed.stdout)
        inputs = processor(
            videos=[frame[None, ...] * 255],
            input_data_format="channels_last",
            do_sample_frames=False,
            return_tensors="pt",
        )
        expected = inputs.pixel_values_videos[0, 0].numpy()
        actual = np.fromfile(target, dtype="<f4").reshape(3, 256, 256)
        record_evidence(
            "preprocessing",
            {
                "sample": index,
                "native_pixels": actual,
                "reference_pixels": expected,
                "native_geometry": native_geometry,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
            },
        )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        resized = vision.resize(
            torch.from_numpy(frame * 255).permute(2, 0, 1),
            [292],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        rh, rw = resized.shape[-2:]
        assert native_geometry == dict(
            height=rh, width=rw, top=round((rh - 256) / 2), left=round((rw - 256) / 2)
        )
        if index == 0:
            geometry = native_geometry
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
        VJEPA2Model.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager")
        .eval()
        .to("cuda")
    )
    base = np.asarray(image, dtype=np.float32) / 255
    clip = np.stack([np.roll(base, 2 * index, axis=1) for index in range(64)])
    comparison_failures = []
    for label, video in [("forward", clip), ("reversed", clip[::-1])]:
        video = np.ascontiguousarray(video)
        source = tmp_path / f"video-{label}.f32"
        target = tmp_path / f"features-{label}.f32"
        source.write_bytes(video.astype("<f4").tobytes())
        inputs = processor(
            videos=[video * 255],
            input_data_format="channels_last",
            do_sample_frames=False,
            return_tensors="pt",
        ).to("cuda")
        assert inputs.pixel_values_videos.shape == (1, 64, 3, 256, 256)
        with torch.no_grad():
            expected = reference(**inputs, skip_predictor=True).last_hidden_state[0].cpu().numpy()
        completed = subprocess.run(
            [
                str(_native_test_binary("test_vjepa2_runtime")),
                str(bundle),
                str(runtime_root),
                str(source),
                str(target),
                "64",
                str(image.height),
                str(image.width),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        metadata = json.loads(completed.stdout.splitlines()[-1])
        assert metadata["rows"] == 8192 and metadata["columns"] == 1024
        assert metadata["grid"] == [32, 16, 16] and metadata["source_frames"] == 64
        assert metadata["first"]["role"] == metadata["last"]["role"] == 1
        assert metadata["first"]["grid"] == [0, 0, 0] and metadata["last"]["grid"] == [31, 15, 15]
        assert metadata["first"]["span"] == [0, 2] and metadata["last"]["span"] == [62, 64]
        top, left, rh, rw = [geometry[key] for key in ("top", "left", "height", "width")]
        np.testing.assert_allclose(
            metadata["first"]["box"],
            [left / rw, top / rh, (left + 16) / rw, (top + 16) / rh],
            atol=1e-7,
            rtol=0,
        )
        np.testing.assert_allclose(
            metadata["last"]["box"],
            [(left + 240) / rw, (top + 240) / rh, (left + 256) / rw, (top + 256) / rh],
            atol=1e-7,
            rtol=0,
        )
        tokens_path = Path(metadata["tokens_path"])
        tokens = json.loads(tokens_path.read_text())
        # Record this sidecar once. Retain its name in the output summary without
        # asking the recorder to copy the same file again through metadata.
        metadata["tokens_path"] = tokens_path.name
        record_evidence(
            "token_geometry", {"sample": label, "all_tokens": tokens_path, "expected_rows": 8192}
        )
        assert len(tokens) == 8192
        cells = np.arange(8192)
        expected_grid = np.column_stack([cells // 256, cells // 16 % 16, cells % 16])
        expected_span = np.column_stack([expected_grid[:, 0] * 2, (expected_grid[:, 0] + 1) * 2])
        assert all(token["role"] == 1 for token in tokens)
        assert np.array_equal(np.asarray([token["grid"] for token in tokens]), expected_grid)
        assert np.array_equal(np.asarray([token["span"] for token in tokens]), expected_span)
        expected_boxes = np.column_stack(
            [
                (left + expected_grid[:, 2] * 16) / rw,
                (top + expected_grid[:, 1] * 16) / rh,
                (left + (expected_grid[:, 2] + 1) * 16) / rw,
                (top + (expected_grid[:, 1] + 1) * 16) / rh,
            ]
        )
        np.testing.assert_allclose(
            np.asarray([token["box"] for token in tokens]), expected_boxes, atol=1e-7, rtol=0
        )
        actual = np.fromfile(target, dtype="<f4").reshape(8192, 1024)
        if evidence_enabled():
            pair = {}
            for role, values in (("native", actual), ("reference", expected)):
                archive = tmp_path / f"{role}-{label}-features.npz"
                np.savez_compressed(archive, features=values)
                pair[role] = archive
            # Preserve both complete matrices losslessly within the recorder's
            # file bound; the inline numeric view explicitly contains a prefix.
            record_evidence(
                "native",
                {
                    "sample": label,
                    "metadata": metadata,
                    "features": actual[:16],
                    "features_scope": "First 16 rows; complete matrix in reference_comparison",
                    "full_features_file": pair["native"].name,
                    "full_shape": list(actual.shape),
                },
            )
            record_evidence(
                "reference",
                {
                    "mode": "hf_transformers",
                    "sample": label,
                    "features": expected[:16],
                    "features_scope": "First 16 rows; complete matrix in reference_comparison",
                    "full_features_file": pair["reference"].name,
                    "full_shape": list(expected.shape),
                },
            )
        record_evidence(
            "native_process",
            {
                "command": shlex.join(completed.args),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        assert expected.shape == actual.shape and np.isfinite(actual).all()
        absolute_passed = True
        try:
            np.testing.assert_allclose(actual, expected, atol=case["max_abs_error"], rtol=0)
        except AssertionError as error:
            absolute_passed = False
            comparison_failures.append(f"{label} full-feature absolute error: {error}")
        cosine = np.sum(actual * expected, axis=1) / (
            np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
        )
        record_evidence(
            "metrics",
            {
                "sample": label,
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "min_cosine": float(cosine.min()),
            },
        )
        cosine_passed = True
        try:
            assert float(cosine.min()) >= case["min_cosine"]
        except AssertionError as error:
            cosine_passed = False
            comparison_failures.append(f"{label} full-feature cosine: {error}")
        if evidence_enabled():
            record_evidence(
                "reference_comparison",
                {
                    "label": f"V-JEPA2 {label} full clip features",
                    "scope": "independent_reference",
                    "enforced": True,
                    **pair,
                    "checks": [
                        {
                            "name": "max_abs_error",
                            "scope": "independent_reference",
                            "actual": float(np.max(np.abs(actual - expected))),
                            "operator": "<=",
                            "expected": case["max_abs_error"],
                            "passed": absolute_passed,
                        },
                        {
                            "name": "min_cosine",
                            "scope": "independent_reference",
                            "actual": float(cosine.min()),
                            "operator": ">=",
                            "expected": case["min_cosine"],
                            "passed": cosine_passed,
                        },
                    ],
                },
            )
        print(
            f"V-JEPA2 {label}: max_abs={np.max(np.abs(actual - expected)):.7g}, min_cosine={cosine.min():.8f}"
        )

    if comparison_failures:
        raise AssertionError("\n\n".join(comparison_failures))
