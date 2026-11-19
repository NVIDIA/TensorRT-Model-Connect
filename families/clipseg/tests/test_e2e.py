# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned checkpoint parity through build, native DSO, and the image-feature API."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "clipseg"
TASKS = frozenset({"text_prompted_segmentation"})
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
    for fixture in ("case_name",):
        if fixture not in metafunc.fixturenames:
            continue
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
        metafunc.parametrize(fixture, parameters, ids=names)


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
            check=True,
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


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


def test_official_checkpoint_e2e(case_name, tmp_path):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
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
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, CLIPSegForImageSegmentation

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    processor = AutoProcessor.from_pretrained(directory, use_fast=False)
    model = CLIPSegForImageSegmentation.from_pretrained(directory, dtype=torch.float32).eval()
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    image_path = tmp_path / "native-input.png"
    image.save(image_path)
    record_evidence(
        "thresholds",
        {"probability_atol": 5e-4, "probability_rtol": 1e-4, "prompt_separation_min": 0.001},
    )
    masks = []
    for sample, prompt in enumerate(case["prompts"]):
        record_evidence("inputs", {"sample": sample, "prompt": prompt, "image": image_path})
        inputs = processor(
            text=[prompt], images=image, padding="max_length", max_length=77, return_tensors="pt"
        )
        with evidence_stage("reference"), torch.no_grad():
            expected = model(**inputs).logits.sigmoid()[0].numpy()
        actual = _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "segment-prompted",
            "--image",
            str(image_path),
            "--prompt",
            prompt,
        )
        with evidence_stage("comparison"):
            assert actual["num_masks"] == 1
        with evidence_stage("comparison"):
            assert [actual["height"], actual["width"]] == list(expected.shape)
        values = np.asarray(actual["masks"], dtype=np.float32).reshape(expected.shape)
        record_evidence("native", {"sample": sample, "prompt": prompt, "masks": values})
        record_evidence(
            "reference",
            {"mode": "hf_transformers", "sample": sample, "prompt": prompt, "masks": expected},
        )
        record_evidence(
            "metrics", {"sample": sample, "max_abs_error": float(np.max(np.abs(values - expected)))}
        )
        if evidence_enabled():
            views = []
            for role, mask in (("native", values), ("reference", expected)):
                output = tmp_path / f"{role}-mask-{sample}.png"
                Image.fromarray(np.rint(np.clip(mask, 0, 1) * 255).astype(np.uint8)).save(output)
                views.append(
                    {
                        "title": f"Sample {sample} / {role}",
                        "image": output,
                        "caption": f"{prompt}; shared probability scale 0–1",
                    }
                )
            record_evidence("views", views)
        with evidence_stage("comparison"):
            assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        with evidence_stage("comparison"):
            np.testing.assert_allclose(values, expected, atol=5e-4, rtol=1e-4)
        masks.append(values)
        print(f"CLIPSeg {prompt!r} probability maxabs={np.max(np.abs(values - expected)):.6g}")
    with evidence_stage("comparison"):
        assert np.mean(np.abs(masks[0] - masks[1])) > 0.001
