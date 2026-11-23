# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned BLIP-2 native image/question generation and processor parity."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "blip2"
TASKS = frozenset({"images_text_to_text"})
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


def _checkpoint_source(manifest: dict, path: Path, *, override: bool) -> Path:
    """An arbitrary local directory does not establish the manifest's pinned identity."""
    record_evidence(
        "checkpoint",
        {
            "repo_id": None if override else manifest["hf_id"],
            "revision": None if override else manifest["hf_revision"],
            "source": "local_override" if override else "pinned_snapshot",
            "path": str(path.resolve()),
            "revision_verified": not override,
            "expected_repo_id": manifest["hf_id"],
            "expected_revision": manifest["hf_revision"],
            "precision": manifest["precision"],
        },
    )
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _checkpoint_source(
            manifest, _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR"), override=True
        )
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return _checkpoint_source(manifest, Path(snapshot), override=False)


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


@pytest.fixture
def strict_reference_math():
    import torch

    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def test_official_checkpoint_e2e(case_name, tmp_path, strict_reference_math):
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
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    binary, runtime_root = _runtime()
    with evidence_stage("reference"):
        processor = AutoProcessor.from_pretrained(directory, use_fast=False)
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(processor.tokenizer.backend_tokenizer.to_str())
    image = Image.open(TEST_ROOT / "data" / case["image"]).convert("RGB")
    path = tmp_path / "image.png"
    image.save(path)
    source = tmp_path / "source.f32"
    source.write_bytes((np.asarray(image, dtype=np.float32) / 255).astype("<f4").tobytes())
    batches = []
    for question in case["questions"]:
        inputs = processor(images=image, text=question, return_tensors="pt")
        batches.append(inputs)
        pixels = tmp_path / "pixels.f32"
        completed = subprocess.run(
            [str(_native_test_binary("test_blip2_preprocess"))],
            input=json.dumps(
                dict(
                    source=str(source),
                    output=str(pixels),
                    height=image.height,
                    width=image.width,
                    tokenizer=str(tokenizer_json),
                    image_id=50265,
                    question=question,
                )
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        record_evidence(
            "preprocess_process",
            {
                "stdout": completed.stdout[-4096:],
                "stderr": completed.stderr[-4096:],
                "returncode": completed.returncode,
            },
        )
        with evidence_stage("comparison"):
            assert completed.returncode == 0, completed.stderr
        record_evidence(
            "native_preprocess",
            {
                "question": question,
                "token_ids": json.loads(completed.stdout)["ids"],
                "pixels": np.fromfile(pixels, dtype="<f4").reshape(1, 3, 224, 224),
            },
        )
        record_evidence(
            "reference_preprocess",
            {
                "question": question,
                "token_ids": inputs.input_ids[0].tolist(),
                "pixels": inputs.pixel_values.numpy(),
            },
        )
        with evidence_stage("comparison"):
            assert json.loads(completed.stdout)["ids"] == inputs.input_ids[0].tolist()
        with evidence_stage("comparison"):
            np.testing.assert_allclose(
                np.fromfile(pixels, dtype="<f4").reshape(1, 3, 224, 224),
                inputs.pixel_values.numpy(),
                atol=3e-7,
                rtol=0,
            )
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    with evidence_stage("reference"):
        reference = (
            AutoModelForImageTextToText.from_pretrained(
                directory, dtype=torch.float32, attn_implementation="eager"
            )
            .eval()
            .to("cuda")
        )
    for question, inputs in zip(case["questions"], batches):
        inputs = inputs.to("cuda")
        with evidence_stage("reference"), torch.no_grad():
            generated = reference.generate(
                **inputs, do_sample=False, max_new_tokens=case["max_new_tokens"]
            )
        expected_ids = generated[0, inputs.input_ids.shape[1] :].tolist()
        expected_text = processor.batch_decode([expected_ids], skip_special_tokens=True)[0]
        payload = _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "run",
            "--task",
            manifest["task"],
            "--image",
            str(path),
            "--prompt",
            question,
            "--max-new-tokens",
            str(case["max_new_tokens"]),
        )
        record_evidence("inputs", {"question": question, "image": path})
        record_evidence("thresholds", {"token_ids": "exact equality", "text": "exact equality"})
        record_evidence("native", payload)
        record_evidence(
            "reference",
            {"mode": "hf_transformers", "token_ids": expected_ids, "text": expected_text},
        )
        with evidence_stage("comparison"):
            assert payload["token_ids"] == expected_ids
        with evidence_stage("comparison"):
            assert payload["text"] == expected_text
        with evidence_stage("comparison"):
            assert expected_text.strip()
        token_mismatch = int(payload["token_ids"] != expected_ids)
        text_mismatch = int(payload["text"] != expected_text)
        _record_comparison(
            tmp_path,
            f"generation-{len(expected_ids)}-{len(expected_text)}",
            payload,
            {"token_ids": expected_ids, "text": expected_text},
            [
                {
                    "name": "token_mismatch",
                    "actual": token_mismatch,
                    "operator": "==",
                    "expected": 0,
                    "passed": token_mismatch == 0,
                },
                {
                    "name": "text_mismatch",
                    "actual": text_mismatch,
                    "operator": "==",
                    "expected": 0,
                    "passed": text_mismatch == 0,
                },
            ],
        )
        print(f"BLIP-2 {question!r}: exact tokens={len(expected_ids)}, text={expected_text!r}")
