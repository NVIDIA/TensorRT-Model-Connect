# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and timm reference proof for NFNet."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence


FAMILY = "timm_nfnet"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == "classification"
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selection(config) -> set[str]:
    selected = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option, default=[]) or []:
            selected.update(value.strip() for value in str(raw).split(",") if value.strip())
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return selected


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selection(metafunc.config)
    names = [
        name
        for name, (manifest, _) in CASES.items()
        if not selected or selected & {FAMILY, name, manifest["name"]}
    ]
    if not selected:
        names = [
            pytest.param(
                name,
                marks=pytest.mark.skip(reason="real NFNet E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected NFNet E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected NFNet E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_TIMM_TIMM_NFNET_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_TIMM_TIMM_NFNET_MODEL_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    )


def _asset(case: dict) -> Path:
    path = TEST_ROOT / str(case["test_image"])
    assert path.is_file(), f"selected NFNet E2E image is missing: {path}"
    record_evidence("inputs", {"asset": path})
    return path


# FP8 (quantization=fp8, FP8 dense convolutions on the fp16 graph) adds a cosine floor on
# the logits to the top-1 check. Measured on the FP8 case (B200 and B300, TensorRT 11.4), a
# correct FP8 engine reaches cosine 0.9953 to 0.9961 against the timm fp32 model. With every
# activation scale x0.3 the cosine drops to about 0.983 and fails this floor (top-1 unchanged);
# with every scale x0.1 the top-1 class changes. The floor leaves about 2x the measured FP8
# error (1 - cosine) as headroom for other GPUs.
_FP8_COSINE_FLOOR = 0.99


def _gate(manifest: dict) -> str:
    """The gate of a case: "fp8" for FP8 builds, else the build precision."""
    if manifest.get("quantization") == "fp8":
        return "fp8"
    return str(manifest["precision"])


def _cosine(left, right) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _assert_parity(actual_logits, expected_logits, gate: str = "fp16") -> None:
    assert int(np.argmax(actual_logits)) == int(np.argmax(expected_logits))
    if gate == "fp8":
        cosine = _cosine(actual_logits, expected_logits)
        record_evidence("fp8_logit_cosine", cosine)
        assert cosine >= _FP8_COSINE_FLOOR, f"{FAMILY} fp8 logit cosine {cosine}"


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence(
        "inputs",
        {"manifest": manifest, "case": case, "asset": TEST_ROOT / str(case.get("test_image", ""))},
    )
    gate = _gate(manifest)
    thresholds = {"top1_match": True}
    if gate == "fp8":
        thresholds["logit_cosine_min"] = _FP8_COSINE_FLOOR
    record_evidence("thresholds", thresholds)
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest["hf_id"],
            "hf_revision": manifest["hf_revision"],
        },
    )
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=int(manifest["max_sequence_length"]),
                tensor_parallel_size=int(manifest["tensor_parallel_size"]),
                quantization=manifest.get("quantization"),
            )
        )
    with evidence_stage("native"):
        completed = subprocess.run(
            [
                str(binary),
                "classify",
                str(bundle),
                "--runtime-root",
                str(runtime_root),
                "--image",
                str(_asset(case)),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        record_evidence(
            "native_process",
            {
                "argv": getattr(completed, "args", None),
                "stdout": getattr(completed, "stdout", None),
                "stderr": getattr(completed, "stderr", None),
            },
        )
        actual = json.loads(completed.stdout)
    record_evidence("native", actual)

    with evidence_stage("reference"):
        import timm
        import torch
        from PIL import Image
        from timm.data import create_transform, resolve_model_data_config

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        reference = timm.create_model(
            config["architecture"],
            pretrained=False,
            pretrained_cfg=config["pretrained_cfg"],
            num_classes=int(config["num_classes"]),
            checkpoint_path=str(model_dir / "model.safetensors"),
        )
        reference = reference.to("cuda").eval()
        transform = create_transform(**resolve_model_data_config(reference), is_training=False)
        pixels = transform(Image.open(_asset(case)).convert("RGB")).unsqueeze(0).to("cuda")
        with torch.no_grad():
            expected = reference(pixels).float().cpu().numpy()
    record_evidence("reference", {"top_class": int(np.argmax(expected)), "logits": expected})
    with evidence_stage("compare"):
        assert int(actual["top_class"]) == int(np.argmax(expected))
        if gate == "fp8":
            assert "logits" in actual, "the native classify output carries no logits"
            _assert_parity(actual["logits"], expected, gate)


def test_fp8_gate_checks_logit_cosine() -> None:
    """An fp8 case also needs the logit cosine; the same top-1 with a lower cosine fails."""
    reference = np.array([[4.0, 1.0, 0.5, 0.0]])
    close = [4.0, 1.02, 0.49, 0.01]
    far = [4.0, 3.5, -3.0, 2.5]
    _assert_parity(close, reference, "fp8")
    _assert_parity(far, reference, "fp16")
    assert _cosine(far, reference) < _FP8_COSINE_FLOOR
    with pytest.raises(AssertionError, match="fp8 logit cosine"):
        _assert_parity(far, reference, "fp8")
    assert _gate({"precision": "fp16", "quantization": "fp8"}) == "fp8"
    assert _gate({"precision": "fp16"}) == "fp16"
