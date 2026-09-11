# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native runtime, and timm reference proof for Xception."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build


FAMILY = "timm_xception"
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
                marks=pytest.mark.skip(reason="real Xception E2E requires explicit selection"),
                id=name,
            )
            for name in names
        ]
    metafunc.parametrize("case_name", names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected Xception E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected Xception E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get("TRTMC_TIMM_XCEPTION_MODEL_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_TIMM_XCEPTION_MODEL_DIR")
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
    assert path.is_file(), f"selected Xception E2E image is missing: {path}"
    record_evidence("inputs", {"asset": path})
    return path


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": CASES[case_name][-1]})
    record_evidence("thresholds", {"top1_match": True})
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    with evidence_stage("setup"):
        assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    with evidence_stage("setup"):
        assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"), "hf_revision": manifest.get("hf_revision")})
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
            {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr},
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
            record_evidence(
                "reference", {"top_class": int(np.argmax(expected)), "logits": expected}
            )
    with evidence_stage("compare"):
        assert int(actual["top_class"]) == int(np.argmax(expected))
