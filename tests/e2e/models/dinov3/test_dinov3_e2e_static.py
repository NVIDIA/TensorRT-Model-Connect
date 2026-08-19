# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static and comparator coverage for the model-owned DINOv3 E2E harness."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from tests.e2e.models.dinov3.e2e_plugins.comparator import ImageFeatureExtractionComparator
from tests.e2e.models.dinov3.e2e_plugins import runner as runner_module
from tests.e2e.models.dinov3.e2e_plugins.runner import Dinov3ReproCommandProvider
from tests.e2e_harness.contracts import RunContext, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_model_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_repro_command_provider,
    get_runner,
)
from tests.e2e_harness.registry import get_contract_plugin

_MODEL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODEL_DIR.parents[3]
_MODELS = {
    "dinov3-vits16-pretrain-lvd1689m": (
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        "114c1379950215c8b35dfcd4e90a5c251dde0d32",
        4,
    ),
    "dinov3-convnext-tiny-pretrain-lvd1689m": (
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "10d30274b4d445111e2d5bf75ac93bbd94db274b",
        0,
    ),
}
_PUBLIC_L0 = (
    "dinov3-vits16-timm-l0",
    "timm/vit_small_patch16_dinov3_qkvb.lvd1689m",
    "2c7705788ac282557562465d6443606664a55f05",
)
_OWNED_IMAGE_SHA256 = "d68cb42a55f79e51f71b78cf7d726f01c80a0e2dab8674da6f68361cce004cbc"
_THRESHOLDS = {
    "full_cosine": 0.999,
    "cls_cosine": 0.999,
    "pooler_cosine": 0.999,
    "register_cosine": 0.999,
    "mean_patch_cosine": 0.999,
    "p01_patch_cosine": 0.995,
    "relative_frobenius": 0.01,
    "shape_match": 1.0,
    "register_count_match": 1.0,
    "pooler_token_invariant": 1.0,
    "finite_tensors": 1.0,
}
_RELATIVE_FROBENIUS_BY_MODEL = {
    "dinov3-convnext-tiny-pretrain-lvd1689m": 0.015,
}


def _profile(*, relative_frobenius: float = 0.01) -> ThresholdProfile:
    metrics = dict(_THRESHOLDS)
    metrics["relative_frobenius"] = relative_frobenius
    return ThresholdProfile(
        task_strategy="image_feature_extraction",
        metrics=metrics,
    )


def _case(name: str):
    return load_model_manifest(_MODEL_DIR / "manifests" / f"{name}.json").build_case


def _tensor_payload(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return {"shape": list(contiguous.shape), "data": contiguous.reshape(-1).tolist()}


def _feature_output(hidden: np.ndarray, register_count: int) -> StageOutput:
    return StageOutput(
        stage_name="full_inference",
        data={
            "last_hidden_state": _tensor_payload(hidden),
            "pooler_output": _tensor_payload(hidden[:, 0, :]),
            "num_register_tokens": register_count,
        },
    )


def test_official_gated_manifests_pin_revisions_and_auth_preflight() -> None:
    for name, (hf_id, revision, register_count) in _MODELS.items():
        manifest_path = _MODEL_DIR / "manifests" / f"{name}.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = load_model_manifest(manifest_path)
        case = model.build_case

        assert raw["hf_id"] == hf_id
        assert raw["hf_revision"] == revision
        assert raw["testcases"][0]["gated"] is True
        assert raw["testcases"][0]["ci_tier"] == "nightly_only"
        assert raw["testcases"][0]["l0_replacement"] == _PUBLIC_L0[0]
        assert raw["testcases"][0]["num_register_tokens"] == register_count
        assert case.hf_id == hf_id
        assert case.hf_revision == revision
        assert case.runtime_strategy == "dinov3_image_feature_extraction"
        assert case.task_strategy == "image_feature_extraction"
        assert case.reference_backend == "hf_transformers"
        assert case.reference_family == "image_feature_extraction"
        assert case.user_contract == "representation_parity"
        assert case.metadata["num_register_tokens"] == register_count
        expected_thresholds = dict(_THRESHOLDS)
        expected_thresholds["relative_frobenius"] = _RELATIVE_FROBENIUS_BY_MODEL.get(
            name, _THRESHOLDS["relative_frobenius"]
        )
        assert case.threshold_overrides == expected_thresholds

        preflight_kinds = [requirement.kind for requirement in case.preflight]
        assert "binary_exists" in preflight_kinds
        assert "asset_exists" in preflight_kinds
        assert "hf_auth_token_present" in preflight_kinds
        auth = next(
            requirement
            for requirement in case.preflight
            if requirement.kind == "hf_auth_token_present"
        )
        assert auth.gating is True
        assert auth.args["hf_id"] == hf_id
        assert Path(case.inputs["image"]).is_file()


def test_public_timm_l0_is_secretless_full_scale_premerge_parity() -> None:
    name, hf_id, revision = _PUBLIC_L0
    manifest_path = _MODEL_DIR / "manifests" / f"{name}.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_model_manifest(manifest_path).build_case

    assert raw["hf_id"] == hf_id
    assert raw["hf_revision"] == revision
    assert raw["testcases"][0]["ci_tier"] == "l0_only"
    assert "gated" not in raw["testcases"][0]
    assert case.hf_id == hf_id
    assert case.hf_revision == revision
    assert case.metadata["checkpoint_layout"] == "timm_dinov3_vit"
    assert case.reference_backend == "timm_dinov3"
    assert case.metadata["num_register_tokens"] == 4
    assert case.threshold_overrides == _THRESHOLDS
    preflight_kinds = [req.kind for req in case.preflight]
    assert preflight_kinds.count("binary_exists") == 1
    assert preflight_kinds.count("asset_exists") == 1
    assert not any(req.kind == "hf_auth_token_present" for req in case.preflight)
    assert any(
        req.kind == "python_module_available" and req.args.get("module") == "timm"
        for req in case.preflight
    )
    assert Path(case.inputs["image"]).is_file()


def test_owned_image_is_exact() -> None:
    owned_image = _MODEL_DIR / "data" / "test_img.jpeg"
    assert hashlib.sha256(owned_image.read_bytes()).hexdigest() == _OWNED_IMAGE_SHA256


def test_model_owned_plugins_register_complete_feature_path() -> None:
    activate_model_plugins(_MODEL_DIR)
    assert get_runner("image_feature_extraction") is not None
    assert get_reference("hf_transformers") is not None
    assert get_reference("timm_dinov3") is not None
    assert get_comparator("image_feature_extraction") is not None
    assert get_repro_command_provider("dinov3") is not None
    assert get_contract_plugin("image_feature_extraction") is not None


def test_runtime_runner_invokes_extract_features_and_reads_full_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    case = _case("dinov3-vits16-pretrain-lvd1689m")
    hidden = np.arange(1, 1 + 6 * 4, dtype=np.float32).reshape(1, 6, 4)

    def fake_run(command, **kwargs):
        del kwargs
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "last_hidden_state": _tensor_payload(hidden),
                    "pooler_output": _tensor_payload(hidden[:, 0, :]),
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="saved\n", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    output = runner_module.ImageFeatureExtractionRunner().run_stage(
        case,
        StageSpec(name="full_inference", required=True),
        RunContext(
            case=case,
            binary_path="/opt/trtmc/bin/trtmc",
            engine_dir=str(tmp_path / "engines"),
            artifacts_dir=str(tmp_path / "artifacts"),
        ),
    )

    command = output.metadata["command"]
    assert command[1] == "extract-features"
    assert "--image" in command
    assert "--output-json" in command
    assert output.data["last_hidden_state"]["shape"] == [1, 6, 4]
    assert output.data["pooler_output"]["shape"] == [1, 4]
    assert output.data["num_register_tokens"] == 4


def test_repro_provider_owns_native_cli_command(tmp_path: Path) -> None:
    case = _case("dinov3-convnext-tiny-pretrain-lvd1689m")
    command = Dinov3ReproCommandProvider().build_trt_inference_command(
        case,
        RunContext(case=case, binary_path="trtmc", engine_dir=str(tmp_path)),
        str(tmp_path / case.bundle),
    )

    assert command is not None
    assert command[:2] == ["trtmc", "extract-features"]
    assert "--image" in command
    assert "--output-json" in command


def test_semantic_comparator_accepts_exact_full_cls_register_and_patch_features() -> None:
    register_count = 2
    hidden = np.arange(1, 1 + (1 + register_count + 4) * 8, dtype=np.float32).reshape(
        1, 1 + register_count + 4, 8
    )
    comparator = ImageFeatureExtractionComparator()
    result = comparator.compare(
        _feature_output(hidden, register_count),
        _feature_output(hidden.copy(), register_count),
        _profile(),
        StageSpec(name="full_inference", required=True),
    )

    assert result.passed
    assert result.metrics["shape_match"].passed
    assert result.metrics["pooler_token_invariant"].passed
    assert np.isclose(result.metrics["register_cosine"].value, 1.0)
    assert np.isclose(result.metrics["mean_patch_cosine"].value, 1.0)
    assert np.isclose(result.metrics["p01_patch_cosine"].value, 1.0)
    assert result.metrics["relative_frobenius"].value == 0.0


def test_convnext_relative_frobenius_override_remains_bounded() -> None:
    hidden = np.ones((1, 5, 8), dtype=np.float32)
    comparator = ImageFeatureExtractionComparator()
    stage = StageSpec(name="full_inference", required=True)

    default_result = comparator.compare(
        _feature_output(hidden * 1.0149, 0),
        _feature_output(hidden, 0),
        _profile(),
        stage,
    )
    calibrated_result = comparator.compare(
        _feature_output(hidden * 1.0149, 0),
        _feature_output(hidden, 0),
        _profile(relative_frobenius=0.015),
        stage,
    )
    beyond_result = comparator.compare(
        _feature_output(hidden * 1.0151, 0),
        _feature_output(hidden, 0),
        _profile(relative_frobenius=0.015),
        stage,
    )

    assert not default_result.metrics["relative_frobenius"].passed
    assert calibrated_result.passed
    assert "relative Frobenius <= 0.015" in calibrated_result.composite_rule
    assert not beyond_result.metrics["relative_frobenius"].passed


def test_semantic_comparator_rejects_bad_patch_tail_without_weakening_thresholds() -> None:
    hidden = np.arange(1, 1 + 5 * 8, dtype=np.float32).reshape(1, 5, 8)
    corrupted = hidden.copy()
    corrupted[:, -1, :] *= -1.0
    comparator = ImageFeatureExtractionComparator()
    result = comparator.compare(
        _feature_output(corrupted, 0),
        _feature_output(hidden, 0),
        _profile(),
        StageSpec(name="full_inference", required=True),
    )

    assert not result.passed
    assert not result.metrics["p01_patch_cosine"].passed
    assert result.metrics["p01_patch_cosine"].threshold == 0.995


def test_semantic_comparator_rejects_pooler_token_invariant_break() -> None:
    hidden = np.arange(1, 1 + 4 * 8, dtype=np.float32).reshape(1, 4, 8)
    trt = _feature_output(hidden, 0)
    trt.data["pooler_output"] = _tensor_payload(hidden[:, 0, :] + 1.0)
    comparator = ImageFeatureExtractionComparator()
    result = comparator.compare(
        trt,
        _feature_output(hidden, 0),
        _profile(),
        StageSpec(name="full_inference", required=True),
    )

    assert not result.passed
    assert not result.metrics["pooler_token_invariant"].passed


def test_semantic_comparator_rejects_any_shape_mismatch() -> None:
    trt_hidden = np.arange(1, 1 + 4 * 8, dtype=np.float32).reshape(1, 4, 8)
    ref_hidden = np.arange(1, 1 + 5 * 8, dtype=np.float32).reshape(1, 5, 8)
    comparator = ImageFeatureExtractionComparator()
    result = comparator.compare(
        _feature_output(trt_hidden, 0),
        _feature_output(ref_hidden, 0),
        _profile(),
        StageSpec(name="full_inference", required=True),
    )

    assert not result.passed
    assert not result.metrics["shape_match"].passed
    assert result.metrics["shape_match"].threshold == 1.0


def test_runtime_strategy_matrix_routes_native_extract_features() -> None:
    matrix = json.loads(
        (_REPO_ROOT / "tests" / "runtime_strategy_matrix.yaml").read_text(encoding="utf-8")
    )
    assert "dinov3_image_feature_extraction" in matrix["new_runtime_guard_strategies"]
    entry = matrix["runtime_strategies"]["dinov3_image_feature_extraction"]
    assert entry["task_strategy"] == "image_feature_extraction"
    assert entry["cli_commands"] == ["extract-features"]
    assert entry["runner_class"].endswith("ImageFeatureExtractionRunner")
    assert entry["comparator_class"].endswith("ImageFeatureExtractionComparator")

    runner_text = (_MODEL_DIR / "e2e_plugins" / "runner.py").read_text(encoding="utf-8")
    reference_text = (_MODEL_DIR / "e2e_plugins" / "reference.py").read_text(encoding="utf-8")
    assert '"extract-features"' in runner_text
    assert '"--output-json"' in runner_text
    assert "AutoImageProcessor" in reference_text
    assert "AutoModel" in reference_text
    assert "timm.create_model" in reference_text
    assert '_TIMM_REFERENCE_VERSION = "1.0.28"' in reference_text
    assert "timm.__version__" in reference_text
    assert "model.forward_features" in reference_text
    assert "pooled = hidden[:, 0, :]" in reference_text
    assert "case.hf_revision" in reference_text
