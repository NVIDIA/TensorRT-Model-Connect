# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot GR00T N1.7 build -> bundle -> native Task API, against the pinned LeRobot reference.

Every case builds the pinned ``nvidia/gr00t17-lerobot-libero_spatial-640`` checkpoint straight
from its directory, runs a recorded LIBERO-spatial observation through the bundle with the
reference's flow noise, and compares all 16 x 7 action values with the LeRobot float32 oracle
(and, for a bfloat16 bundle, with LeRobot's own bfloat16 policy).
"""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from ..model import crop_size
from ..tokenizer import TOKENIZER_FILE, TOKENIZER_REPO, TOKENIZER_REVISION, load_tokenizer_json


FAMILY = "lerobot_groot"
TASK = "images_state_text_to_action_chunk"
CHUNK, ACTION_DIM = 16, 7
CAMERAS = ("observation.images.image", "observation.images.wrist_image")
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"
MODEL_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
# CI stages the manifests' external_files (the public tokenizer copy) outside the checkpoint:
# Internal CI in the directory this variable names, the Community GPU image under /opt/trtmc-ci.
EXTERNAL_FILES_ENV = "TRTMC_LEROBOT_GROOT_MODEL_DIR"
CI_ASSET_DIR = Path("/opt/trtmc-ci")


def _cases() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY and manifest["task"] == TASK
        assert manifest["tensor_parallel_size"] == 1
        for case in manifest["testcases"]:
            assert case["name"] not in result
            result[case["name"]] = (manifest, case)
    assert result
    return result


CASES = _cases()


def _selected(config) -> set[str]:
    values = set()
    for option in ("--e2e-model", "--e2e-testcase"):
        for raw in config.getoption(option) or []:
            values.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        values.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return values


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    selected = _selected(metafunc.config)
    names = sorted(
        name
        for name, (manifest, _) in CASES.items()
        if not selected or {FAMILY, manifest["name"], name}.intersection(selected)
    )
    parameters = (
        names
        if selected
        else [
            pytest.param(name, marks=pytest.mark.skip(reason="real family E2E was not selected"))
            for name in names
        ]
    )
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, name: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {name}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {name} does not exist: {path}"
    return path


def _checkpoint_dir(manifest: dict) -> Path:
    """The checkpoint directory: explicit, or the snapshot CI staged before this offline run.

    The override is not TRTMC_LEROBOT_GROOT_MODEL_DIR, which CI points at the external files.
    """
    explicit = os.environ.get("TRTMC_LEROBOT_GROOT_CHECKPOINT_DIR")
    if explicit:
        return _required_path(explicit, "TRTMC_LEROBOT_GROOT_CHECKPOINT_DIR")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=manifest["hf_id"],
            revision=manifest["hf_revision"],
            local_files_only=True,
        )
    )


def _ci_tokenizer(manifest: dict) -> Path | None:
    (asset,) = manifest["external_files"]
    assert asset["path"] == TOKENIZER_FILE
    for root in (os.environ.get(EXTERNAL_FILES_ENV), CI_ASSET_DIR):
        if root and (Path(root) / asset["path"]).is_file():
            return Path(root) / asset["path"]
    return None


@pytest.fixture(scope="module")
def model_dirs(tmp_path_factory):
    """Overlay CI's public tokenizer on the unchanged staged checkpoint directory."""
    resolved: dict[str, Path] = {}

    def get(manifest: dict) -> Path:
        name = manifest["name"]
        if name not in resolved:
            directory = _checkpoint_dir(manifest)
            ci_tokenizer = _ci_tokenizer(manifest)
            if ci_tokenizer is not None and not (directory / TOKENIZER_FILE).is_file():
                overlay = tmp_path_factory.mktemp(f"{name}-with-tokenizer")
                for path in directory.iterdir():
                    (overlay / path.name).symlink_to(path)
                (overlay / TOKENIZER_FILE).symlink_to(ci_tokenizer)
                directory = overlay
            resolved[name] = directory
        return resolved[name]

    return get


def _staged_state_files(directory: Path) -> list[str]:
    names = []
    for processor in ("policy_preprocessor.json", "policy_postprocessor.json"):
        steps = json.loads((directory / processor).read_text(encoding="utf-8")).get("steps", [])
        names.extend(step["state_file"] for step in steps if step.get("state_file"))
    return names


def test_pinned_assets_are_staged(case_name: str, model_dirs) -> None:
    """Fail before any build, with the asset named, when the build would not find an input.

    Pytest runs this first. CI stages the manifest's checkpoint revision and the public
    tokenizer copy, then runs offline. The checkpoint, every statistics file its processors
    name, and the tokenizer must be local.
    """
    manifest, _ = CASES[case_name]
    try:
        directory = model_dirs(manifest)
    except Exception as error:  # noqa: BLE001 - surface why the staged snapshot is unusable
        pytest.fail(f"{manifest['hf_id']}@{manifest['hf_revision']} is not staged locally: {error}")
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    missing += [name for name in _staged_state_files(directory) if not (directory / name).is_file()]
    assert not missing, f"staged checkpoint is missing {missing}"
    if (directory / TOKENIZER_FILE).is_file():
        return
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(TOKENIZER_REPO, TOKENIZER_FILE, revision=TOKENIZER_REVISION, local_files_only=True)
    except Exception as error:  # noqa: BLE001 - name the dependency that is not staged
        pytest.fail(f"{TOKENIZER_FILE} is in neither the checkpoint, ${EXTERNAL_FILES_ENV} nor {CI_ASSET_DIR}, "
                    f"and {TOKENIZER_REPO}@{TOKENIZER_REVISION} is not staged locally: {error}")


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, model_dirs):
    """Build each manifest's bundle once per module: it takes minutes and many cases share it."""
    built: dict[str, Path] = {}

    def get(manifest: dict) -> Path:
        if manifest["name"] not in built:
            output = tmp_path_factory.mktemp(manifest["name"]) / manifest["bundle"]
            with evidence_stage("build"):
                build(
                    BuildRequest(
                        model_dir=model_dirs(manifest),
                        output_path=output,
                        family=FAMILY,
                        task=manifest["task"],
                        precision=manifest["precision"],
                    )
                )
            assert output.is_file()
            built[manifest["name"]] = output
        return built[manifest["name"]]

    return get


@cache
def _native_binaries() -> tuple[Path, Path]:
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "8", "--target",
         "trtmc_lerobot_groot_qualification", "trtmc_lerobot_groot_preprocess_check",
         "trtmc_backend_trt", "trtmc_model_lerobot_groot"],
        check=True,
        timeout=900,
    )
    binary = build_dir / "families" / FAMILY / "trtmc_lerobot_groot_qualification"
    assert binary.is_file(), f"selected {FAMILY} qualification binary is missing: {binary}"
    return binary, build_dir


def _asset(case: dict, name: str) -> Path:
    path = TEST_ROOT / case["inputs"][name]
    assert path.exists(), path
    record_evidence("inputs", {name: path})
    return path


def _reference(case: dict, name: str) -> np.ndarray:
    return np.fromfile(_asset(case, name), dtype="<f4").reshape(CHUNK, ACTION_DIM)


def _metrics(got: np.ndarray, want: np.ndarray) -> dict[str, float]:
    # float64, so the cosine of a float32 bundle is not float32 rounding noise.
    got, want = got.astype(np.float64), want.astype(np.float64)
    diff = np.abs(got - want)
    cosine = float(got.ravel() @ want.ravel() / (np.linalg.norm(got) * np.linalg.norm(want)))
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "cosine": cosine}


def _run_native(binary: Path, root: Path, bundle: Path, case: dict, tmp_path: Path):
    frame = _asset(case, "frame")
    observation = json.loads((frame / "observation.json").read_text(encoding="utf-8"))
    state = tmp_path / "state.f32"
    np.asarray(observation["state"], dtype="<f4").tofile(state)
    output = tmp_path / "actions.f32"
    command = [
        str(binary), "--bundle", str(bundle), "--runtime-root", str(root),
        "--camera", f"{CAMERAS[0]}={frame / 'image.png'}",
        "--camera", f"{CAMERAS[1]}={frame / 'wrist_image.png'}",
        "--state", str(state), "--noise", str(_asset(case, "noise")),
        "--text", observation["task"], "--repeat", "20", "--check-rejections",
        "--output", str(output),
    ]
    alternate = None
    if "alternate_task" in case:
        alternate = tmp_path / "alternate.f32"
        command += ["--alternate-text", case["alternate_task"], "--alternate-output", str(alternate)]
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(part for part in (str(root), environment.get("LD_LIBRARY_PATH", "")) if part)
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=environment, timeout=1800)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    record_evidence("native_process", {"argv": completed.args, "stdout": completed.stdout, "stderr": completed.stderr})
    summary = json.loads(completed.stdout)
    read = lambda path: np.fromfile(path, dtype="<f4").reshape(CHUNK, ACTION_DIM)  # noqa: E731
    return read(output), (read(alternate) if alternate else None), summary


# Messages are part of the contract: a caller has to learn which input was wrong.
EXPECTED_REJECTIONS = {
    "unknown_camera": "does not accept a camera",
    "missing_camera": "requires camera",
    "short_state": "state must have 8 values",
    "non_ascii_text": "accepts ASCII instructions only",
    "long_text": "instruction exceeds 256 tokens",
    "huge_text": "instruction exceeds 4096 bytes",
    "wrong_image_size": "must be 256x256x3",
}


def _check_limits(limits: dict, case: dict, actions: np.ndarray, prefix: str) -> dict:
    vs_fp32 = _metrics(actions, _reference(case, f"{prefix}fp32"))
    vs_bf16 = _metrics(actions, _reference(case, f"{prefix}bf16"))
    assert vs_fp32["max_abs"] <= limits["max_abs_vs_fp32"], vs_fp32
    assert vs_fp32["mean_abs"] <= limits["mean_abs_vs_fp32"], vs_fp32
    assert vs_fp32["cosine"] >= limits["cosine_vs_fp32_min"], vs_fp32
    if "max_abs_vs_bf16" in limits:
        assert vs_bf16["max_abs"] <= limits["max_abs_vs_bf16"], vs_bf16
    return {"vs_fp32": vs_fp32, "vs_bf16": vs_bf16}


def test_groot_bundle_matches_the_lerobot_reference_e2e(case_name: str, bundles, tmp_path) -> None:
    manifest, case = CASES[case_name]
    limits = json.loads((THRESHOLD_ROOT / f"{case_name}.json").read_text(encoding="utf-8"))["threshold_overrides"]
    bundle = bundles(manifest)
    with evidence_stage("native_build"):
        binary, root = _native_binaries()
    with evidence_stage("native"):
        actions, alternate, summary = _run_native(binary, root, bundle, case, tmp_path)

    assert summary["rows"] == CHUNK and summary["columns"] == ACTION_DIM
    assert np.isfinite(actions).all()
    p95 = float(np.percentile(summary["wall_ms"], 95))
    metrics = {"primary": _check_limits(limits, case, actions, "reference_")}
    if alternate is not None:
        # Same images and state, another instruction: the bundle follows the text it is given and
        # keeps nothing between calls, so the first instruction's chunk comes back unchanged.
        assert summary["stateless"] is True
        metrics["alternate"] = _check_limits(limits, case, alternate, "reference_alternate_")
        followed = _metrics(alternate, _reference(case, "reference_fp32"))["max_abs"]
        assert metrics["alternate"]["vs_fp32"]["max_abs"] < followed, (metrics["alternate"], followed)
    record_evidence("metrics", {**metrics, "chunk_inference_p95_ms": p95, "bundle_load_ms": summary["load_ms"],
                                "bundle_bytes": bundle.stat().st_size, "precision": manifest["precision"],
                                "within_training_bounds": summary["within_training_bounds"]})
    assert p95 <= limits["chunk_inference_p95_ms"], p95

    rejections = {item["case"]: item for item in summary["rejections"]}
    assert set(rejections) == set(EXPECTED_REJECTIONS)
    for name, needle in EXPECTED_REJECTIONS.items():
        assert rejections[name]["code"] != "ok", f"{name} was accepted"
        assert needle in rejections[name]["message"], rejections[name]


def test_native_runtime_has_no_python_or_pytorch_dependency(case_name: str) -> None:
    # Parametrized only so it is skipped unless the real E2E is selected; the result is case-independent.
    binary, root = _native_binaries()
    for library in (binary, root / "libtrtmc_model_lerobot_groot.so", root / "libtrtmc_backend_trt.so"):
        assert library.is_file(), library
        listing = subprocess.run(
            ["ldd", str(library)], check=True, capture_output=True, text=True,
            env={**os.environ, "LD_LIBRARY_PATH": str(root)},
        ).stdout.lower()
        record_evidence("linked_libraries", {library.name: listing})
        assert "python" not in listing and "torch" not in listing, f"{library.name}:\n{listing}"


def test_native_preprocessing_matches_lerobot(case_name: str, model_dirs, tmp_path) -> None:
    """The runtime's prompt ids and camera pixels equal LeRobot's, bit for bit.

    The golden prompts stress the instruction formalisation and the tokenizer; the processed
    images are each recorded camera after LeRobot's evaluation crop and INTER_AREA resize.
    """
    manifest, _ = CASES[case_name]
    binary, root = _native_binaries()
    tokenizer = tmp_path / TOKENIZER_FILE
    tokenizer.write_bytes(load_tokenizer_json(model_dirs(manifest)))
    from ..config import load_config

    cfg = load_config(model_dirs(manifest))
    command = [str(binary.parent / "trtmc_lerobot_groot_preprocess_check"), "--tokenizer", str(tokenizer),
               "--golden", str(TEST_ROOT / "data" / "tokenizer_golden.json"),
               "--size", str(cfg.camera_height), "--crop", str(crop_size(cfg))]
    for frame in sorted((TEST_ROOT / "data" / "recorded_observation").iterdir()):
        for camera in ("image", "wrist_image"):
            command += ["--image", f"{frame / (camera + '.png')}={frame / (camera + '.processed.png')}"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True,
                               env={**os.environ, "LD_LIBRARY_PATH": str(root)}, timeout=600)
    record_evidence("preprocess_check", {"stdout": completed.stdout, "stderr": completed.stderr})
    assert completed.returncode == 0, completed.stdout or completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["ok"] and summary["tokenizer_mismatches"] == 0 and summary["images"]
