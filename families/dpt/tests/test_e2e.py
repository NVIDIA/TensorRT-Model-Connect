# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for DPT.

The reference is the checkpoint's own Transformers pipeline in float32 on the
CPU: ``DPTImageProcessor`` (the PIL processor the model card uses),
``DPTForDepthEstimation`` and ``post_process_depth_estimation`` onto the
input size. The test image is written as PNG first so that the native CLI and
the reference decode the same 8-bit pixels.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, mark_loaded, record_evidence

FAMILY = "dpt"
TASKS = frozenset({"monocular_depth"})
PRECISIONS = frozenset({"fp32", "fp16"})
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
MODEL_DIR_OVERRIDE = f"TRTMC_{FAMILY.upper()}_MODEL_DIR"
THRESHOLDS = ("min_depth_correlation", "max_depth_relative_error",
              "max_mean_depth_relative_error")
# The one item of a testcase-only selection that names no case of this family; it
# checks, once the session is collected, that another collected family owns them.
ELSEWHERE = "selected-elsewhere"


def _require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise ValueError(f"{path.name}: {message}")


def _check_manifest(path: Path, manifest: dict) -> None:
    """Every field the test reads, with its JSON type and range."""
    for key in ("name", "hf_id", "hf_revision", "bundle"):
        _require(isinstance(manifest.get(key), str) and bool(manifest[key]), path,
                 f"{key} must be a non-empty string")
    _require(manifest.get("family") == FAMILY, path, f"family must be {FAMILY!r}")
    _require(manifest.get("task") in TASKS, path, f"task must be one of {sorted(TASKS)}")
    _require(manifest.get("precision") in PRECISIONS, path,
             f"precision must be one of {sorted(PRECISIONS)}")
    _require(manifest.get("trust_remote_code") is False, path, "trust_remote_code must be false")
    for key in ("tensor_parallel_size", "max_sequence_length"):
        value = manifest.get(key)
        _require(type(value) is int and value == 1, path, f"{key} must be the integer 1")
    _require(isinstance(manifest.get("testcases"), list) and bool(manifest["testcases"]), path,
             "testcases must be a non-empty list")
    for case in manifest["testcases"]:
        _require(isinstance(case, dict) and isinstance(case.get("name"), str) and bool(case["name"]),
                 path, "every testcase needs a non-empty string name")
        for key in ("premerge", "transpose"):
            _require(isinstance(case.get(key), bool), path, f"{case['name']} {key} must be a boolean")
        image = case.get("test_image")
        _require(isinstance(image, str) and Path(image).name == image and image not in ("", ".", ".."),
                 path, f"{case['name']} test_image must name a file in tests/data")
        for key in THRESHOLDS:
            value = case.get(key)
            _require(type(value) in (int, float) and math.isfinite(value) and 0.0 < value <= 1.0,
                     path, f"{case['name']} {key} must be a number in (0, 1]; got {value!r}")


def _case_index() -> dict[str, tuple[Path, dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(manifest, dict), path, "must hold a JSON object")
        _check_manifest(path, manifest)
        for case in manifest["testcases"]:
            name = case["name"]
            _require(name not in result and name != ELSEWHERE, path,
                     f"testcase {name} is declared twice or reserved")
            result[name] = (path, manifest, case)
    return result


CASES = _case_index()


def _filters(config, option: str) -> set[str]:
    return {item.strip() for raw in config.getoption(option) or [] for item in str(raw).split(",")
            if item.strip()}


def _collects_only_this_family(config) -> bool:
    """Whether every path the session was given lies in this family (pytest families/dpt/...)."""
    args = [str(arg).split("::")[0] for arg in getattr(config, "args", None) or []]
    base = Path(getattr(getattr(config, "invocation_params", None), "dir", None) or Path.cwd())
    family_root = TEST_ROOT.parent
    return bool(args) and all((base / arg).resolve().is_relative_to(family_root) for arg in args)


def _selected_cases(config) -> tuple[list[str], bool]:
    """The selected cases, and whether any E2E selector was given.

    A selector is this family's when it is one of its names (the family, a
    manifest, a case) or extends a manifest name, so that a misspelled case such
    as ``dpt-hybrid-midas-landscap`` is caught in any session. Every case or
    manifest selector of this family must then run. When the selection is this
    family's alone (every model selector is its own, or the session collects only
    its files), every selector must run. A selection of another family leaves this
    file empty, so pytest skips it.
    """
    model_filters = _filters(config, "--e2e-model")
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = _filters(config, "--e2e-testcase")
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = not model_filters or bool({FAMILY, manifest["name"], name} & model_filters)
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    manifests = {manifest["name"] for _, manifest, _ in CASES.values()}
    own_names = {FAMILY} | manifests | set(CASES)

    def ours(name: str) -> bool:
        return name in own_names or any(name.startswith(f"{manifest}-") for manifest in manifests)

    our_models = {name for name in model_filters if ours(name)}
    our_cases = {name for name in testcase_filters if ours(name)}
    if model_filters and not our_models and not our_cases:
        return [], True  # another family's selection
    alone = (bool(model_filters) and model_filters == our_models) or _collects_only_this_family(config)
    if not (our_models or our_cases or alone):
        return [], True  # another family's testcases
    if alone:
        unmatched = (testcase_filters - set(selected)) | (model_filters - own_names)
    else:
        unmatched = (our_cases - set(selected)) | (our_models - own_names)
    if not selected or unmatched:
        raise pytest.UsageError(
            f"the E2E selection of {FAMILY} "
            + (f"asks for what it cannot run ({', '.join(sorted(unmatched))})" if unmatched
               else "matches none of its cases")
            + f" (models: {', '.join(sorted(model_filters)) or 'none'}; testcases: "
            f"{', '.join(sorted(testcase_filters)) or 'none'}); its cases are "
            + ", ".join(sorted(CASES)))
    return sorted(selected), True


def _require_one_checkpoint(names: list[str]) -> None:
    """The model-directory override names one checkpoint; refuse it for a selection of several.

    Otherwise every selected case would build and run the override's checkpoint,
    and a case of another checkpoint would pass without testing it.
    """
    if not os.environ.get(MODEL_DIR_OVERRIDE):
        return
    checkpoints = sorted({(CASES[name][1]["hf_id"], CASES[name][1]["hf_revision"])
                          for name in names})
    if len(checkpoints) > 1:
        raise pytest.UsageError(
            f"{MODEL_DIR_OVERRIDE} stands for one checkpoint, but the selected cases use "
            f"{len(checkpoints)}: " + ", ".join(f"{hf_id}@{revision}" for hf_id, revision in checkpoints)
            + "; select the cases of one checkpoint")


def _testcase_only(config) -> bool:
    return bool(_filters(config, "--e2e-testcase")) and not _filters(config, "--e2e-model") \
        and not config.getoption("--e2e-models-file")


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        if enabled and not names and _testcase_only(metafunc.config):
            metafunc.parametrize("case_name", [ELSEWHERE], ids=[ELSEWHERE])
            return
        parameters = names
        if enabled:
            _require_one_checkpoint(names)
        else:
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
    # Absolute, so that a relative TRTMC_BINARY such as ./trtmc launches the
    # checked file: Path drops the "./", and subprocess looks a bare name up on PATH.
    path = Path(value).absolute()
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    # pytest_generate_tests refuses the override for a selection of several checkpoints.
    explicit = os.environ.get(MODEL_DIR_OVERRIDE)
    if explicit:
        return _required_path(explicit, MODEL_DIR_OVERRIDE)
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True
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
    return binary, runtime_root


def _asset(name: str) -> Path:
    path = TEST_ROOT / "data" / str(name)
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


# One bundle per manifest and one reference model per checkpoint, shared by the
# cases of a session.
_BUNDLES: dict[str, Path] = {}
_REFERENCES: dict[str, tuple] = {}


def _bundle(model_dir: Path, manifest: dict, tmp_path_factory) -> Path:
    if manifest["name"] not in _BUNDLES:
        bundle = tmp_path_factory.mktemp(manifest["name"]) / manifest["bundle"]
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
                max_sequence_length=manifest["max_sequence_length"],
            )
        )
        _BUNDLES[manifest["name"]] = bundle
    return _BUNDLES[manifest["name"]]


def _input_image(case: dict, tmp_path: Path):
    """The case's image as RGB, turned to portrait when asked, and its PNG copy."""
    from PIL import Image

    with Image.open(_asset(case["test_image"])) as source:
        image = source.convert("RGB")
    if case["transpose"]:
        image = image.transpose(Image.Transpose.ROTATE_90)
    path = tmp_path / "input.png"
    image.save(path)
    return image, path


def _native(binary: Path, runtime_root: Path, bundle: Path, image_path: Path):
    import numpy as np

    invocation = [str(binary), "depth", str(bundle), "--runtime-root", str(runtime_root),
                  "--image", str(image_path)]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        invocation, check=True, capture_output=True, text=True, env=env, timeout=3600
    )
    record_evidence("commands", {"argv": completed.args})
    # The CLI prints its result as the last line, one JSON object.
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            payload = json.loads(line)
            break
    assert isinstance(payload, dict), f"native {FAMILY} returned no JSON object: {completed.stdout[-1000:]}"
    height, width = payload.get("height"), payload.get("width")
    assert type(height) is int and type(width) is int and height > 0 and width > 0, (
        f"native {FAMILY} returned no positive integer height and width: {height!r}, {width!r}"
    )
    assert isinstance(payload.get("depth"), list), f"native {FAMILY} returned no depth list"
    depth = np.asarray(payload["depth"], dtype=np.float64)
    assert depth.size == height * width, f"native {FAMILY} depth map does not cover the image"
    return depth.reshape(height, width)


def _official(model_dir: Path):
    import torch
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    key = str(model_dir)
    if key not in _REFERENCES:
        processor = DPTImageProcessor.from_pretrained(key)
        model = DPTForDepthEstimation.from_pretrained(key, dtype=torch.float32).eval()
        _REFERENCES[key] = (processor, model)
    return _REFERENCES[key]


def _reference(model_dir: Path, image):
    import numpy as np
    import torch

    processor, model = _official(model_dir)
    encoding = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]
    return depth.float().numpy().astype(np.float64)


def _depth_png(depth, scale: float, path: Path) -> Path:
    import numpy as np
    from PIL import Image

    levels = np.clip(np.asarray(depth) / scale * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(levels).save(path)
    return path


def _depth_metrics(native, reference) -> dict[str, float]:
    """Correlation, and the largest and mean error relative to the largest reference depth."""
    import numpy as np

    assert reference.ndim == 2 and native.shape == reference.shape, (
        f"native depth map shape {native.shape} differs from the reference's {reference.shape}"
    )
    assert np.isfinite(native).all(), f"native {FAMILY} depth map is not finite"
    assert np.isfinite(reference).all(), "reference depth map is not finite"
    scale = float(np.abs(reference).max())
    assert scale > 0.0, "reference depth map is all zero"
    correlation = float(np.corrcoef(native.ravel(), reference.ravel())[0, 1])
    assert math.isfinite(correlation), "a constant depth map has no correlation"
    difference = np.abs(native - reference)
    return {
        "scale": scale,
        "correlation": correlation,
        "max_error": float(difference.max() / scale),
        "mean_error": float(difference.mean() / scale),
    }


def _assert_parity(native, reference, image, case: dict, tmp_path: Path) -> None:
    assert native.shape == (image.height, image.width), (
        f"depth map shape {native.shape} is not the image's {(image.height, image.width)}"
    )
    metrics = _depth_metrics(native, reference)
    # Bounds from the manifests (correlation / largest / mean error): fp32 0.999999 /
    # 1e-4 / 1e-5, fp16 0.99999 / 0.05 / 0.002. Calibrated on 1x B300 (two systems),
    # TensorRT 11.4:
    # - healthy fp32, 14 runs (8 builds): > 0.9999999999995 / 2.4e-6 - 8.8e-6 /
    #   2.0e-7 - 2.5e-7; healthy fp16, 12 runs (6 builds x 2 cases, each build picks other
    #   kernels): >= 0.9999977 / 6.2e-3 - 3.0e-2 / 2.5e-4 - 6.1e-4.
    # - defective, 24 runs: 12 porting mistakes injected into the float32 reference, each
    #   on both cases; at best 0.999999993 / 6.3e-4 / 4.9e-5. The fp32 bounds reject all
    #   24. The fp16 bounds reject the 16 whose errors exceed fp16 rounding (resize
    #   filter, normalization, channel order, stem padding, ViT taps, fusion and output
    #   interpolation); the other 8 (rescale 1/256, BiT weight-standardization eps,
    #   tanh GELU, LayerNorm eps) stay within the fp16 engine's own error, and the fp32
    #   case, which runs in premerge, rejects them.
    checks = [
        {"name": "depth_correlation", "label": "Pearson correlation",
         "scope": "independent_reference", "actual": metrics["correlation"], "operator": ">=",
         "expected": case["min_depth_correlation"],
         "passed": metrics["correlation"] >= case["min_depth_correlation"]},
        {"name": "max_depth_relative_error", "label": "Largest |error| / max |reference|",
         "scope": "independent_reference", "actual": metrics["max_error"], "operator": "<=",
         "expected": case["max_depth_relative_error"],
         "passed": metrics["max_error"] <= case["max_depth_relative_error"]},
        {"name": "mean_depth_relative_error", "label": "Mean |error| / max |reference|",
         "scope": "independent_reference", "actual": metrics["mean_error"], "operator": "<=",
         "expected": case["max_mean_depth_relative_error"],
         "passed": metrics["mean_error"] <= case["max_mean_depth_relative_error"]},
    ]
    native_png = _depth_png(native, metrics["scale"], tmp_path / "native_depth.png")
    reference_png = _depth_png(reference, metrics["scale"], tmp_path / "reference_depth.png")
    record_evidence("native", {"image": native_png})
    record_evidence("reference", {"image": reference_png})
    record_evidence(
        "reference_comparison",
        {
            "label": case["name"],
            "scope": "independent_reference",
            "enforced": True,
            "native": native_png,
            "reference": reference_png,
            "checks": checks,
        },
    )
    failed = [check for check in checks if not check["passed"]]
    assert not failed, f"depth parity failed: {failed}"


def _check_session_selection(request) -> None:
    """Fail when a testcase selector names a case that no collected E2E test of the session runs.

    Selectors that name another collected family's cases pass; a misspelled one
    fails here even when other selectors matched, in any session.
    """
    requested = _filters(request.config, "--e2e-testcase")
    if not requested:
        return
    collected = set()
    for item in request.session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if re.fullmatch(r"test_.*e2e", getattr(item, "originalname", "") or "") and "case_name" in params:
            collected.add(params["case_name"])
    collected.discard(ELSEWHERE)  # this placeholder runs no case
    missing = sorted(requested - collected)
    if missing:
        pytest.fail(f"--e2e-testcase names no E2E case of this session: {', '.join(missing)}")


def check_selection(request) -> None:
    """Both checks of the E2E selectors, for any test of the session (the host-side ones too)."""
    _selected_cases(request.config)
    _check_session_selection(request)


def test_e2e(case_name: str, request, tmp_path: Path, tmp_path_factory) -> None:
    _check_session_selection(request)
    if case_name == ELSEWHERE:
        pytest.skip(f"the E2E selection names other families' cases, not {FAMILY}'s")
    _, manifest, case = CASES[case_name]
    binary, runtime_root = _runtime()
    model_dir = _model_dir(manifest)
    image, image_path = _input_image(case, tmp_path)
    record_evidence("inputs", {"image": image_path})

    with evidence_stage("build"):
        bundle = _bundle(model_dir, manifest, tmp_path_factory)
    with evidence_stage("native"):
        native = _native(binary, runtime_root, bundle, image_path)
    # The CLI has loaded the case's only engine and run it.
    mark_loaded()
    with evidence_stage("reference"):
        reference = _reference(model_dir, image)
    with evidence_stage("compare"):
        _assert_parity(native, reference, image, case, tmp_path)
