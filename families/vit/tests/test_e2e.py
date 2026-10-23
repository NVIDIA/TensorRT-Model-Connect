# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for ViT.

**The gate is the logit values, and the prediction is no gate at all.** Six
errors were built on purpose and measured against this checkpoint. Every one
of them still returned the same top-1 class:

=========================================  =========  ==========  =====  =======
rule broken                                maxabs     cosine      top-1  top-5
=========================================  =========  ==========  =====  =======
(correct build)                            1.0e-05    1.00000000  817    ok
layer_norm_eps 1e-6 instead of 1e-12       2.2e-05    0.99999988  817    ok
layer_norm_eps 1e-5 instead of 1e-12       2.7e-04    1.00000012  817    ok
tanh GELU instead of exact                 1.9e-02    0.99999392  817    ok
bicubic resize instead of bilinear         3.8e-01    0.99555570  817    differs
nearest resize instead of bilinear         1.38       0.96767884  817    differs
classifier reads a patch row, not class    3.18       0.86614889  817    differs
=========================================  =========  ==========  =====  =======

A top-1 assertion passes all six, including a nearest-neighbour resize that
moves pixels by 0.949. Top-5 still passes the epsilon and GELU rows, and no
cosine floor separates them either. So the logits themselves are gated.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "vit"
TASKS = frozenset({"image_to_class_scores"})
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




# Measured on this case: a correct build is within 1.0e-05 of the reference
# and the weakest break the gate is meant to catch - the tanh GELU - is
# 1.9e-02. This sits 200x above the noise and 10x below that break.
#
# The two epsilon rows above fall under this tolerance on purpose. Measured,
# they are worth 2.2e-05 and 2.7e-04, which is the same order as numerical
# noise; claiming to catch them would claim a precision this gate does not
# have. config.py still reads the epsilon from the checkpoint rather than
# assuming one.
_LOGIT_TOLERANCE = 2e-3
# Kept only to fail early and loudly on a gross error.
_COSINE_FLOOR = 0.999


def _reference(model_dir: Path, case: dict):
    """The reference's own logits, computed the way the reference computes them."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    model = AutoModelForImageClassification.from_pretrained(
        str(model_dir), dtype=torch.float32).eval()
    # use_fast=False is required, not a preference: the fast processor resizes
    # with torchvision and the slow one with Pillow, and this family
    # reproduces Pillow's bilinear coefficient for coefficient.
    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False)
    batch = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(pixel_values=batch["pixel_values"]).logits[0]
    labels = [model.config.id2label[index] for index in range(logits.shape[0])]
    return logits.tolist(), labels


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    reference, reference_labels = _reference(model_dir, case)
    payload = _run_json(binary, runtime_root, bundle, case, "classify", "--task",
                        "image_to_class_scores", "--image",
                        str(_asset(case["test_image"])))

    scores = [float(value) for value in payload["scores"]]
    assert len(scores) == len(reference), (
        f"{FAMILY} returned {len(scores)} scores, reference {len(reference)}"
    )

    worst = max(abs(a - b) for a, b in zip(scores, reference))
    cosine = _cosine(scores, reference)
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} logit cosine {cosine}"
    assert worst <= _LOGIT_TOLERANCE, (
        f"{FAMILY} logits differ from the reference by {worst}"
    )

    # The head emits logits; nothing in the runtime softmaxes them, and the
    # Task reports which kind they are rather than leaving it to the caller.
    assert payload["score_kind"] == "logit"

    # Every class is named, in class order. A partial or reordered vocabulary
    # would line names up against the wrong scores while still returning the
    # right count.
    labels = [str(label) for label in payload["labels"]]
    assert labels == reference_labels, (
        f"{FAMILY} labels do not match the checkpoint's id2label"
    )

    # The prediction is reported but not gated: measured, it survives every
    # structural error this family was tested against.
    top = max(range(len(scores)), key=lambda index: scores[index])
    assert top == max(range(len(reference)), key=lambda index: reference[index])
    print(
        f"{FAMILY} parity: {len(scores)} classes, logit maxabs {worst:.3e}, "
        f"cosine {cosine:.8f}, top-1 {top} ({labels[top]!r})"
    )
