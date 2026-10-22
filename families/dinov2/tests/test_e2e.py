# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for DINOv2.

One engine serves both halves of one Task, so the native side is exercised
once: the patch rows are the token features and the class row is the pooled
vector.

**The gate is the feature, not the cosine.** Five structural errors were
built on purpose and measured against this checkpoint - LayerScale dropped,
the position table truncated instead of interpolated, resized bilinear,
resized with ``align_corners=True``, and the tanh GELU in place of the exact
one. The weakest of them sets the gate:

=====================================  =========  ==========
rule broken                            maxabs     cosine
=====================================  =========  ==========
(correct build)                        3.6e-04    1.00000012
tanh GELU instead of exact             6.1e-02    0.99998868
position table bilinear                8.75       0.99477798
position table align_corners=True      22.96      0.97298789
position table truncated               24.18      0.43815657
LayerScale dropped                     26.77      0.04267145
=====================================  =========  ==========

A cosine floor cannot separate the first two rows, and a floor loose enough
to admit the correct build would also admit an error of 8.75. The feature
values are gated instead.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "dinov2"
TASKS = frozenset({"image_to_token_and_pooled_features"})
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




# Measured on this case: a correct build is within 3.6e-04 of the reference
# feature and the weakest deliberate break is 6.1e-02, so this sits 14x above
# the noise and 12x below the break.
_FEATURE_TOLERANCE = 5e-3
# Kept only to fail early and loudly on a gross error. It cannot separate a
# correct build from the tanh-GELU break above, which is why it is not the
# gate.
_COSINE_FLOOR = 0.999


def _reference(model_dir: Path, case: dict):
    """The reference's own features, computed the way the reference computes them."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    # use_fast=False is required, not a preference: the fast processor resizes
    # with torchvision and the slow one with Pillow, and this family
    # reproduces Pillow's bicubic coefficient for coefficient.
    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False)
    batch = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model(pixel_values=batch["pixel_values"])
    return output.last_hidden_state[0].tolist()


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

    reference = _reference(model_dir, case)
    payload = _run_json(binary, runtime_root, bundle, case, "extract-features", "--task",
                        "image_to_token_and_pooled_features", "--image",
                        str(_asset(case["test_image"])))

    shape = [int(value) for value in payload["last_hidden_state_shape"]]
    assert shape[0] == 1, f"{FAMILY} returned a batch of {shape[0]}"
    rows, columns = shape[1], shape[2]
    assert (rows, columns) == (len(reference), len(reference[0])), (
        f"{FAMILY} returned {rows}x{columns}, reference {len(reference)}x{len(reference[0])}"
    )

    values = [float(value) for value in payload["last_hidden_state"]]
    assert len(values) == rows * columns
    native = [values[index * columns:(index + 1) * columns] for index in range(rows)]

    deltas = [max(abs(a - b) for a, b in zip(row, want))
              for row, want in zip(native, reference)]
    worst = max(deltas)
    cosine = _cosine([v for row in native for v in row],
                     [v for row in reference for v in row])
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} feature cosine {cosine}"
    assert worst <= _FEATURE_TOLERANCE, (
        f"{FAMILY} features differ from the reference by {worst}"
    )

    # The class row comes first and the patches follow it in row-major order.
    # Getting this ordering wrong would still return the right count.
    tokens = payload["tokens"]
    assert len(tokens) == rows
    assert tokens[0]["role"] == "class"
    patches = tokens[1:]
    grid = int(round(len(patches) ** 0.5))
    assert grid * grid == len(patches), f"{FAMILY} patch count {len(patches)} is not square"
    assert all(token["role"] == "patch" for token in patches)
    expected_positions = [(row, column) for row in range(grid) for column in range(grid)]
    actual_positions = [(int(token["grid_row"]), int(token["grid_column"]))
                        for token in patches]
    assert actual_positions == expected_positions, f"{FAMILY} patch order is not row-major"

    # Each footprint is in original-image coordinates, so the centre crop puts
    # every box strictly inside the source and in increasing order.
    boxes = [[float(v) for v in token["source_normalized_box"]] for token in patches]
    for x_min, y_min, x_max, y_max in boxes:
        assert 0.0 <= x_min < x_max <= 1.0, f"{FAMILY} box x out of range: {x_min}..{x_max}"
        assert 0.0 <= y_min < y_max <= 1.0, f"{FAMILY} box y out of range: {y_min}..{y_max}"
    assert boxes[0][0] < boxes[1][0], f"{FAMILY} boxes do not advance across a row"
    assert boxes[0][1] < boxes[grid][1], f"{FAMILY} boxes do not advance down a column"

    # The pooled vector is the class row of the same forward pass, not a
    # second inference and not a mean over the patches.
    pooled = [float(value) for value in payload["pooler_output"]]
    assert payload["pooling"] == "class_token"
    assert payload["normalization"] == "none"
    pooled_delta = max(abs(a - b) for a, b in zip(pooled, native[0]))
    assert pooled_delta == 0.0, (
        f"{FAMILY} pooled vector is not the class row: {pooled_delta}"
    )
    norm = math.sqrt(sum(value * value for value in pooled))
    assert abs(norm - 1.0) > 1e-3, (
        f"{FAMILY} reports normalization=none but returned a unit vector: {norm}"
    )

    print(
        f"{FAMILY} parity: {rows} tokens x {columns} features, "
        f"feature maxabs {worst:.3e}, cosine {cosine:.8f}, "
        f"class-row maxabs {deltas[0]:.3e}"
    )
