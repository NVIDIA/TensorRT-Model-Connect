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

fp16 (the default precision) is noisier than the tanh-GELU row, so its case
adds a second check; see ``_FP16_FEATURE_TOLERANCE``.
"""
from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence, record_reference_comparison

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
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
    for fixture in ("case_name", "fp16_case_name"):
        if fixture not in metafunc.fixturenames:
            continue
        names, enabled = _selected_cases(metafunc.config)
        if fixture == "fp16_case_name":
            names = [name for name in names if _gate(CASES[name][1]) == "fp16"]
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

# fp16 cannot use the tolerance above. Measured on this case (B200 and B300,
# TensorRT 11.4 and 11.1, several builds each), a correct fp16 engine is 0.35
# to 1.16 off the reference, and a tanh-GELU fp16 engine 0.40 to 0.98: the
# tanh-GELU break (6.2e-02) is far inside the fp16 noise. fp16 is gated twice:
#
# * a feature tolerance set from the fp16 noise: 3.0 is 2.6x above the worst
#   correct build and 4.8x below the bilinear position table of the table
#   above, which is 14.3 to 16.0 off in fp16 (cosine 0.9952 to 0.9956, also
#   below the floor). It is not what catches the tanh GELU.
# * the tanh-GELU check. The reference is computed a second time with the
#   tanh GELU, and the engine must be closer to the exact reference than to
#   that one: the share of the defect it carries,
#   (engine - exact) . (tanh - exact) / |tanh - exact|^2, must stay below 0.5.
#   It uses all 197,376 feature values rather than the single worst one, so
#   the fp16 noise mostly cancels. Measured on the same builds: correct fp16
#   -0.04 to 0.04, tanh-GELU fp16 1.02 to 1.06, tanh-GELU fp32 1.00.
#
# The cosine floor stays 0.999; correct fp16 builds measure 0.99995 to 0.99999.
# test_fp16_gate_rejects_broken_builds builds a correct, a tanh-GELU and a
# bilinear-table fp16 engine and checks that only the first passes.
_FP16_FEATURE_TOLERANCE = 3.0
_TANH_SHARE_LIMIT = 0.5


def _gate(manifest: dict) -> str:
    """The gate of a case: the build precision (fp32 or fp16)."""
    return manifest["precision"]


def _reference(model_dir: Path, case: dict, hidden_act: str | None = None):
    """The reference's own features, computed the way the reference computes them.

    ``hidden_act`` overrides the checkpoint's activation, which is how the
    tanh-GELU reference of the fp16 check is made.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    overrides = {} if hidden_act is None else {"hidden_act": hidden_act}
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32, **overrides).eval()
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


def _tanh_share(values: list[float], reference: list[float],
                tanh_reference: list[float]) -> float:
    """How much of the tanh-GELU defect the values carry: 0 for none, 1 for all of it."""
    defect = [t - r for t, r in zip(tanh_reference, reference)]
    size = sum(d * d for d in defect)
    assert size > 0.0, "the tanh-GELU reference does not differ from the exact one"
    return sum((v - r) * d for v, r, d in zip(values, reference, defect)) / size


def _check_features(native: list[list[float]], reference: list[list[float]], precision: str,
                    tanh_reference: list[list[float]] | None = None) -> tuple[float, float]:
    """Assert the gate ``precision`` (fp32 or fp16); return the worst row maxabs and cosine."""
    values = [v for row in native for v in row]
    wanted = [v for row in reference for v in row]
    assert len(values) == len(wanted), (
        f"{FAMILY} returned {len(values)} values, reference {len(wanted)}"
    )
    worst = max(abs(a - b) for a, b in zip(values, wanted))
    cosine = _cosine(values, wanted)
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} {precision} feature cosine {cosine}"
    tolerance = _FP16_FEATURE_TOLERANCE if precision == "fp16" else _FEATURE_TOLERANCE
    assert worst <= tolerance, (
        f"{FAMILY} {precision} features differ from the reference by {worst}"
    )
    if precision == "fp16":
        assert tanh_reference is not None, "the fp16 gate needs the tanh-GELU reference"
        share = _tanh_share(values, wanted, [v for row in tanh_reference for v in row])
        assert share < _TANH_SHARE_LIMIT, (
            f"{FAMILY} fp16 features carry {share:.2f} of the tanh-GELU defect"
        )
    record_evidence("thresholds", {"cosine_floor": _COSINE_FLOOR, "max_abs_error": tolerance})
    record_reference_comparison(
        "Reference values",
        {"values": native},
        {"values": reference},
        [
            {
                "name": "cosine",
                "scope": "independent_reference",
                "actual": cosine,
                "operator": ">=",
                "expected": _COSINE_FLOOR,
                "passed": cosine >= _COSINE_FLOOR,
            },
            {
                "name": "max_abs_error",
                "scope": "independent_reference",
                "actual": worst,
                "operator": "<=",
                "expected": tolerance,
                "passed": worst <= tolerance,
            },
        ],
    )
    if precision == "fp16":
        record_evidence("thresholds", {"max_tanh_defect_share": _TANH_SHARE_LIMIT})
        record_evidence("diagnostics", {"counterfactual_control": {
            "name": "Tanh approximation control", "native": {"values": native},
            "reference": {"values": reference}, "tanh_reference": tanh_reference,
            "actual": share, "operator": "<", "expected": _TANH_SHARE_LIMIT,
            "passed": share < _TANH_SHARE_LIMIT,
        }})
    return worst, cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence(
        "inputs", {"manifest": manifest, "case": case, "image": TEST_ROOT / case["test_image"]}
    )
    record_evidence(
        "thresholds",
        {
            "cosine_floor": _COSINE_FLOOR,
            "max_abs_error": _FP16_FEATURE_TOLERANCE
            if manifest["precision"] == "fp16"
            else _FEATURE_TOLERANCE,
        },
    )
    model_dir = _model_dir(manifest)
    record_evidence(
        "checkpoint",
        {
            "model_dir": str(model_dir),
            "hf_id": manifest.get("hf_id"),
            "hf_revision": manifest.get("hf_revision"),
        },
    )
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)

    with evidence_stage("reference"):
        reference = _reference(model_dir, case)
    record_evidence("reference", {"values": np.asarray(reference)})
    gate = _gate(manifest)
    tanh_reference = None
    if gate == "fp16":
        with evidence_stage("reference"):
            tanh_reference = _reference(model_dir, case, hidden_act="gelu_pytorch_tanh")
        record_evidence("diagnostics", {"tanh_reference": tanh_reference})
    with evidence_stage("native"):
        payload = _run_json(binary, runtime_root, bundle, case, "extract-features", "--task",
                            "image_to_token_and_pooled_features", "--image",
                            str(_asset(case["test_image"])))
    record_evidence("native", payload)

    shape = [int(value) for value in payload["last_hidden_state_shape"]]
    assert shape[0] == 1, f"{FAMILY} returned a batch of {shape[0]}"
    rows, columns = shape[1], shape[2]
    assert (rows, columns) == (len(reference), len(reference[0])), (
        f"{FAMILY} returned {rows}x{columns}, reference {len(reference)}x{len(reference[0])}"
    )

    values = [float(value) for value in payload["last_hidden_state"]]
    assert len(values) == rows * columns
    native = [values[index * columns:(index + 1) * columns] for index in range(rows)]
    # Persist the complete matrix in the same row-major shape as the reference;
    # the native JSON fields remain available alongside this display tensor.
    record_evidence("native", {**payload, "values": np.asarray(native)})

    with evidence_stage("compare"):
        worst, cosine = _check_features(native, reference, gate, tanh_reference)
    record_evidence("metrics", {"max_abs_error": worst, "cosine": cosine})
    class_delta = max(abs(a - b) for a, b in zip(native[0], reference[0]))

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
        f"{FAMILY} {gate} parity: {rows} tokens x {columns} features, "
        f"feature maxabs {worst:.3e}, cosine {cosine:.8f}, "
        f"class-row maxabs {class_delta:.3e}"
    )


def _synthetic(seed: int, share: float, noise: float) -> tuple[list, list, list]:
    """Rows that carry ``share`` of a tanh-like defect plus random noise of size ``noise``."""
    import random

    rng = random.Random(seed)
    reference = [[rng.gauss(0.0, 2.0) for _ in range(64)] for _ in range(32)]
    defect = [[rng.gauss(0.0, 1e-2) for _ in range(64)] for _ in range(32)]
    tanh_reference = [[r + d for r, d in zip(rr, dd)] for rr, dd in zip(reference, defect)]
    native = [[r + share * d + rng.gauss(0.0, noise) for r, d in zip(rr, dd)]
              for rr, dd in zip(reference, defect)]
    return native, reference, tanh_reference


def test_tanh_share_measures_the_defect_not_the_noise() -> None:
    """Noise of several times the defect leaves the share near 0; the defect moves it to 1."""
    for seed in range(5):
        native, reference, tanh_reference = _synthetic(seed, 0.0, 2e-2)
        flat = [[v for row in rows for v in row] for rows in (native, reference, tanh_reference)]
        assert abs(_tanh_share(*flat)) < 0.2
        native, reference, tanh_reference = _synthetic(seed, 1.0, 2e-2)
        flat = [[v for row in rows for v in row] for rows in (native, reference, tanh_reference)]
        assert abs(_tanh_share(*flat) - 1.0) < 0.2


def test_fp16_gate_fails_a_defect_inside_the_fp16_tolerance() -> None:
    """The fp16 gate does not rest on its feature tolerance to catch the tanh GELU."""
    native, reference, tanh_reference = _synthetic(0, 0.0, 2e-2)
    _check_features(native, reference, "fp16", tanh_reference)
    native, reference, tanh_reference = _synthetic(0, 1.0, 2e-2)
    with pytest.raises(AssertionError, match="tanh-GELU defect"):
        _check_features(native, reference, "fp16", tanh_reference)
    with pytest.raises(AssertionError, match="tanh-GELU reference"):
        _check_features(native, reference, "fp16")


def test_fp32_gate_is_unchanged() -> None:
    """fp32 keeps its own tolerance and needs no second reference."""
    native, reference, _ = _synthetic(0, 0.0, 1e-4)
    _check_features(native, reference, "fp32")
    native[0][0] += 6e-3
    with pytest.raises(AssertionError, match="differ from the reference"):
        _check_features(native, reference, "fp32")
    assert _gate({"precision": "fp16"}) == "fp16"


def _tanh_gelu(network, inp, dtype=None):
    """The defect under test: the tanh approximation in place of the exact GELU."""
    import numpy as np
    import tensorrt as trt

    from .. import graph

    dtype = np.float32 if dtype is None else dtype
    ones = tuple([1] * len(inp.shape))

    def constant(value):
        return graph.add_constant(network, ones, np.full((1,), value), dtype=dtype)

    def op(left, right, kind):
        return network.add_elementwise(left, right, kind).get_output(0)

    prod, total = trt.ElementWiseOperation.PROD, trt.ElementWiseOperation.SUM
    cube = op(op(inp, inp, prod), inp, prod)
    inner = op(constant(math.sqrt(2.0 / math.pi)),
               op(inp, op(constant(0.044715), cube, prod), total), prod)
    tanh = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    return op(op(constant(0.5), inp, prod), op(constant(1.0), tanh, total), prod)


def _bilinear_table(table, trained_grid, grid):
    """The second defect under test: the position table resized bilinear, not bicubic."""
    import numpy as np
    import torch

    table = np.asarray(table)
    if table.ndim == 3:
        table = table[0]
    width = table.shape[1]
    patches = torch.from_numpy(np.ascontiguousarray(table[1:])).float()
    patches = patches.reshape(1, trained_grid, trained_grid, width).permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        patches, size=(grid, grid), mode="bilinear", align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(grid * grid, width).numpy()
    return np.concatenate([table[:1], resized], axis=0).astype(np.float32)


def test_fp16_gate_rejects_broken_builds(fp16_case_name: str, monkeypatch) -> None:
    """A correct fp16 engine passes the fp16 gate; tanh-GELU and bilinear-table ones fail it.

    The engines are built from the case checkpoint and run in this process on
    the case image, preprocessed the way the reference preprocesses it.
    """
    import numpy as np
    import tensorrt as trt
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from .. import config as config_module, graph, model, position

    _, manifest, case = CASES[fp16_case_name]
    assert _gate(manifest) == "fp16"
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    reference = _reference(model_dir, case)
    tanh_reference = _reference(model_dir, case, hidden_act="gelu_pytorch_tanh")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False)
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    pixels = pixels.to(device="cuda", dtype=torch.float32).contiguous()
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)

    def run(plan: bytes) -> list[list[float]]:
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        out = torch.empty(tuple(engine.get_tensor_shape("last_hidden_state")), device="cuda")
        context.set_tensor_address("pixel_values", pixels.data_ptr())
        context.set_tensor_address("last_hidden_state", out.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return out.reshape(len(reference), -1).tolist()

    def share(native):
        flat = [[v for row in rows for v in row] for rows in (native, reference, tanh_reference)]
        return _tanh_share(*flat)

    correct = run(model.build_engine(weights, cfg, dtype=np.float16))
    worst, _ = _check_features(correct, reference, "fp16", tanh_reference)

    with monkeypatch.context() as patch:
        patch.setattr(graph, "add_gelu", _tanh_gelu)
        tanh = run(model.build_engine(weights, cfg, dtype=np.float16))
    # The share check alone fails it, whatever its feature distance.
    assert share(tanh) >= _TANH_SHARE_LIMIT, share(tanh)
    with pytest.raises(AssertionError, match="tanh-GELU defect"):
        _check_features(tanh, reference, "fp16", tanh_reference)

    with monkeypatch.context() as patch:
        patch.setattr(position, "interpolate_position_table", _bilinear_table)
        bilinear = run(model.build_engine(weights, cfg, dtype=np.float16))
    # Its cosine is far below the floor and its features far beyond the tolerance.
    with pytest.raises(AssertionError, match="feature cosine|differ from the reference"):
        _check_features(bilinear, reference, "fp16", tanh_reference)

    def maxabs(native):
        return max(abs(a - b) for row, want in zip(native, reference) for a, b in zip(row, want))

    def cosine(native):
        return _cosine([v for row in native for v in row], [v for row in reference for v in row])

    print(
        f"{FAMILY} fp16 gate: correct maxabs {worst:.3e} share {share(correct):.2f}; "
        f"tanh GELU maxabs {maxabs(tanh):.3e} share {share(tanh):.2f}; "
        f"bilinear table maxabs {maxabs(bilinear):.3e} cosine {cosine(bilinear):.5f}"
    )
