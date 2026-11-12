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

from tools.e2e_evidence import evidence_stage, record_evidence, record_reference_comparison

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

# fp16 cannot use the tolerance above. Measured on this case (B200 and B300,
# TensorRT 11.1 and 11.4, several builds each), a correct fp16 engine is 5.2e-03 to
# 1.3e-02 off the reference and a tanh-GELU fp16 engine 1.6e-02 to 2.5e-02,
# so no logit tolerance sits clearly between the two. fp16 is gated twice:
#
# * a logit tolerance set from the fp16 noise: 3e-02 is 2.3x above the worst
#   correct build and 13x below the bicubic-resize row of the table above,
#   so it still catches the resize and classifier-row errors. It is not what
#   catches the tanh GELU.
# * the tanh-GELU check. The reference is computed a second time with the
#   tanh GELU, and the engine must be closer to the exact reference than to
#   that one: the share of the defect it carries,
#   (engine - exact) . (tanh - exact) / |tanh - exact|^2, must stay below 0.5.
#   It uses all 1000 logits rather than the single worst one, so the fp16
#   noise mostly cancels. Measured on the same builds: correct fp16 -0.25
#   to 0.10, tanh-GELU fp16 0.87 to 1.02, tanh-GELU fp32 1.00.
#
# test_fp16_gate_rejects_a_tanh_gelu_build builds both and checks this.
_FP16_LOGIT_TOLERANCE = 3e-2
_TANH_SHARE_LIMIT = 0.5

# fp8 (FP8 projections on the fp16 graph) has its own gate. Measured on this case (B200
# and B300, TensorRT 11.4 and 11.1), a correct fp8 engine is 0.17 to 0.30 off the
# reference at cosine 0.99881 to 0.99903. The tolerance and the cosine floor sit 3x above
# that noise, so the gate still catches the nearest-resize and classifier-row errors of the
# table above (1.38 and 3.18, cosine 0.968 and 0.866). The smaller errors (bicubic
# resize, tanh GELU, epsilon) are inside the FP8 noise; the fp32 and fp16 cases, which
# share the graph and the preprocessing, guard them.
_FP8_LOGIT_TOLERANCE = 0.9
_FP8_COSINE_FLOOR = 0.996


def _gate(manifest: dict) -> str:
    """The gate of a case: "fp8" for FP8 builds, else the build precision."""
    if manifest.get("quantization") == "fp8":
        return "fp8"
    return manifest["precision"]


def _reference(model_dir: Path, case: dict, hidden_act: str | None = None):
    """The reference's own logits, computed the way the reference computes them.

    ``hidden_act`` overrides the checkpoint's activation, which is how the
    tanh-GELU reference of the fp16 check is made.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    overrides = {} if hidden_act is None else {"hidden_act": hidden_act}
    model = AutoModelForImageClassification.from_pretrained(
        str(model_dir), dtype=torch.float32, **overrides).eval()
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


def _tanh_share(scores: list[float], reference: list[float],
                tanh_reference: list[float]) -> float:
    """How much of the tanh-GELU defect the scores carry: 0 for none, 1 for all of it."""
    defect = [t - r for t, r in zip(tanh_reference, reference)]
    error = [s - r for s, r in zip(scores, reference)]
    size = sum(d * d for d in defect)
    assert size > 0.0, "the tanh-GELU reference does not differ from the exact one"
    return sum(e * d for e, d in zip(error, defect)) / size


def _check_logits(scores: list[float], reference: list[float], precision: str,
                  tanh_reference: list[float] | None = None) -> tuple[float, float]:
    """Assert the gate ``precision`` (fp32, fp16 or fp8); return max abs difference and cosine."""
    assert len(scores) == len(reference), (
        f"{FAMILY} returned {len(scores)} scores, reference {len(reference)}"
    )
    worst = max(abs(a - b) for a, b in zip(scores, reference))
    cosine = _cosine(scores, reference)
    floor = _FP8_COSINE_FLOOR if precision == "fp8" else _COSINE_FLOOR
    assert cosine >= floor, f"{FAMILY} {precision} logit cosine {cosine}"
    tolerance = {"fp16": _FP16_LOGIT_TOLERANCE, "fp8": _FP8_LOGIT_TOLERANCE}.get(
        precision, _LOGIT_TOLERANCE)
    assert worst <= tolerance, (
        f"{FAMILY} {precision} logits differ from the reference by {worst}"
    )
    record_evidence("thresholds", {"cosine_floor": floor, "max_abs_error": tolerance})
    record_reference_comparison(
        "Reference scores",
        {"scores": scores},
        {"scores": reference},
        [
            {
                "name": "cosine",
                "scope": "independent_reference",
                "actual": cosine,
                "operator": ">=",
                "expected": floor,
                "passed": cosine >= floor,
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
        assert tanh_reference is not None, "the fp16 gate needs the tanh-GELU reference"
        share = _tanh_share(scores, reference, tanh_reference)
        record_evidence("thresholds", {"max_tanh_defect_share": _TANH_SHARE_LIMIT})
        record_evidence("diagnostics", {"counterfactual_control": {
            "name": "Tanh approximation control", "native": {"scores": scores},
            "reference": {"scores": reference}, "tanh_reference": tanh_reference,
            "actual": share, "operator": "<", "expected": _TANH_SHARE_LIMIT,
            "passed": share < _TANH_SHARE_LIMIT,
        }})
        assert share < _TANH_SHARE_LIMIT, (
            f"{FAMILY} fp16 logits carry {share:.2f} of the tanh-GELU defect"
        )
    return worst, cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence(
        "inputs", {"manifest": manifest, "case": case, "image": TEST_ROOT / case["test_image"]}
    )
    record_evidence(
        "thresholds",
        {
            "cosine_floor": _FP8_COSINE_FLOOR
            if manifest.get("quantization") == "fp8"
            else _COSINE_FLOOR,
            "max_abs_error": _FP8_LOGIT_TOLERANCE
            if manifest.get("quantization") == "fp8"
            else _FP16_LOGIT_TOLERANCE
            if manifest["precision"] == "fp16"
            else _LOGIT_TOLERANCE,
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
        reference, reference_labels = _reference(model_dir, case)
    record_evidence("reference", {"scores": reference, "labels": reference_labels})
    gate = _gate(manifest)
    tanh_reference = None
    if gate == "fp16":
        with evidence_stage("reference"):
            tanh_reference, _ = _reference(model_dir, case, hidden_act="gelu_pytorch_tanh")
        record_evidence("diagnostics", {"tanh_reference": tanh_reference})
    with evidence_stage("native"):
        payload = _run_json(binary, runtime_root, bundle, case, "classify", "--task",
                            "image_to_class_scores", "--image",
                            str(_asset(case["test_image"])))
    record_evidence("native", payload)

    scores = [float(value) for value in payload["scores"]]
    with evidence_stage("compare"):
        worst, cosine = _check_logits(scores, reference, gate, tanh_reference)
    record_evidence("metrics", {"max_abs_error": worst, "cosine": cosine})

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


def _synthetic(seed: int, share: float, noise: float) -> tuple[list, list, list]:
    """Scores that carry ``share`` of a tanh-like defect plus random noise of size ``noise``."""
    import random

    rng = random.Random(seed)
    reference = [rng.gauss(0.0, 3.0) for _ in range(1000)]
    defect = [rng.gauss(0.0, 4e-3) for _ in range(1000)]
    tanh_reference = [r + d for r, d in zip(reference, defect)]
    scores = [r + share * d + rng.gauss(0.0, noise) for r, d in zip(reference, defect)]
    return scores, reference, tanh_reference


def test_tanh_share_measures_the_defect_not_the_noise() -> None:
    """Noise of the fp16 size leaves the share near 0; the defect moves it to 1."""
    for seed in range(5):
        scores, reference, tanh_reference = _synthetic(seed, 0.0, 2e-3)
        assert abs(_tanh_share(scores, reference, tanh_reference)) < 0.1
        scores, reference, tanh_reference = _synthetic(seed, 1.0, 2e-3)
        assert abs(_tanh_share(scores, reference, tanh_reference) - 1.0) < 0.1


def test_fp16_gate_fails_a_defect_inside_the_fp16_tolerance() -> None:
    """The fp16 gate does not rest on its logit tolerance to catch the tanh GELU."""
    scores, reference, tanh_reference = _synthetic(0, 0.0, 2e-3)
    _check_logits(scores, reference, "fp16", tanh_reference)
    scores, reference, tanh_reference = _synthetic(0, 1.0, 2e-3)
    assert max(abs(a - b) for a, b in zip(scores, reference)) < _FP16_LOGIT_TOLERANCE
    with pytest.raises(AssertionError, match="tanh-GELU defect"):
        _check_logits(scores, reference, "fp16", tanh_reference)
    with pytest.raises(AssertionError, match="tanh-GELU reference"):
        _check_logits(scores, reference, "fp16")


def test_fp32_gate_is_unchanged() -> None:
    """fp32 keeps its own tolerance and needs no second reference."""
    scores, reference, _ = _synthetic(0, 0.0, 1e-5)
    _check_logits(scores, reference, "fp32")
    shifted = list(scores)
    shifted[0] += 2.5e-3
    with pytest.raises(AssertionError, match="differ from the reference"):
        _check_logits(shifted, reference, "fp32")


def test_fp8_gate_has_its_own_tolerance() -> None:
    """fp8 noise passes the fp8 gate, fails the fp16 one, and a gross error still fails."""
    scores, reference, _ = _synthetic(0, 0.0, 0.1)
    worst, cosine = _check_logits(scores, reference, "fp8")
    assert _FP16_LOGIT_TOLERANCE < worst < _FP8_LOGIT_TOLERANCE and cosine > _FP8_COSINE_FLOOR
    shifted = list(scores)
    shifted[0] += 1.5
    with pytest.raises(AssertionError, match="differ from the reference"):
        _check_logits(shifted, reference, "fp8")
    assert _gate({"precision": "fp16", "quantization": "fp8"}) == "fp8"
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


def test_fp16_gate_rejects_a_tanh_gelu_build(fp16_case_name: str, monkeypatch) -> None:
    """A correct fp16 engine passes the fp16 gate and a tanh-GELU fp16 engine fails it.

    Both engines are built from the case checkpoint and run in this process on
    the case image, preprocessed the way the reference preprocesses it.
    """
    import numpy as np
    import tensorrt as trt
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from .. import config as config_module, graph, model

    _, manifest, case = CASES[fp16_case_name]
    assert _gate(manifest) == "fp16"
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    reference, _ = _reference(model_dir, case)
    tanh_reference, _ = _reference(model_dir, case, hidden_act="gelu_pytorch_tanh")
    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False)
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    pixels = pixels.to(device="cuda", dtype=torch.float32).contiguous()
    weights = model._open_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)

    def run(plan: bytes) -> list[float]:
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        context = engine.create_execution_context()
        out = torch.empty(tuple(engine.get_tensor_shape("logits")), device="cuda")
        context.set_tensor_address("pixel_values", pixels.data_ptr())
        context.set_tensor_address("logits", out.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return [float(value) for value in out.reshape(-1).tolist()]

    correct = run(model.build_engine(weights, cfg, dtype=np.float16))
    worst, _ = _check_logits(correct, reference, "fp16", tanh_reference)
    correct_share = _tanh_share(correct, reference, tanh_reference)

    monkeypatch.setattr(graph, "add_gelu", _tanh_gelu)
    broken = run(model.build_engine(weights, cfg, dtype=np.float16))
    broken_share = _tanh_share(broken, reference, tanh_reference)
    # The share check alone fails it, whatever its logit distance.
    assert broken_share >= _TANH_SHARE_LIMIT, broken_share
    with pytest.raises(AssertionError):
        _check_logits(broken, reference, "fp16", tanh_reference)
    print(
        f"{FAMILY} fp16 gate: correct maxabs {worst:.3e} share {correct_share:.2f}; "
        f"tanh GELU maxabs {max(abs(a - b) for a, b in zip(broken, reference)):.3e} "
        f"share {broken_share:.2f}"
    )
