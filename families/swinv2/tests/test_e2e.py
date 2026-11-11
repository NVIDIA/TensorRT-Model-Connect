# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for Swin Transformer V2.

**The gate is the logit values; the prediction is no gate at all.** Rules of the
model were broken on purpose and the result compared with the transformers
float64 reference on this case's image (NumPy float64 re-run of the engine's
layer plan, the same plan as the engine):

=============================================  =========  ==========  =====  ========
rule broken                                    maxabs     cosine      top-1  top-5
=============================================  =========  ==========  =====  ========
(correct plan)                                 2.4e-07    1.00000000  656    same
shifted-window mask added once, not twice      2.4e-07    1.00000000  656    same
layer_norm_eps 1e-6 instead of 1e-5            2.2e-03    0.99999992  656    same
tanh GELU instead of exact                     9.4e-03    0.99999474  656    same
no cyclic shift                                3.3e-01    0.99747780  656    differs
bilinear resize instead of bicubic             1.65       0.98179880  656    differs
patch-merge gather order swapped               4.89       0.63002536  656    differs
dot-product attention instead of cosine        4.97       0.63811204  656    differs
position bias without ``16 * sigmoid``         12.6       0.71447263  654    differs
=============================================  =========  ==========  =====  ========

A top-1 assertion passes all but the last row, including an attention that is
not SwinV2's at all. So the logits themselves are gated. Built as TensorRT
engines (GB200), the tanh-GELU, epsilon, mask and shift rows land at the same
values; ``test_gate_rejects_a_tanh_gelu_engine`` re-measures the tanh row on
every run. The mask row cannot be told apart on this checkpoint: no
shifted-window score comes within reach of either fill. The family reproduces
the reference's double application anyway (``window.py``), and
``tests/test_model.py`` pins it against the reference mask.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "swinv2"
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
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (_, manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
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
    return binary, runtime_root


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
            fp32_layers=tuple(int(layer) for layer in manifest.get("fp32_layers", ())),
        )
    )


def _run_json(
    binary: Path, runtime_root: Path, bundle: Path, case: dict, command: str, *arguments: str
) -> dict:
    invocation = [str(binary), command, str(bundle), "--runtime-root", str(runtime_root),
                  *arguments]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
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
    assert all(payload == payloads[0] for payload in payloads)
    return payloads[0]


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    return path


# Measured (GB200, TensorRT 11.4, TF32 off): a correct float32 engine is
# 7.6e-06 from the float32 reference on this case, and at most 1.9e-05 on nine
# other PNG images run through the native CLI. The weakest structural break of
# the table above that a logit tolerance can claim to catch, the layer-norm
# epsilon, is 2.2e-03 away. This sits 50x above the noise and 2.2x below that
# break, 9x below the tanh GELU.
_LOGIT_TOLERANCE = 1e-3
# Kept only to fail early and loudly on a gross error.
_COSINE_FLOOR = 0.9999


def _reference(model_dir: Path, case: dict) -> tuple[list[float], list[str]]:
    """The reference's own float32 logits, computed the way the reference computes them."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    model = AutoModelForImageClassification.from_pretrained(
        str(model_dir), dtype=torch.float32).eval()
    # use_fast=False is required, not a preference: the fast processor resizes
    # with torchvision and the slow one with Pillow, and this family
    # reproduces Pillow's bicubic coefficient for coefficient.
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


def _check_logits(scores: list[float], reference: list[float]) -> tuple[float, float]:
    """Assert the float32 gate; return the max abs difference and the cosine."""
    assert len(scores) == len(reference), (
        f"{FAMILY} returned {len(scores)} scores, reference {len(reference)}"
    )
    worst = max(abs(a - b) for a, b in zip(scores, reference))
    cosine = _cosine(scores, reference)
    assert cosine >= _COSINE_FLOOR, f"{FAMILY} logit cosine {cosine}"
    assert worst <= _LOGIT_TOLERANCE, f"{FAMILY} logits differ from the reference by {worst}"
    return worst, cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    reference, reference_labels = _reference(model_dir, case)
    payload = _run_json(binary, runtime_root, bundle, case, "classify", "--task",
                        "image_to_class_scores", "--image", str(_asset(case["test_image"])))

    scores = [float(value) for value in payload["scores"]]
    worst, cosine = _check_logits(scores, reference)

    # The head emits logits; nothing in the runtime softmaxes them, and the
    # Task reports which kind they are rather than leaving it to the caller.
    assert payload["score_kind"] == "logit"

    # Every class is named, in class order. A partial or reordered vocabulary
    # would line names up against the wrong scores while still returning the
    # right count.
    labels = [str(label) for label in payload["labels"]]
    assert labels == reference_labels, f"{FAMILY} labels do not match the checkpoint's id2label"

    # The prediction is reported but not gated: measured, it survives every
    # structural error but one in the table above.
    top = max(range(len(scores)), key=lambda index: scores[index])
    assert top == max(range(len(reference)), key=lambda index: reference[index])
    print(
        f"{FAMILY} parity: {len(scores)} classes, logit maxabs {worst:.3e}, "
        f"cosine {cosine:.8f}, top-1 {top} ({labels[top]!r})"
    )


def test_float32_gate_holds_its_tolerance() -> None:
    """Noise of the measured size passes; an epsilon-sized defect fails."""
    import random

    rng = random.Random(0)
    reference = [rng.gauss(0.0, 3.0) for _ in range(1000)]
    noisy = [value + rng.gauss(0.0, 4e-6) for value in reference]
    _check_logits(noisy, reference)
    shifted = list(noisy)
    shifted[7] += 2.2e-3
    with pytest.raises(AssertionError, match="differ from the reference"):
        _check_logits(shifted, reference)


def _tanh_gelu(network, tensor, *, dtype):
    """The defect under test: the tanh approximation in place of the exact GELU."""
    import tensorrt as trt

    from .. import graph

    rank = len(tensor.shape)

    def constant(value):
        return graph.scalar(network, value, rank, dtype=dtype)

    cube = graph.multiply(network, graph.multiply(network, tensor, tensor), tensor)
    inner = graph.multiply(network, constant(math.sqrt(2.0 / math.pi)),
                           graph.add(network, tensor, graph.multiply(network, constant(0.044715), cube)))
    tanh = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    gate = graph.add(network, constant(1.0), tanh)
    return graph.multiply(network, graph.multiply(network, constant(0.5), tensor), gate)


def test_gate_rejects_a_tanh_gelu_engine(case_name: str, monkeypatch) -> None:
    """A correct engine passes the gate and a tanh-GELU engine fails it.

    Both engines are built from the case checkpoint and run in this process on
    the case image, preprocessed the way the reference preprocesses it.
    """
    import tensorrt as trt
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    from .. import config as config_module, graph, model

    _, manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    reference, _ = _reference(model_dir, case)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False)
    image = Image.open(_asset(case["test_image"])).convert("RGB")
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    pixels = pixels.to(device="cuda", dtype=torch.float32).contiguous()
    weights = model._open_torch_checkpoint(model_dir)
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

    correct = run(model.build_engine(weights, cfg))
    worst, _ = _check_logits(correct, reference)

    monkeypatch.setattr(graph, "gelu", _tanh_gelu)
    broken = run(model.build_engine(weights, cfg))
    broken_worst = max(abs(a - b) for a, b in zip(broken, reference))
    with pytest.raises(AssertionError, match="differ from the reference"):
        _check_logits(broken, reference)
    print(f"{FAMILY} gate: correct maxabs {worst:.3e}; tanh GELU maxabs {broken_worst:.3e}")
