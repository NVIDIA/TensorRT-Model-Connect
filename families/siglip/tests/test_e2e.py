# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for SigLIP.

One bundle serves all three tasks, so the native side is exercised three
times through ``--task``: the image tower alone, the text tower alone, and
the scored pair.

**The gate is the embedding, not the score and not the cosine.** Four
structural errors were built on purpose and measured - the exact GELU in
place of the tanh one, the pooling head's residual taken after its norm
instead of before, the text sequence pooled at row 0 instead of row -1, and
the packed head projection split as q, v, k. The weakest of them, the GELU,
is the one that sets the gate, and measured on *this* test case it moves:

====================  =========  ==========
quantity              correct    GELU break
====================  =========  ==========
embedding maxabs      2.1e-07    8.4e-04
embedding cosine      0.9999999  0.9999911
pair logit            3.1e-06    1.2e-02
pair probability      9.5e-09    3.8e-05
====================  =========  ==========

A cosine floor cannot separate those. Neither can an absolute tolerance on
the probability: this pair scores 0.0031, far out on the sigmoid's flat
tail, where a logit shift of 0.012 is worth only 3.8e-05 of probability. The
same break on a pair scoring 0.11 moves the probability by 0.0066, which is
why a probability gate looks adequate until it is measured on the case that
actually ships. The embedding is gated instead: it is where the error
originates and its sensitivity does not depend on where the pair happens to
land on the sigmoid.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "siglip"
TASKS = frozenset({"image_to_embedding", "text_to_embedding", "text_image_to_relevance"})
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




# Measured on this case: a correct build is within 2.1e-07 of the reference
# embedding and the weakest deliberate break is 8.4e-04, so this sits 480x
# above the noise and 8x below the break. The logit is checked as well, as an
# end-to-end confirmation that the scale and bias are applied the right way
# round; a correct build is within 3.1e-06 and the break moves it by 1.2e-02.
_EMBEDDING_TOLERANCE = 1e-4
_LOGIT_TOLERANCE = 0.002
# Kept only to fail early and loudly on a gross error. It cannot separate a
# correct build from the GELU break above, which is why it is not the gate.
_COSINE_FLOOR = 0.999


def _native_embedding(binary: Path, runtime_root: Path, bundle: Path, case: dict,
                      *arguments: str) -> list[float]:
    payload = _run_json(binary, runtime_root, bundle, case, "embed", *arguments)
    values = [float(value) for value in payload["values"]]
    assert values, f"native {FAMILY} returned an empty embedding"
    assert payload["normalization"] == "l2"
    assert payload["embedding_space"] == "siglip"
    norm = math.sqrt(sum(value * value for value in values))
    assert abs(norm - 1.0) <= 1e-3, f"native {FAMILY} embedding is not unit length: {norm}"
    return values


def _reference(model_dir: Path, case: dict):
    """The reference's own numbers, computed the way the reference computes them.

    Note the processor is given no attention mask to pass on and the model is
    not given one either: measured, an all-ones mask changes the pooled text
    embedding by exactly 0, and the reference's processor does not emit one.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    # use_fast=False is required, not a preference. The fast processor resizes
    # with torchvision and the slow one with Pillow, and measured on this image
    # the two disagree by 7.8e-03 in pixel_values - ten times the 8.4e-04 that
    # the weakest structural break is worth. This family reproduces Pillow's
    # bicubic coefficient for coefficient, so comparing against the fast path
    # would bury the signal this test exists to find.
    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=False)
    texts = [str(case["query"]), str(case["negative_query"])]
    batch = processor(text=texts, images=image, padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt")

    def pooled(output):
        return output if torch.is_tensor(output) else output.pooler_output

    with torch.no_grad():
        image_vector = pooled(model.get_image_features(pixel_values=batch["pixel_values"]))
        text_vectors = pooled(model.get_text_features(input_ids=batch["input_ids"]))
    scale = float(model.logit_scale.detach().exp())
    bias = float(model.logit_bias.detach())

    def unit(vector):
        return (vector / vector.norm()).tolist()

    return (unit(image_vector[0]), [unit(row) for row in text_vectors], scale, bias)


def _max_abs(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right), (
        f"{FAMILY} embedding width mismatch: native={len(left)} reference={len(right)}"
    )
    return max(abs(a - b) for a, b in zip(left, right))


def _logit(probability: float) -> float:
    """Undo the runtime's sigmoid, so the comparison is in the linear space."""
    assert 0.0 < probability < 1.0, f"{FAMILY} probability is saturated: {probability}"
    return math.log(probability / (1.0 - probability))


def _cosine(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right), (
        f"{FAMILY} embedding width mismatch: native={len(left)} reference={len(right)}"
    )
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    assert left_norm > 0.0 and right_norm > 0.0
    return dot / (left_norm * right_norm)


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    # A quantized case states its own tolerances in the manifest: the defaults above
    # are sized to catch structural errors in a float32 build and are far tighter than
    # FP8 rounding. A case without them keeps the defaults.
    embedding_tolerance = float(case.get("embedding_tolerance", _EMBEDDING_TOLERANCE))
    logit_tolerance = float(case.get("logit_tolerance", _LOGIT_TOLERANCE))
    cosine_floor = float(case.get("cosine_floor", _COSINE_FLOOR))
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    reference_image, reference_texts, scale, bias = _reference(model_dir, case)

    native_image = _native_embedding(binary, runtime_root, bundle, case, "--task",
                                     "image_to_embedding", "--image",
                                     str(_asset(case["test_image"])))
    image_delta = _max_abs(native_image, reference_image)
    image_cosine = _cosine(native_image, reference_image)
    assert image_cosine >= cosine_floor, f"image embedding cosine {image_cosine}"
    assert image_delta <= embedding_tolerance, (
        f"image embedding differs from the reference by {image_delta}"
    )

    text_deltas = []
    for text, expected in zip((case["query"], case["negative_query"]), reference_texts):
        native = _native_embedding(binary, runtime_root, bundle, case, "--task",
                                   "text_to_embedding", "--text", str(text))
        cosine = _cosine(native, expected)
        assert cosine >= cosine_floor, f"text embedding cosine {cosine} for {text!r}"
        delta = _max_abs(native, expected)
        assert delta <= embedding_tolerance, (
            f"text embedding for {text!r} differs from the reference by {delta}"
        )
        text_deltas.append(delta)

    # SigLIP scores a pair on its own: the sigmoid is calibrated by the
    # checkpoint's own scale and bias and needs no other candidate, so the
    # runtime returns a probability rather than a bare logit. It is compared
    # in logit space because that is where the sensitivity is uniform.
    scores = []
    logit_deltas = []
    for text, expected_text in zip((case["query"], case["negative_query"]), reference_texts):
        payload = _run_json(binary, runtime_root, bundle, case, "rerank", "--task",
                            "text_image_to_relevance", "--query", str(text), "--image",
                            str(_asset(case["test_image"])))
        assert payload["score_kind"] == "probability"
        score = float(payload["score"])
        assert 0.0 <= score <= 1.0, f"{FAMILY} returned a score outside [0, 1]: {score}"
        expected_logit = scale * sum(a * b for a, b in zip(reference_image, expected_text)) + bias
        delta = abs(_logit(score) - expected_logit)
        assert delta <= logit_tolerance, (
            f"relevance logit for {text!r} differs from the reference by {delta}"
        )
        logit_deltas.append(delta)
        scores.append(score)
    assert scores[0] > scores[1], (
        f"{FAMILY} scored the negative text higher: {scores[0]} <= {scores[1]}"
    )
    print(
        f"{FAMILY} parity: image maxabs {image_delta:.3e}, "
        f"text maxabs {text_deltas[0]:.3e}/{text_deltas[1]:.3e}, "
        f"logit maxabs {max(logit_deltas):.3e}, "
        f"probabilities {scores[0]:.8f} vs {scores[1]:.8f}"
    )
