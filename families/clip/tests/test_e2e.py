# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and official-reference E2E for clip.

One bundle serves all three tasks, so the native side is exercised three
times through ``--task``: the image tower alone, the text tower alone, and
the scored pair. Each is compared against the reference it corresponds to.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "clip"
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




_COSINE_FLOOR = 0.9995
_SCORE_TOLERANCE = 0.1


def _native_embedding(binary: Path, runtime_root: Path, bundle: Path, case: dict,
                      *arguments: str) -> list[float]:
    payload = _run_json(binary, runtime_root, bundle, case, "embed", *arguments)
    values = [float(value) for value in payload["values"]]
    assert values, f"native {FAMILY} returned an empty embedding"
    assert payload["normalization"] == "l2"
    assert payload["embedding_space"] == "clip"
    norm = math.sqrt(sum(value * value for value in values))
    assert abs(norm - 1.0) <= 1e-3, f"native {FAMILY} embedding is not unit length: {norm}"
    return values


def _reference(model_dir: Path, case: dict):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    image = Image.open(_asset(case["test_image"])).convert("RGB")
    model = CLIPModel.from_pretrained(str(model_dir)).eval()
    processor = CLIPProcessor.from_pretrained(str(model_dir))
    texts = [str(case["query"]), str(case["negative_query"])]
    batch = processor(text=texts, images=image, padding="max_length", max_length=77,
                      return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=batch["pixel_values"])
        text_features = model.get_text_features(input_ids=batch["input_ids"],
                                                attention_mask=batch["attention_mask"])
    image_vector = getattr(image_features, "image_embeds", None)
    if image_vector is None:
        image_vector = image_features.pooler_output
    text_vectors = getattr(text_features, "text_embeds", None)
    if text_vectors is None:
        text_vectors = text_features.pooler_output
    scale = float(model.logit_scale.detach().exp())

    def unit(vector):
        return (vector / vector.norm()).tolist()

    return (unit(image_vector[0]), [unit(row) for row in text_vectors], scale)


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
    model_dir = _model_dir(manifest)
    binary, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)

    reference_image, reference_texts, scale = _reference(model_dir, case)

    native_image = _native_embedding(binary, runtime_root, bundle, case, "--task",
                                     "image_to_embedding", "--image",
                                     str(_asset(case["test_image"])))
    image_cosine = _cosine(native_image, reference_image)
    assert image_cosine >= _COSINE_FLOOR, f"image embedding cosine {image_cosine}"

    text_cosines = []
    native_texts = []
    for text, expected in zip((case["query"], case["negative_query"]), reference_texts):
        native = _native_embedding(binary, runtime_root, bundle, case, "--task",
                                   "text_to_embedding", "--text", str(text))
        native_texts.append(native)
        cosine = _cosine(native, expected)
        assert cosine >= _COSINE_FLOOR, f"text embedding cosine {cosine} for {text!r}"
        text_cosines.append(cosine)

    # The scored pair must agree with the reference in value, and the query
    # must beat the negative one: a pair of embeddings can each be close and
    # still rank the wrong way round, so the ordering is checked separately.
    scores = []
    for text, expected_text in zip((case["query"], case["negative_query"]), reference_texts):
        payload = _run_json(binary, runtime_root, bundle, case, "rerank", "--task",
                            "text_image_to_relevance", "--query", str(text), "--image",
                            str(_asset(case["test_image"])))
        assert payload["score_kind"] == "logit"
        score = float(payload["score"])
        expected_score = scale * sum(a * b for a, b in zip(reference_image, expected_text))
        assert abs(score - expected_score) <= _SCORE_TOLERANCE, (
            f"relevance {score} differs from reference {expected_score} for {text!r}"
        )
        scores.append(score)
    assert scores[0] > scores[1], (
        f"{FAMILY} scored the negative text higher: {scores[0]} <= {scores[1]}"
    )
    print(
        f"{FAMILY} parity: image cosine {image_cosine:.6f}, "
        f"text cosines {text_cosines[0]:.6f}/{text_cosines[1]:.6f}, "
        f"scores {scores[0]:.3f} vs {scores[1]:.3f}"
    )
