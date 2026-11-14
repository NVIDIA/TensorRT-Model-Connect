# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime and official-reference E2E for SigLIP 2 NaFlex.

One bundle serves all three tasks, so the native side is exercised through ``--task``:
the image tower alone on three aspect ratios (a landscape photo, a portrait crop and a
small crop the processor upscales), the text tower alone on English, mixed-case,
French, Chinese and an over-long prompt, and the scored pair.

Two oracles, in this order:

* **host preprocessing, exact.** The runtime's NaFlex resize, patch cut and
  normalisation must reproduce the reference processor value for value; its position
  taps must reproduce the reference's table resampling to float32 rounding. Checked
  through ``siglip2_preprocess_dump`` from the native build, before any engine runs.
* **embeddings and the relevance logit**, against the reference model in float32 on the
  CPU. The gate is the embedding, where an error originates; see the constants below
  for how the tolerances were set.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "siglip2"
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
        model_match = (not model_filters or FAMILY in model_filters or name in model_filters
                       or manifest["name"] in model_filters)
        testcase_match = not testcase_filters or name in testcase_filters
        if model_match and testcase_match:
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" in metafunc.fixturenames:
        names, enabled = _selected_cases(metafunc.config)
        parameters = names
        if not enabled:
            parameters = [
                pytest.param(name, marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"))
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
        snapshot = snapshot_download(repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"),
                                     local_files_only=True)
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return Path(snapshot)


def _runtime() -> tuple[Path, Path, Path]:
    binary = _required_path(os.environ.get("TRTMC_BINARY"), "TRTMC_BINARY")
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / f"libtrtmc_model_{FAMILY}.so").is_file()
    # CI builds a family's CTests and the runtime by name, not this E2E-only tool.
    subprocess.run(["cmake", "--build", str(native), "--target", "siglip2_preprocess_dump"],
                   check=True, timeout=600)
    dump = native / "siglip2_preprocess_dump"
    assert dump.is_file(), f"selected {FAMILY} E2E requires the native test tool {dump}"
    import torch

    assert torch.cuda.is_available(), f"selected {FAMILY} E2E requires CUDA"
    return binary, runtime_root, dump


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


def _run_json(binary: Path, runtime_root: Path, bundle: Path, command: str, *arguments: str) -> dict:
    invocation = [str(binary), command, str(bundle), "--runtime-root", str(runtime_root), *arguments]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value)
    completed = subprocess.run(invocation, check=True, capture_output=True, text=True, env=env,
                               timeout=3600)
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


def _images(case: dict, tmp_path: Path) -> list[Path]:
    """The test photograph and the crops the case names, as lossless PNG files."""
    from PIL import Image

    source = TEST_ROOT / case["test_image"]
    assert source.is_file(), f"selected {FAMILY} E2E asset does not exist: {source}"
    paths = [source]
    photo = Image.open(source).convert("RGB")
    for index, box in enumerate(case.get("image_crops", [])):
        path = tmp_path / f"crop_{index}.png"
        photo.crop(tuple(int(value) for value in box)).save(path)
        paths.append(path)
    return paths


# Host preprocessing: the pixels are exact by construction (Pillow's fixed-point resize
# and numpy's rescale/normalise arithmetic are reproduced), so any difference is a
# defect. The position taps are built in float64 where the reference builds them in
# float32; the resampled table differs by at most 1.7e-05 on the test images.
_POSITION_TOLERANCE = 1e-4


def _check_preprocessing(dump: Path, model_dir: Path, image_path: Path, tmp_path: Path) -> list[int]:
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoImageProcessor
    from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionEmbeddings

    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    reference = processor(images=[image], return_tensors="np", input_data_format="channels_last")
    grid_height, grid_width = (int(value) for value in reference["spatial_shapes"][0])
    count = grid_height * grid_width
    expected = reference["pixel_values"][0, :count].astype(np.float32)

    with safe_open(str(model_dir / "model.safetensors"), "np") as handle:
        table = handle.get_tensor("vision_model.embeddings.position_embedding.weight")
    grid = int(round(table.shape[0] ** 0.5))

    raw = tmp_path / f"{image_path.stem}.u8"
    image.tofile(raw)
    out = tmp_path / f"{image_path.stem}.bin"
    subprocess.run([str(dump), str(raw), str(height), str(width), str(processor.patch_size),
                    str(processor.max_num_patches), str(grid), *map(str, processor.image_mean),
                    *map(str, processor.image_std), str(out)], check=True, timeout=600)
    data = out.read_bytes()
    native_grid = [int(value) for value in np.frombuffer(data[:8], dtype=np.int32)]
    assert native_grid == [grid_height, grid_width], (
        f"{image_path.name}: native grid {native_grid}, reference {[grid_height, grid_width]}")
    values = np.frombuffer(data[8:], dtype=np.float32)
    pixels = values[: expected.size].reshape(expected.shape)
    mismatched = int((pixels != expected).sum())
    assert mismatched == 0, f"{image_path.name}: {mismatched} pixel values differ from the processor"

    taps = values[expected.size:].reshape(count, grid * grid).astype(np.float64)
    resampled = Siglip2VisionEmbeddings.resize_positional_embeddings(
        torch.from_numpy(table).reshape(grid, grid, -1), torch.tensor([[grid_height, grid_width]]),
        max_length=count)[0].double().numpy()
    delta = float(np.abs(taps @ table.astype(np.float64) - resampled).max())
    assert delta <= _POSITION_TOLERANCE, f"{image_path.name}: position taps differ by {delta}"
    return [grid_height, grid_width]


def _reference(model_dir: Path, images: list[Path], texts: list[str]):
    """The reference's own embeddings, in float32 on the CPU, the way it computes them.

    The processor is the Pillow one (``use_fast=False``), which the runtime reproduces
    exactly; the text is padded to the 64-token window with truncation, as the
    checkpoint's processor does by default.
    """
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=False)

    def pooled(output):
        return output if torch.is_tensor(output) else output.pooler_output

    def unit(vector):
        return (vector / vector.norm()).tolist()

    image_vectors = []
    with torch.no_grad():
        for path in images:
            batch = processor(images=[Image.open(path).convert("RGB")], return_tensors="pt")
            vector = pooled(model.get_image_features(**batch))[0]
            image_vectors.append(unit(vector))
        batch = processor(text=texts, padding="max_length", max_length=64, truncation=True,
                          return_tensors="pt")
        text_vectors = [unit(row) for row in pooled(model.get_text_features(**batch))]
    scale = float(model.logit_scale.detach().exp())
    bias = float(model.logit_bias.detach())
    return image_vectors, text_vectors, scale, bias


# Gates, measured on this case on a B300 (family README): a correct float32 build is
# within 6.4e-07 of the reference on every embedding and 1.3e-05 on the logit. The
# weakest deliberate break, the exact GELU in place of the tanh one, moves a text
# embedding by 4.5e-04, an image embedding by 1.1e-03 and the logit by 2.3e-02; feeding
# the padded patch budget without a mask moves the image embedding by 5.5e-03. The
# embedding is the gate: it is where an error originates and its sensitivity does not
# depend on where a pair lands on the sigmoid. The logit confirms that the scale and
# bias are applied the right way round. The cosine floor only fails early and loudly
# on a gross error (the unmasked padding still has cosine 0.9993).
_EMBEDDING_TOLERANCE = 1e-4
_LOGIT_TOLERANCE = 0.002
_COSINE_FLOOR = 0.999


def _max_abs(left: list[float], right: list[float]) -> float:
    assert len(left) == len(right), f"{FAMILY} embedding width mismatch"
    return max(abs(a - b) for a, b in zip(left, right))


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


def _native_embedding(binary: Path, runtime_root: Path, bundle: Path, *arguments: str) -> list[float]:
    payload = _run_json(binary, runtime_root, bundle, "embed", *arguments)
    values = [float(value) for value in payload["values"]]
    assert values, f"native {FAMILY} returned an empty embedding"
    assert payload["normalization"] == "l2"
    assert payload["embedding_space"] == "siglip2"
    norm = math.sqrt(sum(value * value for value in values))
    assert abs(norm - 1.0) <= 1e-3, f"native {FAMILY} embedding is not unit length: {norm}"
    return values


def _logit(probability: float) -> float:
    assert 0.0 < probability < 1.0, f"{FAMILY} probability is saturated: {probability}"
    return math.log(probability / (1.0 - probability))


def _record_comparison(tmp_path: Path, native: dict, reference: dict, checks: list[dict]) -> None:
    """Keep both sides and the family's checks in the E2E evidence, when it is recorded."""
    if not evidence_enabled():
        return
    native_path = tmp_path / "siglip2-native.json"
    reference_path = tmp_path / "siglip2-reference.json"
    native_path.write_text(json.dumps(native) + "\n", encoding="utf-8")
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    record_evidence(
        "reference_comparison",
        {
            "label": "Embeddings and relevance logits against the official float32 model",
            "scope": "independent_reference",
            "enforced": True,
            "native": native_path,
            "reference": reference_path,
            "checks": checks,
        },
    )


def _check(name: str, label: str, actual: float, limit: float) -> dict:
    return {"name": name, "label": label, "scope": "independent_reference", "actual": actual,
            "operator": "<=", "expected": limit, "passed": actual <= limit}


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest.get("hf_id"),
                                   "hf_revision": manifest.get("hf_revision")})
    binary, runtime_root, dump = _runtime()

    images = _images(case, tmp_path)
    with evidence_stage("preprocess"):
        grids = [_check_preprocessing(dump, model_dir, path, tmp_path) for path in images]
    assert len({tuple(grid) for grid in grids}) == len(grids), f"image grids repeat: {grids}"
    record_evidence("grids", grids)

    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)

    texts = [str(case["query"]), str(case["negative_query"]), *map(str, case.get("texts", []))]
    with evidence_stage("reference"):
        reference_images, reference_texts, scale, bias = _reference(model_dir, images, texts)

    with evidence_stage("native"):
        native_images = [
            _native_embedding(binary, runtime_root, bundle, "--task", "image_to_embedding",
                              "--image", str(path))
            for path in images
        ]
        native_texts = [
            _native_embedding(binary, runtime_root, bundle, "--task", "text_to_embedding",
                              "--text", text)
            for text in texts
        ]
        # A per-pair sigmoid calibrated by the checkpoint's own scale and bias: compared
        # in logit space, where the sensitivity is uniform.
        scores = []
        for text in texts[:2]:
            payload = _run_json(binary, runtime_root, bundle, "rerank", "--task",
                                "text_image_to_relevance", "--query", text, "--image",
                                str(images[0]))
            assert payload["score_kind"] == "probability"
            scores.append(float(payload["score"]))

    with evidence_stage("compare"):
        image_deltas = [_max_abs(n, r) for n, r in zip(native_images, reference_images)]
        text_deltas = [_max_abs(n, r) for n, r in zip(native_texts, reference_texts)]
        expected_logits = [
            scale * sum(a * b for a, b in zip(reference_images[0], vector)) + bias
            for vector in reference_texts[:2]
        ]
        logit_deltas = [abs(_logit(score) - expected)
                        for score, expected in zip(scores, expected_logits)]
        checks = [_check(f"image_{index}_maxabs", f"Image {path.name} embedding max |diff|",
                         delta, _EMBEDDING_TOLERANCE)
                  for index, (path, delta) in enumerate(zip(images, image_deltas))]
        checks += [_check(f"text_{index}_maxabs", f"Text {index} embedding max |diff|", delta,
                          _EMBEDDING_TOLERANCE) for index, delta in enumerate(text_deltas)]
        checks += [_check(f"logit_{index}_absdiff", f"Relevance logit {index} |diff|", delta,
                          _LOGIT_TOLERANCE) for index, delta in enumerate(logit_deltas)]
        _record_comparison(
            tmp_path,
            {"images": native_images, "texts": native_texts, "probabilities": scores},
            {"images": reference_images, "texts": reference_texts, "logits": expected_logits,
             "logit_scale": scale, "logit_bias": bias},
            checks,
        )
        for path, native, expected, delta in zip(images, native_images, reference_images,
                                                 image_deltas):
            cosine = _cosine(native, expected)
            assert cosine >= _COSINE_FLOOR, f"image embedding cosine {cosine} for {path.name}"
            assert delta <= _EMBEDDING_TOLERANCE, (
                f"image embedding of {path.name} differs from the reference by {delta}")
        for text, native, expected, delta in zip(texts, native_texts, reference_texts,
                                                 text_deltas):
            cosine = _cosine(native, expected)
            assert cosine >= _COSINE_FLOOR, f"text embedding cosine {cosine} for {text!r}"
            assert delta <= _EMBEDDING_TOLERANCE, (
                f"text embedding for {text!r} differs from the reference by {delta}")
        for text, score, delta in zip(texts, scores, logit_deltas):
            assert 0.0 <= score <= 1.0, f"{FAMILY} returned a score outside [0, 1]: {score}"
            assert delta <= _LOGIT_TOLERANCE, f"relevance logit for {text!r} differs by {delta}"
        assert scores[0] > scores[1], f"{FAMILY} scored the negative text higher: {scores}"
    print(
        f"{FAMILY} parity: grids {grids}, image maxabs "
        + ", ".join(f"{value:.3e}" for value in image_deltas)
        + ", text maxabs " + ", ".join(f"{value:.3e}" for value in text_deltas)
        + f", logit maxabs {max(logit_deltas):.3e}, probabilities {scores[0]:.8f} vs {scores[1]:.8f}"
    )
