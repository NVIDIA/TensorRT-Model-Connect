# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned LayoutLMv3 native document preprocessing, word mapping and joint features."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "layoutlmv3"
TASKS = frozenset({"document_words_to_token_features"})
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


@cache
def _native_test_binary(name: str) -> Path:
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


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
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 3600)),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
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


def _bundle_section(path, name):
    import struct

    with path.open("rb") as file:
        assert file.read(8) == b"BUNDLE\x01\x00"
        length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(length))
        section = header["sections"][name]
        file.seek(16 + length + section["offset"])
        return file.read(section["length"])


def test_official_checkpoint_e2e(case_name, tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import LayoutLMv3ImageProcessor, LayoutLMv3Model, LayoutLMv3Tokenizer

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    config = json.loads(_bundle_section(bundle, "runtime.json"))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_bytes(_bundle_section(bundle, "tokenizer.json"))
    tokenizer = LayoutLMv3Tokenizer.from_pretrained(directory)
    processor = LayoutLMv3ImageProcessor.from_pretrained(directory, apply_ocr=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = LayoutLMv3Model.from_pretrained(directory, dtype=torch.float32).eval().to("cuda")
    original = Image.open(TEST_ROOT / "data" / case["image"]).convert("RGB")
    documents = [
        (
            original,
            ["Tesla", "Model", "3"],
            [(0.1, 0.22, 0.32, 0.3), (0.32, 0.22, 0.55, 0.3), (0.55, 0.22, 0.62, 0.3)],
        ),
        (
            original.transpose(Image.Transpose.ROTATE_90),
            ["Café", "Zürich", "Invoice", "123"],
            [
                (0.12, 0.4, 0.3, 0.5),
                (0.35, 0.4, 0.7, 0.5),
                (0.12, 0.6, 0.5, 0.72),
                (0.6, 0.6, 0.9, 0.72),
            ],
        ),
    ]
    length = manifest["max_sequence_length"]
    total = length + 197
    for index, (image, words, boxes) in enumerate(documents):
        pixels = np.asarray(image, dtype=np.float32) / 255
        source = tmp_path / f"image-{index}.f32"
        source.write_bytes(pixels.astype("<f4").tobytes())
        # Public boxes are float32 original-image coordinates. Normalize those
        # exact values for the official tokenizer, without invoking OCR.
        pixel_boxes = [
            [
                float(np.float32(value * (image.width if axis % 2 == 0 else image.height)))
                for axis, value in enumerate(box)
            ]
            for box in boxes
        ]
        normalized = [
            [
                int(1000 * value / (image.width if axis % 2 == 0 else image.height))
                for axis, value in enumerate(box)
            ]
            for box in pixel_boxes
        ]
        text = tokenizer(
            words, boxes=normalized, padding="max_length", max_length=length, return_tensors="pt"
        )
        visual = processor(
            images=pixels * 255, input_data_format="channels_last", return_tensors="pt"
        ).pixel_values
        request = dict(
            height=image.height,
            width=image.width,
            image_path=str(source),
            words=[dict(text=word, box=box) for word, box in zip(words, pixel_boxes)],
        )
        native_pixels = tmp_path / f"pixels-{index}.f32"
        rx = tmp_path / f"rx-{index}.i32"
        ry = tmp_path / f"ry-{index}.i32"
        completed = subprocess.run(
            [str(_native_test_binary("test_layoutlmv3_tokenizer"))],
            input=json.dumps(
                dict(
                    **request,
                    tokenizer_path=str(tokenizer_path),
                    config=config,
                    pixels_path=str(native_pixels),
                    relative_x_path=str(rx),
                    relative_y_path=str(ry),
                ),
                ensure_ascii=False,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        encoded = json.loads(completed.stdout)
        assert encoded["input_ids"] == text.input_ids[0].tolist()
        assert encoded["attention_mask"] == text.attention_mask[0].tolist()
        assert encoded["bbox"] == text.bbox[0].reshape(-1).tolist()
        word_indices = [
            word if word is not None else (-1 if text.attention_mask[0, i] else -2)
            for i, word in enumerate(text.word_ids())
        ]
        assert encoded["word_indices"] == word_indices
        expected_pixels = visual[0].numpy()
        actual_pixels = np.fromfile(native_pixels, dtype="<f4").reshape(3, 224, 224)
        np.testing.assert_allclose(actual_pixels, expected_pixels, atol=case["pixel_atol"], rtol=0)
        all_boxes = torch.cat([text.bbox[0], reference.visual_bbox.cpu()], dim=0)
        for path, axis in [(rx, 0), (ry, 3)]:
            coordinates = all_boxes[:, axis].cuda()
            expected = (
                reference.encoder.relative_position_bucket(
                    coordinates[None, :] - coordinates[:, None],
                    num_buckets=reference.config.rel_2d_pos_bins,
                    max_distance=reference.config.max_rel_2d_pos,
                )
                .cpu()
                .numpy()
            )
            assert np.array_equal(np.fromfile(path, dtype="<i4").reshape(total, total), expected)
        with torch.no_grad():
            expected = (
                reference(**text.to("cuda"), pixel_values=visual.to("cuda"))
                .last_hidden_state[0, :length]
                .cpu()
                .numpy()
            )
        target = tmp_path / f"features-{index}.f32"
        completed = subprocess.run(
            [str(_native_test_binary("test_layoutlmv3_runtime")), str(bundle), str(runtime_root)],
            input=json.dumps(dict(**request, output_path=str(target)), ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        payload = json.loads(completed.stdout.splitlines()[-1])
        assert payload == dict(
            rows=length,
            columns=768,
            source_words=len(words),
            tokens=[
                [token, word_indices[i], i] for i, token in enumerate(text.input_ids[0].tolist())
            ],
        )
        actual = np.fromfile(target, dtype="<f4").reshape(length, 768)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        maximum = float(np.max(np.abs(actual - expected)))
        cosine = np.sum(actual.astype(np.float64) * expected, axis=1) / (
            np.linalg.norm(actual.astype(np.float64), axis=1)
            * np.linalg.norm(expected.astype(np.float64), axis=1)
        )
        assert maximum <= case["max_abs_error"]
        assert float(cosine.min()) >= case["min_cosine"]
        print(f"LayoutLMv3 document={index}: max_abs={maximum:.8g}, min_cosine={cosine.min():.9f}")
