# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned PaddleOCR pixel, token, multimodal position and generated-text parity."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "paddleocr_vl"
TASKS = frozenset({"images_text_to_text"})
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


def test_official_checkpoint_e2e(case_name, tmp_path):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    binary, runtime_root = _runtime()
    processor = AutoProcessor.from_pretrained(directory, use_fast=False)
    config = AutoConfig.from_pretrained(directory)
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(processor.tokenizer.backend_tokenizer.to_str())
    with torch.device("meta"):
        position_oracle = AutoModelForImageTextToText.from_config(
            config, attn_implementation="eager"
        )
    marker = "<|IMAGE_PLACEHOLDER|>"
    sentinel = "TRTMC_QUESTION_PLACEHOLDER"
    framed = processor.apply_chat_template(
        [dict(role="user", content=[dict(type="image"), dict(type="text", text=sentinel)])],
        add_generation_prompt=True,
    )
    prefix, rest = framed.split(marker)
    between, suffix = rest.split(sentinel)
    raw_config = json.loads((directory / "preprocessor_config.json").read_text())
    pages = []
    batches = []
    for index, content in enumerate(case["pages"]):
        page = Image.new("RGB", (manifest["image_width"], manifest["image_height"]), "white")
        ImageDraw.Draw(page).multiline_text(
            (20, 25), content, fill="black", font=ImageFont.load_default(size=32), spacing=14
        )
        path = tmp_path / f"page-{index}.png"
        page.save(path)
        pages.append(path)
        text = processor.apply_chat_template(
            [
                dict(
                    role="user",
                    content=[dict(type="image"), dict(type="text", text=case["question"])],
                )
            ],
            add_generation_prompt=True,
        )
        inputs = processor(text=text, images=page, return_tensors="pt")
        batches.append(inputs)
        source = tmp_path / f"source-{index}.f32"
        source.write_bytes((np.asarray(page, dtype=np.float32) / 255).astype("<f4").tobytes())
        pixels = tmp_path / f"pixels-{index}.f32"
        request = dict(
            source=str(source),
            output=str(pixels),
            height=page.height,
            width=page.width,
            profile_height=manifest["image_height"],
            profile_width=manifest["image_width"],
            profile_length=manifest["max_sequence_length"],
            min_pixels=raw_config["min_pixels"],
            max_pixels=raw_config["max_pixels"],
            tokenizer=str(tokenizer_json),
            prefix=prefix,
            between=between,
            suffix=suffix,
            question=case["question"],
            image_token=marker,
            image_id=config.image_token_id,
        )
        completed = subprocess.run(
            [str(runtime_root / "test_paddleocr_vl_preprocess")],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["ids"] == inputs["input_ids"][0].tolist()
        expected_positions, _ = position_oracle.model.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        actual_positions = np.array(payload["positions"]).reshape(
            3, manifest["max_sequence_length"]
        )
        np.testing.assert_array_equal(
            actual_positions[:, : inputs["input_ids"].shape[1]], expected_positions[:, 0].numpy()
        )
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        expected = (
            inputs["pixel_values"]
            .numpy()
            .reshape(gh, gw, 3, 14, 14)
            .transpose(2, 0, 3, 1, 4)
            .reshape(3, gh * 14, gw * 14)
        )
        actual = np.fromfile(pixels, dtype="<f4").reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    del position_oracle
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        AutoModelForImageTextToText.from_pretrained(
            directory, dtype=torch.float32, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    for inputs, path, expected_value in zip(batches, pages, case["expected_values"]):
        inputs = inputs.to("cuda")
        with torch.no_grad():
            generated = reference.generate(
                **inputs, do_sample=False, max_new_tokens=case["max_new_tokens"]
            )
        expected_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        expected_text = processor.batch_decode([expected_ids], skip_special_tokens=True)[0]
        payload = _run_json(
            binary,
            runtime_root,
            bundle,
            case,
            "run",
            "--task",
            manifest["task"],
            "--image",
            str(path),
            "--prompt",
            case["question"],
            "--max-new-tokens",
            str(case["max_new_tokens"]),
        )
        assert payload["token_ids"] == expected_ids
        assert payload["text"] == expected_text
        assert expected_value in payload["text"]
        print(f"PaddleOCR exact tokens={len(expected_ids)}, text={expected_text!r}")
