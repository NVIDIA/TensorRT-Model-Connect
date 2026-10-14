# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned-checkpoint proof through the native semantic image/text Task interface."""

from __future__ import annotations

import json
import ast
import os
from pathlib import Path
import subprocess
import re
import urllib.request

import pytest

from tools.e2e_evidence import evidence_stage, record_evidence

from .semantic_contract import compare_semantics
from .native_probes import build_native_probe

MANIFEST = json.loads((Path(__file__).parent / "manifests/lfm25-vl-1.6b.json").read_text())
CHECKPOINT = MANIFEST["hf_id"]
REVISION = MANIFEST["hf_revision"]
CONTRACT = MANIFEST["testcases"][0]
CASE = CONTRACT["name"]
CARD_IMAGE = (
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
)


def _selected(config):
    models = config.getoption("--e2e-model", default=[]) or []
    cases = config.getoption("--e2e-testcase", default=[]) or []
    model_set = {part for value in models for part in value.split(",")}
    case_set = {part for value in cases for part in value.split(",")}
    if not (os.environ.get("TRTMC_E2E") == "1" or model_set or case_set):
        pytest.skip("native model-card E2E requires an explicit selection")
    if case_set and CASE not in case_set:
        pytest.skip("model-card case was not selected")
    if model_set and not model_set.intersection({"lfm2_vl", "lfm25-vl-1.6b", CASE}):
        pytest.skip("LFM2-VL was not selected")


def _fixtures(directory, processor):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    photo = directory / "model-card.jpg"
    with urllib.request.urlopen(CARD_IMAGE, timeout=60) as response:
        photo.write_bytes(response.read())
    record_evidence("inputs", {"model_card_image": {"source_url": CARD_IMAGE, "file": photo}})
    ocr = Image.new("RGB", (1024, 768), "white")
    draw = ImageDraw.Draw(ocr)
    font = ImageFont.load_default(size=52)
    for y, text in [
        (100, "CABIN STATUS"),
        (230, "Temperature: 22 C"),
        (360, "Seat belts: FASTENED"),
    ]:
        draw.text((64, y), text, fill="black", font=font)
    images = {
        "photo": Image.open(photo).convert("RGB"),
        "ocr": ocr,
        "red": Image.new("RGB", (256, 256), "red"),
        "blue": Image.new("RGB", (256, 256), "blue"),
        "wide": Image.new("RGB", (1024, 128), "green"),
    }
    metadata = {}
    for name, image in images.items():
        # Both paths use this one decoded RGB image: native reads the lossless
        # Float32 file, while the reference processor receives the PIL object.
        path = directory / (name + ".f32")
        np.asarray(image, dtype=np.float32).tofile(path)
        metadata[name] = {"path": str(path), "height": image.height, "width": image.width}
    cases = []

    def add(name, prompt, media, **options):
        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "images": [metadata[key] for key in media],
                "media_names": media,
                "max_new_tokens": CONTRACT["max_new_tokens"],
                "do_sample": CONTRACT["do_sample"],
                **options,
            }
        )

    add("model-card", "<image>What is in this image?", ["photo"])
    add(
        "recommended-sampling",
        "<image>What is in this image?",
        ["photo"],
        do_sample=True,
        temperature=0.1,
        top_k=50,
        min_p=0.15,
        repetition_penalty=1.05,
        seed=42,
    )
    add(
        "multi-image",
        "<image><image>What color is the first image and the second image?",
        ["red", "blue"],
    )
    image_turns = [
        {"role": "user", "content": "<image>What color is this image?"},
        {"role": "assistant", "content": "The image is red."},
        {"role": "user", "content": "<image>How did the color change from the first image?"},
    ]
    add(
        "image-conversation",
        processor.apply_chat_template(image_turns, add_generation_prompt=True, tokenize=False),
        ["red", "blue"],
        chat=False,
        semantic_messages=image_turns,
    )
    add("ocr", "<image>Read the text in this image.", ["ocr"])
    add("aspect-ratio", "<image>What color is this image?", ["wide"])
    add(
        "image-budget",
        "<image>Read the text in this image.",
        ["ocr"],
        max_image_tokens=128,
        max_tiles=4,
    )
    add("splitting-disabled", "<image>What is in this image?", ["photo"], split_images=0)
    questions = {
        "en": "What color is this image?",
        "ar": "ما لون هذه الصورة؟",
        "zh": "这张图片是什么颜色？",
        "fr": "De quelle couleur est cette image ?",
        "de": "Welche Farbe hat dieses Bild?",
        "ja": "この画像は何色ですか？",
        "ko": "이 이미지는 무슨 색인가요?",
        "es": "¿De qué color es esta imagen?",
    }
    for language, question in questions.items():
        add(
            "language-" + language,
            "<image>" + question,
            ["red"],
            image_format="float32" if language == "en" else "uint8",
        )
    add("text-only", "What is 3 + 5?", [])
    history = [
        {"role": "user", "content": "Say alpha."},
        {"role": "assistant", "reasoning": "old reasoning", "content": "Alpha."},
        {"role": "user", "content": "Say beta."},
        {"role": "assistant", "reasoning": "latest reasoning", "content": "Beta."},
        {"role": "user", "content": "What is 3 + 5?"},
    ]
    original_history = [
        {
            "role": message["role"],
            "content": (
                "<think>" + message["reasoning"] + "</think>" if "reasoning" in message else ""
            )
            + message["content"],
        }
        for message in history
    ]
    add(
        "reasoning-history",
        processor.tokenizer.apply_chat_template(
            original_history, add_generation_prompt=True, tokenize=False
        ),
        [],
        chat=False,
        semantic_messages=history,
    )
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    add(
        "tool-call",
        processor.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        ),
        [],
        chat=False,
        max_new_tokens=256,
        semantic_messages=messages[:],
        tools=tools,
    )
    messages += [
        {
            "role": "assistant",
            "content": '<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>',
        },
        {"role": "tool", "content": "Paris is sunny and 20 degrees Celsius."},
    ]
    add(
        "tool-result",
        processor.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        ),
        [],
        chat=False,
        semantic_messages=[
            messages[0],
            {
                "role": "assistant",
                "tool_call": {
                    "call_id": "call_0",
                    "name": "get_weather",
                    "arguments": {"location": "Paris"},
                },
            },
            {"role": "tool", "tool_result": messages[2]["content"], "call_id": "call_0"},
        ],
        tools=tools,
    )
    # Fill the actual advertised context. Count tokens with the original tokenizer.
    repetitions = 32740
    while True:
        text = " a" * repetitions + "\nReply with OK."
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        count = len(processor.tokenizer.encode(rendered, add_special_tokens=False))
        if count == 32760:
            break
        repetitions += 32760 - count
        assert repetitions > 0
    add("full-context", rendered, [], chat=False, max_new_tokens=8)
    return images, cases


@pytest.mark.parametrize("case_name", [CASE])
def test_e2e(case_name, tmp_path, request):
    _selected(request.config)
    record_evidence("inputs", {"case": case_name, "checkpoint": CHECKPOINT, "revision": REVISION})
    record_evidence(
        "thresholds",
        {
            "semantic_contract": "pinned task facts, language, tool fields, and context behavior",
            "exact_token_ids": "diagnostic",
            "exact_text": "diagnostic",
            "sampled_contract": "public Task response semantics under the recommended policy",
            "coupled_sampling": "64-step/EOS diagnostic on original-reference histories",
        },
    )
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from families.lfm2_vl.model import BuildRequest, build

    assert torch.cuda.is_available(), "selected model-card E2E requires CUDA"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    probe = build_native_probe("lfm2_vl_probe", runtime, tmp_path)
    checkpoint = Path(
        os.environ.get("TRTMC_LFM25_VL_CHECKPOINT")
        or snapshot_download(CHECKPOINT, revision=REVISION)
    )
    assert MANIFEST["tensor_parallel_size"] == 1, "LFM2-VL requires one runtime rank"
    assert CONTRACT["reference_backend"] == "hf_transformers"
    bundle = Path(os.environ.get("TRTMC_LFM25_VL_BUNDLE", str(tmp_path / MANIFEST["bundle"])))
    if not bundle.is_file():
        with evidence_stage("build"):
            writer = BundleWriter(bundle)
            try:
                build(
                    BuildRequest(
                        model_dir=checkpoint,
                        task=MANIFEST["task"],
                        precision=MANIFEST["precision"],
                        max_sequence_length=MANIFEST["max_sequence_length"],
                    ),
                    writer,
                )
                writer.finish()
            except BaseException:
                writer.abort()
                raise
    record_evidence(
        "checkpoint", {"model_dir": str(checkpoint), "hf_id": CHECKPOINT, "hf_revision": REVISION}
    )
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=MANIFEST["trust_remote_code"]
    )
    images, cases = _fixtures(tmp_path, processor)
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    with evidence_stage("native"):
        native = subprocess.run(
            [str(probe), str(bundle), str(runtime), str(cases_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    record_evidence("commands", {"argv": native.args})
    record_evidence("native", {"stdout": native.stdout, "stderr": native.stderr})
    actual = {case["name"]: case for case in map(json.loads, native.stdout.splitlines())}
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[CONTRACT["reference_precision"]],
        device_map="cuda",
        trust_remote_code=MANIFEST["trust_remote_code"],
    ).eval()
    evidence = []
    for case in cases:
        prompt = case["prompt"]
        if case.get("chat", True):
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
        options = {
            key: case[key]
            for key in ("min_image_tokens", "max_image_tokens", "max_tiles")
            if key in case
        }
        if "split_images" in case:
            options["do_image_splitting"] = bool(case["split_images"])
        inputs = processor(
            text=prompt,
            images=[images[name] for name in case["media_names"]] or None,
            return_tensors="pt",
            **options,
        ).to("cuda")
        with evidence_stage("reference"), torch.inference_mode():
            generation = {
                "max_new_tokens": case["max_new_tokens"],
                "do_sample": case.get("do_sample", False),
                "repetition_penalty": case.get("repetition_penalty", 1.0),
            }
            if generation["do_sample"]:
                torch.manual_seed(case["seed"])
                generation.update(
                    temperature=case["temperature"],
                    top_k=case["top_k"],
                    min_p=case["min_p"],
                )
            output = model.generate(
                **inputs,
                **generation,
            )
        expected = output[0, inputs.input_ids.shape[1] :].tolist()
        observed = actual[case["name"]]["tokens"]
        if not case["images"]:
            usage = actual[case["name"]]
            assert usage["input_tokens"] == inputs.input_ids.shape[1]
            assert usage["output_tokens"] == len(observed)
            assert usage["total_tokens"] == inputs.input_ids.shape[1] + len(observed)
        expected_text = processor.tokenizer.decode(
            expected, skip_special_tokens=False
        ).removesuffix("<|im_end|>")
        expected_calls = []
        if not case["images"]:
            for call_text in re.findall(
                r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", expected_text, re.DOTALL
            ):
                expression = ast.parse(call_text, mode="eval").body
                assert isinstance(expression, ast.List)
                for call in expression.elts:
                    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    expected_calls.append(
                        {
                            "name": call.func.id,
                            "arguments": {
                                value.arg: ast.literal_eval(value.value) for value in call.keywords
                            },
                            "state": 1,
                        }
                    )
            expected_text = re.sub(
                r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", expected_text, flags=re.DOTALL
            )
        calls = actual[case["name"]].get("calls", [])
        observed_calls = [
            {key: call[key] for key in ("name", "arguments", "state")} for call in calls
        ]
        call_ids = [call["call_id"] for call in calls]
        observed_text = actual[case["name"]]["text"]
        semantics = compare_semantics(
            case["name"],
            {
                "text": expected_text,
                "calls": expected_calls,
                "tokens": expected,
                "input_tokens": inputs.input_ids.shape[1],
            },
            {"text": observed_text, "calls": calls, "tokens": observed},
        )
        evidence.append(
            {
                "name": case["name"],
                "input_tokens": inputs.input_ids.shape[1],
                "reference_tokens": expected,
                "native_tokens": observed,
                "exact_match": observed == expected
                and observed_text == expected_text
                and observed_calls == expected_calls
                and all(call_ids)
                and len(set(call_ids)) == len(call_ids),
                "reference_text": expected_text,
                "native_text": observed_text,
                "reference_calls": expected_calls,
                "native_calls": calls,
                "semantic": semantics,
            }
        )
        (tmp_path / "parity.json").write_text(json.dumps(evidence, indent=2))
        (tmp_path / "semantic-parity.json").write_text(json.dumps(evidence, indent=2))
    from .sampled_parity import compare_sampled

    sampled_case = cases[0]
    sampled_prompt = processor.apply_chat_template(
        [{"role": "user", "content": sampled_case["prompt"]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    sampled_inputs = processor(
        text=sampled_prompt, images=[images["photo"]], return_tensors="pt"
    ).to("cuda")
    sampled = compare_sampled(
        model, sampled_inputs, sampled_case["images"], bundle, runtime, tmp_path
    )
    record_evidence("reference", {"greedy_cases": evidence, "sampled": sampled})
    with evidence_stage("compare"):
        assert all(item["semantic"]["equivalent"] for item in evidence), evidence
