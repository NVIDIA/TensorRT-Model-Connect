# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full checkpoint native/reference image-language generation parity.

The normalized-edit-distance limit of 0.15 is the existing Qwen-VL oracle.
GPU component tests separately gate routed experts and DeepStack math.
"""

import json
import gc
import os
from pathlib import Path
import re
import string
import subprocess
import pytest
from tensorrt_model_connect import BuildRequest, build

FAMILY = "qwen3_vl_moe"
ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifests/qwen3-vl-30b-a3b-instruct.json").read_text())
CASES = {case["name"]: case for case in MANIFEST["testcases"]}


def _edit_distance(left: str, right: str) -> float:
    a = " ".join(left.lower().split())
    b = " ".join(right.lower().split())
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, start=1):
        current = [index]
        for offset, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b)
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b), 1)


def _canonical_vl_answers(left: str, right: str) -> tuple[str, str]:
    def normalize(value: str) -> str:
        return " ".join(value.casefold().split()).strip(string.punctuation + string.whitespace)

    left = normalize(left)
    right = normalize(right)
    left_words = re.findall(r"\b\w+\b", left)
    right_words = re.findall(r"\b\w+\b", right)
    if len(left_words) == 1 and len(right_words) > 1 and left_words[0] in right_words:
        return left_words[0], left_words[0]
    if len(right_words) == 1 and len(left_words) > 1 and right_words[0] in left_words:
        return right_words[0], right_words[0]
    return left, right


def pytest_generate_tests(metafunc):
    if "case_name" not in metafunc.fixturenames:
        return
    config = metafunc.config
    models = {v.strip() for raw in config.getoption("--e2e-model") for v in raw.split(",")}
    path = config.getoption("--e2e-models-file")
    if path:
        models.update(
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    cases = {v.strip() for raw in config.getoption("--e2e-testcase") for v in raw.split(",")}
    enabled = bool(models or cases)
    names = [
        name
        for name in CASES
        if (not models or models.intersection({FAMILY, MANIFEST["name"], name}))
        and (not cases or name in cases)
    ]
    metafunc.parametrize(
        "case_name",
        [
            name
            if enabled
            else pytest.param(
                name, marks=pytest.mark.skip(reason="select the Qwen3-VL MoE E2E family explicitly")
            )
            for name in names
        ],
    )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    import torch
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    assert torch.cuda.is_available(), "selected Qwen3-VL MoE E2E requires CUDA"
    directory = os.environ.get("TRTMC_QWEN3_VL_MOE_MODEL_DIR")
    if not directory:
        from huggingface_hub import snapshot_download

        directory = snapshot_download(
            MANIFEST["hf_id"], revision=MANIFEST["hf_revision"], local_files_only=True
        )
    directory = Path(directory)
    work = tmp_path_factory.mktemp("qwen3-vl-moe")
    bundle_path = os.environ.get("TRTMC_QWEN3_VL_MOE_BUNDLE")
    if bundle_path:
        bundle = Path(bundle_path)
        assert bundle.is_file(), "explicit Qwen3-VL MoE bundle is missing"
    else:
        bundle = work / MANIFEST["bundle"]
        build(
            BuildRequest(
                model_dir=directory,
                output_path=bundle,
                family=MANIFEST["family"],
                task=MANIFEST["task"],
                precision=MANIFEST["precision"],
                max_sequence_length=MANIFEST["max_sequence_length"],
                tensor_parallel_size=MANIFEST["tensor_parallel_size"],
            )
        )
    binary = Path(os.environ["TRTMC_BINARY"])
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    assert binary.is_file() and (runtime / "libtrtmc_model_qwen3_vl_moe.so").is_file()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[MANIFEST["reference_precision"]]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(
        directory,
        use_fast=False,
        min_pixels=MANIFEST["min_pixels"],
        max_pixels=MANIFEST["max_pixels"],
    )
    reference = (
        Qwen3VLMoeForConditionalGeneration.from_pretrained(
            directory, dtype=dtype, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    expected = {
        name: _reference_answer(name, case, work, processor, reference)
        for name, case in CASES.items()
    }
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    return bundle, binary, runtime, expected


def _reference_answer(case_name, case, work, processor, reference):
    import torch
    from PIL import Image, ImageOps

    original = ROOT / case["test_image"]
    image = Image.open(original).convert("RGB")
    if case["mirror"]:
        image = ImageOps.mirror(image)
    image_path = work / (case_name + ".png")
    image.save(image_path)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": case["prompt"]},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|image_pad|>" in prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    assert inputs.input_ids.shape[1] + case["max_new_tokens"] <= MANIFEST["max_sequence_length"]
    with torch.inference_mode():
        generated = reference.generate(
            **inputs, max_new_tokens=case["max_new_tokens"], do_sample=False, top_k=1
        )
    ids = generated[0, inputs.input_ids.shape[1] :].cpu().tolist()
    expected = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert expected.strip(), "official reference produced an empty answer"
    print(json.dumps({"reference_case": case_name, "text": expected, "ids": ids}), flush=True)
    return image_path, expected, ids, inputs.input_ids.shape[1]


def test_official_checkpoint_e2e(case_name, artifacts):
    bundle, binary, runtime, references = artifacts
    case = CASES[case_name]
    image_path, expected, ids, input_tokens = references[case_name]
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime),
        "--prompt",
        case["prompt"],
        "--image",
        str(image_path),
        "--max-new-tokens",
        str(case["max_new_tokens"]),
        "--temperature",
        "1",
        "--top-k",
        "1",
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime) + ":" + env.get("LD_LIBRARY_PATH", "")
    completed = subprocess.run(
        command, env=env, check=False, text=True, capture_output=True, timeout=900
    )
    image_path.with_suffix(".native.stdout").write_text(completed.stdout)
    image_path.with_suffix(".native.stderr").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr[-6000:] + completed.stdout[-1000:]
    payloads = []
    for line in completed.stdout.splitlines():
        start = line.find("{")
        if start >= 0:
            try:
                value = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if "text" in value:
                payloads.append(value)
    assert len(payloads) == 1, completed.stdout[-1000:]
    actual = payloads[0]
    assert actual["text"].strip() and actual["token_ids"]
    canonical_actual, canonical_expected = _canonical_vl_answers(actual["text"], expected)
    distance = _edit_distance(canonical_actual, canonical_expected)
    print(
        json.dumps(
            {
                "case": case_name,
                "reference_text": expected,
                "native_text": actual["text"],
                "reference_ids": ids,
                "native_ids": actual["token_ids"],
                "normalized_edit_distance": distance,
                "input_tokens": input_tokens,
            }
        ),
        flush=True,
    )
    assert distance <= 0.15


def test_parity_oracle_rejects_wrong_color():
    actual, expected = _canonical_vl_answers("Red.", "Gray")
    assert _edit_distance(actual, expected) > 0.15
