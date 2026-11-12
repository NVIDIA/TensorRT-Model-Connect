# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3n bundle through the public C/C++ SDK and CLI against Transformers.

Each case builds the manifest's bundle, runs a fixed set of conversations
(one image, a system prompt, two images, a multi-turn follow-up, float32
pixels) through the family-owned C++ SDK consumer, one of them through the C
SDK consumer and the ``trtmc`` CLI, and compares with the Transformers model in
FP32. The native prompt token ids of every conversation (rendered and tokenized
by the family's host code on the checkpoint's tokenizer) must equal the
reference processor's ids; the greedy generations must then agree:

* an FP32 bundle must reproduce the reference token ids exactly;
* a BF16 bundle is checked teacher-forced: at every generated position the
  FP32 reference must rank the native token within the case's logit margin of
  its own best token. The margin is the reference model's own BF16 error on
  the gap between its two best tokens on these conversations (0.70 logits for
  the E4B checkpoint): BF16 rounding may reorder candidates that close, and
  nothing farther apart.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

FAMILY = "gemma3n"
ROOT = Path(__file__).resolve().parent
CASES = {}
for _path in sorted((ROOT / "manifests").glob("*.json")):
    _manifest = json.loads(_path.read_text(encoding="utf-8"))
    for _case in _manifest["testcases"]:
        CASES[_case["name"]] = (_manifest, _case)

PROMPTS = {
    "describe": "Describe this image in one sentence.",
    "color": "What color is the car? Answer in one word.",
    "compare": "Are these two pictures the same scene? Answer briefly.",
    "first": "What is the main object in this picture?",
    "follow": "Where is it parked?",
}


def _selection(config, name):
    return {part.strip() for value in (config.getoption(name, default=[]) or [])
            for part in value.split(",") if part.strip()}


def _selected(config, manifest, case):
    models = _selection(config, "--e2e-model")
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        models.update(line.strip() for line in Path(path).read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
    cases = _selection(config, "--e2e-testcase")
    if not models and not cases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real family E2E requires explicit selection")
    if models and not models.intersection({FAMILY, manifest["name"], case["name"]}):
        pytest.skip("model profile was not selected")
    if cases and case["name"] not in cases:
        pytest.skip("testcase was not selected")


def _checkpoint(manifest) -> Path:
    explicit = os.environ.get("TRTMC_GEMMA3N_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "model.safetensors.index.json").is_file(), root
        return root
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        manifest["hf_id"], revision=manifest["hf_revision"],
        allow_patterns=["*.json", "*.jinja", "*.safetensors"]))


def _native_build_dir() -> Path:
    native = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    for target in ("test_gemma3n_sdk_cpp", "test_gemma3n_sdk_c", "test_gemma3n_processing"):
        subprocess.run(["cmake", "--build", str(native), "--parallel", "8", "--target", target],
                       check=True, timeout=1200)
    return native


def _images(directory: Path):
    """The test photographs as decoded RGB arrays, raw files and lossless PNGs."""
    from PIL import Image

    result = {}
    for name in ("test_img.jpeg", "test_img_car.jpeg"):
        image = Image.open(ROOT / "data" / name).convert("RGB")
        pixels = np.asarray(image)
        raw, png = directory / f"{name}.rgb", directory / f"{name}.png"
        pixels.tofile(raw)
        (pixels.astype(np.float32) / np.float32(255.0)).tofile(directory / f"{name}.f32")
        image.save(png)
        result[name] = {"image": image, "raw": raw, "png": png, "f32": directory / f"{name}.f32",
                        "height": pixels.shape[0], "width": pixels.shape[1]}
    return result


def _conversations(images, max_new_tokens):
    """(name, messages) with image parts as {"image": name} placeholders."""
    first, second = "test_img.jpeg", "test_img_car.jpeg"
    return [
        ("describe", [{"role": "user", "parts": [{"image": first}, {"text": PROMPTS["describe"]}]}]),
        ("system", [{"role": "system", "parts": [{"text": "You answer as briefly as possible."}]},
                    {"role": "user", "parts": [{"image": second}, {"text": PROMPTS["color"]}]}]),
        ("two_images", [{"role": "user", "parts": [{"image": first}, {"image": second},
                                                   {"text": PROMPTS["compare"]}]}]),
        ("multi_turn", [{"role": "user", "parts": [{"image": first}, {"text": PROMPTS["first"]}]},
                        {"role": "assistant", "parts": [{"text": "A white car."}]},
                        {"role": "user", "parts": [{"text": PROMPTS["follow"]}]}]),
        ("float32", [{"role": "user", "parts": [{"image": first, "float32": True},
                                                {"text": PROMPTS["describe"]}]}]),
    ]


def _chat(messages, images=None):
    """A conversation in the reference processor's message format."""
    chat = []
    for message in messages:
        content = []
        for part in message["parts"]:
            if "text" in part:
                content.append({"type": "text", "text": part["text"]})
            else:
                content.append({"type": "image", **({"image": images[part["image"]]["image"]}
                                                    if images is not None else {})})
        chat.append({"role": message["role"], "content": content})
    return chat


def _native_prompt_ids(native_dir: Path, checkpoint: Path, conversations, directory: Path):
    source = directory / "conversations.json"
    source.write_text(json.dumps([_chat(messages) for _, messages in conversations]))
    helper = native_dir / "test_gemma3n_processing"
    result = subprocess.run([str(helper), "--prompt-ids", str(checkpoint / "tokenizer.json"),
                             str(source)], text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _sdk_requests(conversations, images, max_new_tokens):
    requests = []
    for _, messages in conversations:
        converted = []
        for message in messages:
            parts = []
            for part in message["parts"]:
                if "text" in part:
                    parts.append({"text": part["text"]})
                    continue
                image = images[part["image"]]
                fmt = "float32" if part.get("float32") else "uint8"
                parts.append({"image": {"path": str(image["f32" if fmt == "float32" else "raw"]),
                                        "height": image["height"], "width": image["width"],
                                        "format": fmt}})
            converted.append({"role": message["role"], "parts": parts})
        requests.append({"messages": converted, "config": {"max_new_tokens": max_new_tokens}})
    return requests


def _reference(checkpoint, conversations, images, max_new_tokens, natives):
    """Greedy FP32 generations, plus teacher-forced margins of the native tokens."""
    import torch
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = None
    try:
        model = Gemma3nForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
        model = model.to("cuda").eval()
        results = []
        for (_, messages), native in zip(conversations, natives):
            inputs = processor.apply_chat_template(_chat(messages, images), add_generation_prompt=True,
                                                   tokenize=True, return_dict=True,
                                                   return_tensors="pt").to("cuda")
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                tokens = output[0, prompt_length:].tolist()
                # Teacher forcing on the native continuation.
                forced = torch.tensor([native["token_ids"]], device="cuda")
                ids = torch.cat([inputs["input_ids"], forced], dim=1)
                extra = {key: value for key, value in inputs.items()
                         if key not in ("input_ids", "attention_mask", "token_type_ids")}
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **extra).logits[0]
                steps = logits[prompt_length - 1:prompt_length - 1 + forced.shape[1]].float()
                chosen = steps.gather(1, forced[0].view(-1, 1)).squeeze(1)
                margins = (steps.max(dim=1).values - chosen).tolist()
            results.append({"token_ids": tokens, "prompt_ids": inputs["input_ids"][0].tolist(),
                            "text": processor.decode(tokens, skip_special_tokens=True),
                            "native_margins": margins})
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name, request, tmp_path):
    manifest, case = CASES[case_name]
    _selected(request.config, manifest, case)
    assert case["reference_backend"] == "hf_transformers"
    assert case["reference_precision"] == "fp32"
    assert manifest["trust_remote_code"] is False
    import torch

    assert torch.cuda.is_available(), "selected Gemma 3n E2E requires CUDA"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    binary = Path(os.environ["TRTMC_BINARY"])
    assert (runtime / "libtrtmc_model_gemma3n.so").is_file()
    native_dir = _native_build_dir()
    checkpoint = _checkpoint(manifest)
    max_new_tokens = int(case["max_new_tokens"])
    images = _images(tmp_path)
    conversations = _conversations(images, max_new_tokens)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family=FAMILY,
                           task=manifest["task"], precision=manifest["precision"],
                           max_sequence_length=manifest["max_sequence_length"],
                           tensor_parallel_size=manifest["tensor_parallel_size"]))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    source, destination = tmp_path / "requests.json", tmp_path / "results.json"
    source.write_text(json.dumps(_sdk_requests(conversations, images, max_new_tokens)))
    with evidence_stage("native"):
        command = [str(native_dir / "test_gemma3n_sdk_cpp"), str(bundle), str(runtime),
                   str(source), str(destination)]
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=1800)
        record_evidence("commands", {"argv": command, "returncode": completed.returncode})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        natives = json.loads(destination.read_text())
        first = images["test_img.jpeg"]
        c_command = [str(native_dir / "test_gemma3n_sdk_c"), str(bundle), str(runtime), str(first["raw"]),
                     str(first["height"]), str(first["width"]), PROMPTS["describe"], str(max_new_tokens)]
        c_result = subprocess.run(c_command, env=env, text=True, capture_output=True, timeout=1800)
        assert c_result.returncode == 0, c_result.stdout + c_result.stderr
        c_output = json.loads(c_result.stdout.strip().splitlines()[-1])
        cli_command = [str(binary), "run", str(bundle), "--runtime-root", str(runtime), "--image",
                       str(first["png"]), "--prompt", PROMPTS["describe"],
                       "--max-new-tokens", str(max_new_tokens)]
        cli_result = subprocess.run(cli_command, env=env, text=True, capture_output=True, timeout=1800)
        assert cli_result.returncode == 0, cli_result.stdout + cli_result.stderr
        cli_output = json.loads(cli_result.stdout)
        record_evidence("native", {"sdk_cpp": natives, "sdk_c": c_output, "cli": cli_output})
    # The three public entry points and both pixel formats agree on one request.
    assert c_output["token_ids"] == natives[0]["token_ids"]
    assert cli_output["token_ids"] == natives[0]["token_ids"]
    assert natives[4]["token_ids"] == natives[0]["token_ids"]
    with evidence_stage("reference"):
        expected = _reference(checkpoint, conversations, images, max_new_tokens, natives)
        record_evidence("reference", {"backend": "hf_transformers", "precision": "fp32",
                                      "results": expected})
    prompt_ids = _native_prompt_ids(native_dir, checkpoint, conversations, tmp_path)
    record_evidence("prompt_ids", {"native": prompt_ids})
    assert prompt_ids == [reference["prompt_ids"] for reference in expected]
    rows = []
    for (name, _), actual, reference in zip(conversations, natives, expected):
        assert actual["text"] and 0 < len(actual["token_ids"]) <= max_new_tokens
        rows.append({"conversation": name, "native": actual["token_ids"],
                     "reference": reference["token_ids"], "text": actual["text"],
                     "reference_text": reference["text"],
                     "max_margin": max(reference["native_margins"])})
    record_evidence("metrics", {"rows": rows})
    if manifest["precision"] == "fp32":
        assert all(row["native"] == row["reference"] for row in rows), rows
    else:
        margin = json.loads((ROOT / "thresholds" / f"{case_name}.json").read_text())[
            "threshold_overrides"]["teacher_forced_logit_margin"]
        assert all(row["max_margin"] <= margin for row in rows), rows
