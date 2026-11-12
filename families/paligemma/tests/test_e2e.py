# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PaliGemma build, public images_text_to_text Task and transformers reference.

The FP32 bundle must reproduce the FP32 reference's greedy tokens exactly. The
BF16 bundle is held to a normalized text edit distance against the same FP32
reference. Both runs also check the native prompt ids and decoded text against
the reference processor, using the tokenizer packaged in the bundle.
"""

from functools import cache
from pathlib import Path
import gc
import json
import os
import subprocess

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).resolve().parent
FAMILY = "paligemma"
CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text())
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)

# PaliGemma "pt" checkpoints answer task prefixes: captioning, VQA, detection.
REQUESTS = (
    {"prompt": "caption en", "images": ("shapes",), "max_new_tokens": 32},
    {"prompt": "answer en What color is the circle?", "images": ("shapes",), "max_new_tokens": 16},
    {"prompt": "detect circle", "images": ("shapes",), "max_new_tokens": 16},
    {"prompt": "describe en", "images": ("landscape",), "max_new_tokens": 48},
    {"prompt": "caption en", "images": ("shapes", "landscape"), "max_new_tokens": 32},
)


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


def _checkpoint(manifest):
    explicit = os.environ.get("TRTMC_PALIGEMMA_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "config.json").is_file()
        return root
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"],
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer.model"]))


def _images():
    """Deterministic synthetic scenes; no image asset or license is involved."""
    y, x = np.mgrid[:240, :320]
    shapes = np.full((240, 320, 3), 255, np.uint8)
    shapes[(y - 120) ** 2 + (x - 90) ** 2 <= 60 ** 2] = (220, 30, 30)
    shapes[(np.abs(y - 120) <= 50) & (np.abs(x - 235) <= 50)] = (30, 60, 200)
    y, x = np.mgrid[:300, :200]
    landscape = np.empty((300, 200, 3), np.uint8)
    landscape[:] = (110, 170, 235)
    landscape[y > 190] = (60, 150, 60)
    landscape[(y - 70) ** 2 + (x - 140) ** 2 <= 30 ** 2] = (250, 220, 40)
    return {"shapes": shapes, "landscape": landscape}


def _native_requests(images, directory):
    specs = {}
    for name, pixels in images.items():
        path = directory / f"{name}.rgb"
        pixels.tofile(path)
        specs[name] = {"path": str(path), "height": pixels.shape[0], "width": pixels.shape[1]}
    return [{"prompt": item["prompt"], "images": [specs[name] for name in item["images"]],
             "max_new_tokens": item["max_new_tokens"]} for item in REQUESTS]


def _reference(checkpoint, images, trust_remote_code):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    # The slow processor resamples with Pillow, which the native preprocessing reproduces.
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False,
                                              trust_remote_code=trust_remote_code)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = None
    try:
        model, info = PaliGemmaForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
            output_loading_info=True, trust_remote_code=trust_remote_code)
        assert not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "unexpected_keys",
                                             "error_msgs")), info
        model = model.to("cuda").eval()
        results = []
        for request in REQUESTS:
            pictures = [Image.fromarray(images[name]) for name in request["images"]]
            inputs = processor(images=[pictures], text=[request["prompt"]], return_tensors="pt")
            inputs.pop("labels", None)
            inputs = inputs.to("cuda")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False,
                                        max_new_tokens=request["max_new_tokens"])
            prompt = inputs["input_ids"][0].tolist()
            tokens = output[0, len(prompt):].cpu().tolist()
            results.append({"input_ids": prompt, "token_ids": tokens,
                            "text": processor.decode(tokens, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)})
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _ned(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = current
    return row[-1] / max(len(a), len(b), 1)


@cache
def _native_build():
    root = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    # CI builds only a family's CTest and SDK-consumer targets (tools/ci/e2e.py),
    # so the E2E builds the bundle tokenizer check it runs.
    subprocess.run(["cmake", "--build", str(root), "--parallel", "8", "--target",
                    "paligemma_tokenizer_check"], check=True, timeout=900)
    return root


def _binary(name):
    candidate = _native_build() / name
    assert candidate.is_file(), f"build the {name} CMake target in TRTMC_NATIVE_BUILD_DIR"
    return candidate


def _run(command, runtime, tmp_path, stage):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["TMPDIR"] = str(tmp_path)
    completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=3600)
    record_evidence("commands", {"argv": command, "returncode": completed.returncode})
    record_evidence(stage, {"stdout": completed.stdout, "stderr": completed.stderr})
    return completed


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name, request, tmp_path):
    manifest, case = CASES[case_name]
    _selected(request.config, manifest, case)
    assert case["reference_backend"] == "hf_transformers"
    assert case["reference_precision"] == "fp32"
    assert manifest["trust_remote_code"] is False
    import torch
    assert torch.cuda.is_available(), "selected PaliGemma E2E requires CUDA"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    consumer = _binary("test_paligemma_sdk_cpp")
    checker = _binary("paligemma_tokenizer_check")
    checkpoint = _checkpoint(manifest)
    images = _images()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family=FAMILY,
                           task=manifest["task"], precision=manifest["precision"],
                           max_sequence_length=manifest["max_sequence_length"],
                           tensor_parallel_size=manifest["tensor_parallel_size"]))
    source, destination = tmp_path / "requests.json", tmp_path / "outputs.json"
    source.write_text(json.dumps(_native_requests(images, tmp_path)))
    with evidence_stage("native"):
        completed = _run([str(consumer), str(bundle), str(runtime), str(source), str(destination)],
                         runtime, tmp_path, "native")
        assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(destination.read_text())
    with evidence_stage("reference"):
        expected = _reference(checkpoint, images, manifest["trust_remote_code"])
        record_evidence("reference", {"backend": "hf_transformers", "precision": "fp32",
                                      "results": expected})
    assert len(actual) == len(expected) == len(REQUESTS)

    cases = {
        "prompts": [{"prefix": item["prompt"], "images": len(item["images"]),
                     "input_ids": reference["input_ids"]}
                    for item, reference in zip(REQUESTS, expected)],
        "decodes": [{"token_ids": reference["token_ids"], "text": reference["text"]}
                    for reference in expected],
    }
    (tmp_path / "tokenizer_cases.json").write_text(json.dumps(cases))
    with evidence_stage("tokenizer"):
        completed = _run([str(checker), str(bundle), str(tmp_path / "tokenizer_cases.json")],
                         runtime, tmp_path, "tokenizer_check")
        assert completed.returncode == 0, completed.stdout + completed.stderr

    results = [{"prompt": item["prompt"], "images": list(item["images"]), "actual": a,
                "reference": b, "normalized_text_edit_distance": _ned(a["text"], b["text"])}
               for item, a, b in zip(REQUESTS, actual, expected)]
    for row, item in zip(results, REQUESTS):
        assert 0 < len(row["actual"]["token_ids"]) <= item["max_new_tokens"], row
    if case["oracle"] == "token_exact":
        record_evidence("metrics", {"cases": results, "oracle": "token_exact"})
        assert all(row["actual"]["token_ids"] == row["reference"]["token_ids"]
                   and row["actual"]["text"] == row["reference"]["text"] for row in results), results
    else:
        assert case["oracle"] == "normalized_text_edit_distance"
        threshold = json.loads((ROOT / "thresholds" / f"{case_name}.json").read_text())[
            "threshold_overrides"]["normalized_text_edit_distance"]
        record_evidence("metrics", {"cases": results, "normalized_text_edit_distance_max": threshold})
        assert all(row["normalized_text_edit_distance"] <= threshold for row in results), results
