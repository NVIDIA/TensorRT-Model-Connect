# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real PaliGemma build, public images_text_to_text Task and transformers reference.

The requests use the repository's shared test photograph and a tall synthetic
image, against the transformers model run in FP32:

* ``token_exact`` (FP32 bundle): the native greedy tokens and text equal the
  reference's.
* ``teacher_forced_near_tie`` (BF16 bundle): the FP32 reference, forced through
  the native tokens on its own generation path, ranks every native token within
  ``near_tie_logit_tolerance`` of its top logit, and the native text equals the
  reference processor's decoding of the native tokens. Free-running greedy text
  may leave the FP32 text after such a near tie. The tolerance (``thresholds/``)
  is 1.0 logit, about 1.8 times the largest departure of the official model run
  in BF16 from its own FP32 argmax on 41 image / prompt pairs (0.548 logit;
  19 of 265 tokens below the argmax). The FP8 bundle (``quantization`` in the
  manifest) uses 2.5 logit: about 2 times its largest gap on B200 and B300
  (1.27 logit); a build with every activation scale times 0.1 exceeds it.

Both runs also check the native prompt ids and the decoding of the reference
tokens with the tokenizer packaged in the bundle. The evidence keeps the input
images, both outputs and one reference comparison per request.
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
PHOTO = ROOT / "data" / "test_img.jpeg"
CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text())
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)

# PaliGemma "pt" checkpoints answer task prefixes: captioning, VQA, detection.
REQUESTS = (
    {"prompt": "caption en", "images": ("photo",), "max_new_tokens": 32},
    {"prompt": "answer en What color is the car?", "images": ("photo",), "max_new_tokens": 16},
    {"prompt": "detect car", "images": ("photo",), "max_new_tokens": 16},
    {"prompt": "answer en What color is the circle?", "images": ("shapes",), "max_new_tokens": 16},
    {"prompt": "caption en", "images": ("photo", "shapes"), "max_new_tokens": 32},
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


def _images(directory):
    """Decoded RGB pixels of each request image and the image file they come from.

    The photograph is wide; the synthetic scene (red circle above a blue square)
    is tall, so both aspect ratios reach the squash onto the 224x224 square.
    """
    from PIL import Image

    with Image.open(PHOTO) as image:
        photo = np.asarray(image.convert("RGB"))
    y, x = np.mgrid[:320, :240]
    shapes = np.full((320, 240, 3), 255, np.uint8)
    shapes[(y - 90) ** 2 + (x - 120) ** 2 <= 60 ** 2] = (220, 30, 30)
    shapes[(np.abs(y - 235) <= 50) & (np.abs(x - 120) <= 50)] = (30, 60, 200)
    Image.fromarray(shapes).save(directory / "shapes.png")
    return {"photo": photo, "shapes": shapes}, {"photo": PHOTO, "shapes": directory / "shapes.png"}


def _listing(texts):
    return "\n".join(f"{index}. {text}" for index, text in enumerate(texts, 1))


def _record_inputs(files):
    record_evidence("inputs", {
        "images": [files[name] for name in sorted(files)],
        "prompt": _listing(f"[{' + '.join(item['images'])}] {item['prompt']}" for item in REQUESTS),
        "requests": [dict(item, images=list(item["images"])) for item in REQUESTS],
    })


def _record_comparisons(results, directory, tolerance):
    """One comparison per request: the checks the assertions below enforce."""
    for index, (row, item) in enumerate(zip(results, REQUESTS), 1):
        native, reference = directory / f"native-{index}.json", directory / f"reference-{index}.json"
        native.write_text(json.dumps(row["actual"]))
        reference.write_text(json.dumps(row["reference"]))
        if tolerance is None:
            actual, expected = row["actual"]["token_ids"], row["reference"]["token_ids"]
            differing = sum(a != b for a, b in zip(actual, expected)) + abs(len(actual) - len(expected))
            distance = row["normalized_text_edit_distance"]
            checks = [{"name": "differing_tokens", "label": "Generated tokens unlike the reference",
                       "scope": "independent_reference", "actual": differing, "operator": "==",
                       "expected": 0, "passed": differing == 0},
                      {"name": "normalized_text_edit_distance",
                       "label": "Normalized edit distance to the reference text",
                       "scope": "independent_reference", "actual": distance, "operator": "<=",
                       "expected": 0.0, "passed": distance <= 0.0}]
        else:
            gap = row["max_teacher_forced_gap"]
            decoding = _ned(row["actual"]["text"], row["reference"]["native_decoded"])
            checks = [{"name": "max_teacher_forced_gap",
                       "label": "Largest FP32 logit gap below the argmax, native tokens forced",
                       "scope": "independent_reference", "actual": gap, "operator": "<=",
                       "expected": tolerance, "passed": gap <= tolerance},
                      {"name": "native_text_decoding",
                       "label": "Normalized edit distance to the reference decoding of the native tokens",
                       "scope": "independent_reference", "actual": decoding, "operator": "<=",
                       "expected": 0.0, "passed": decoding <= 0.0}]
        count = len(row["actual"]["token_ids"])
        characters = len(row["actual"]["text"].strip())
        checks += [{"name": "text_characters", "label": "Native text characters, at least one",
                    "scope": "contract", "actual": characters, "operator": ">=", "expected": 1,
                    "passed": characters >= 1},
                   {"name": "generated_tokens", "label": "Generated tokens, at least one",
                    "scope": "contract", "actual": count, "operator": ">=", "expected": 1,
                    "passed": count >= 1},
                   {"name": "max_new_tokens", "label": "Generated tokens, at most max_new_tokens",
                    "scope": "contract", "actual": count, "operator": "<=",
                    "expected": item["max_new_tokens"], "passed": count <= item["max_new_tokens"]}]
        record_evidence("reference_comparison", {
            "label": f"{index}. [{' + '.join(item['images'])}] {item['prompt']}",
            "scope": "independent_reference", "enforced": True,
            "native": native, "reference": reference, "checks": checks,
        })


def _native_requests(images, directory):
    specs = {}
    for name, pixels in images.items():
        path = directory / f"{name}.rgb"
        pixels.tofile(path)
        specs[name] = {"path": str(path), "height": pixels.shape[0], "width": pixels.shape[1]}
    return [{"prompt": item["prompt"], "images": [specs[name] for name in item["images"]],
             "max_new_tokens": item["max_new_tokens"]} for item in REQUESTS]


def _reference(checkpoint, images, trust_remote_code, native=None):
    """FP32 greedy results; with native results, also the FP32 gaps of the native tokens.

    The gaps come from generate() forced through the native tokens, so the prefix
    attention, positions and cache follow transformers' own decoding.
    """
    import torch
    from PIL import Image
    from transformers import (AutoProcessor, LogitsProcessor, LogitsProcessorList,
                              PaliGemmaForConditionalGeneration)

    class Force(LogitsProcessor):
        def __init__(self, tokens):
            self.tokens, self.step = tokens, 0

        def __call__(self, input_ids, scores):
            forced = torch.full_like(scores, float("-inf"))
            forced[:, self.tokens[self.step]] = 0
            self.step += 1
            return forced

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
        for index, request in enumerate(REQUESTS):
            pictures = [Image.fromarray(images[name]) for name in request["images"]]
            inputs = processor(images=[pictures], text=[request["prompt"]], return_tensors="pt")
            inputs.pop("labels", None)
            inputs = inputs.to("cuda")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False,
                                        max_new_tokens=request["max_new_tokens"])
            prompt = inputs["input_ids"][0].tolist()
            tokens = output[0, len(prompt):].cpu().tolist()
            result = {"input_ids": prompt, "token_ids": tokens,
                      "text": processor.decode(tokens, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)}
            forced = native[index]["token_ids"] if native is not None else []
            if forced:
                with torch.inference_mode():
                    scored = model.generate(**inputs, do_sample=False, max_new_tokens=len(forced),
                                            logits_processor=LogitsProcessorList([Force(forced)]),
                                            output_logits=True, return_dict_in_generate=True)
                logits = torch.stack(scored.logits)[:, 0].float()
                chosen = logits.gather(1, torch.tensor(forced, device=logits.device)[:, None])[:, 0]
                result["native_gaps"] = (logits.max(dim=1).values - chosen).tolist()
                result["native_decoded"] = processor.decode(forced, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
            results.append(result)
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
    images, files = _images(tmp_path)
    _record_inputs(files)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family=FAMILY,
                           task=manifest["task"], precision=manifest["precision"],
                           max_sequence_length=manifest["max_sequence_length"],
                           tensor_parallel_size=manifest["tensor_parallel_size"],
                           quantization=manifest.get("quantization")))
    source, destination = tmp_path / "requests.json", tmp_path / "outputs.json"
    source.write_text(json.dumps(_native_requests(images, tmp_path)))
    with evidence_stage("native"):
        completed = _run([str(consumer), str(bundle), str(runtime), str(source), str(destination)],
                         runtime, tmp_path, "native")
        assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = json.loads(destination.read_text())
    record_evidence("native", {"text": _listing(row["text"] for row in actual), "results": actual})
    assert case["oracle"] in {"token_exact", "teacher_forced_near_tie"}
    near_tie = case["oracle"] == "teacher_forced_near_tie"
    with evidence_stage("reference"):
        expected = _reference(checkpoint, images, manifest["trust_remote_code"],
                              native=actual if near_tie else None)
        record_evidence("reference", {"backend": "hf_transformers", "precision": "fp32",
                                      "text": _listing(row["text"] for row in expected),
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
    tolerance = None
    if near_tie:
        tolerance = json.loads((ROOT / "thresholds" / f"{case_name}.json").read_text())[
            "threshold_overrides"]["near_tie_logit_tolerance"]
        record_evidence("thresholds", {"near_tie_logit_tolerance": tolerance})
        for row in results:
            row["max_teacher_forced_gap"] = max(row["reference"].get("native_gaps") or [float("inf")])
    record_evidence("metrics", {"cases": results, "oracle": case["oracle"]})
    _record_comparisons(results, tmp_path, tolerance)
    for row, item in zip(results, REQUESTS):
        assert 0 < len(row["actual"]["token_ids"]) <= item["max_new_tokens"], row
        assert row["actual"]["text"].strip(), row
    if not near_tie:
        assert all(row["actual"]["token_ids"] == row["reference"]["token_ids"]
                   and row["actual"]["text"] == row["reference"]["text"] for row in results), results
        return
    eos = json.loads((checkpoint / "generation_config.json").read_text())["eos_token_id"]
    eos = set(eos if isinstance(eos, list) else [eos])
    for row, item in zip(results, REQUESTS):
        tokens = row["actual"]["token_ids"]
        assert row["max_teacher_forced_gap"] <= tolerance, row
        assert row["actual"]["text"] == row["reference"]["native_decoded"], row
        assert not eos.intersection(tokens[:-1]), row
        assert tokens[-1] in eos or len(tokens) == item["max_new_tokens"], row
