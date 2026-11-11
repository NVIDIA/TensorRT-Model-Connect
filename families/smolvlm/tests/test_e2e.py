# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning build, public-Task and reference tests of the pinned SmolVLM2 checkpoint.

The bundle is built from the pinned checkpoint and driven through the public
``images_text_to_text`` SDK call by the family runner. The Transformers model
(eager attention, FP32, no TF32) on the same GPU is the reference:

* host stages: the native image processor reproduces the reference processor's
  ``pixel_values`` bit for bit, and the native chat rendering plus tokenizer
  reproduce its ``input_ids`` exactly, for every request;
* generation: an FP32 bundle must reproduce the reference's greedy token ids
  exactly (``oracle: token_exact``). An FP16 bundle is checked token by token
  with teacher forcing (``oracle: teacher_forced_near_tie``): every native token
  must be the reference's own greedy choice given the native prefix, or trail
  it by at most the sidecar's ``near_tie_logit_gap`` in reference logits. This
  checkpoint has reference near-ties within the first tokens of several
  conversations, after which free-running greedy texts are no longer
  comparable; teacher forcing checks every generated token without that
  cascade. BF16 bundles are not qualified.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from families.smolvlm.tests.oracle import first_divergence, reference_disagreements
from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).resolve().parent
FAMILY = "smolvlm"
MAX_NEW_TOKENS = 64
# The repository's Apache-2.0 photograph (ASSET_LICENSES.md), its mirror image and a
# 150 x 382 portrait strip of it. The strip tiles to 2048 x 1024, whose 512 x 512 global
# view takes Pillow 13's vertical-first resampling order with both passes.
PHOTO = "test_img.jpeg"
STRIP_BOX = (300, 0, 450, 382)
REQUESTS = [
    [{"role": "user", "content": [{"type": "image", "asset": "photo"},
                                  {"type": "text", "text": "Can you describe this image?"}]}],
    [{"role": "user", "content": [{"type": "text", "text": "What is the similarity between these two images?"},
                                  {"type": "image", "asset": "photo"},
                                  {"type": "image", "asset": "strip"}]}],
    [{"role": "system", "content": [{"type": "text", "text": "Answer in one short sentence."}]},
     {"role": "user", "content": [{"type": "image", "asset": "strip"},
                                  {"type": "text", "text": "What is shown in this picture?"}]}],
    [{"role": "user", "content": [{"type": "image", "asset": "photo"},
                                  {"type": "text", "text": "What vehicle is this?"}]},
     {"role": "assistant", "content": [{"type": "text", "text": "This is a white car."}]},
     {"role": "user", "content": [{"type": "text", "text": "Where is it parked?"}]}],
    # Longer than one 2,048-token prefill chunk: the decoder prefills in two calls.
    [{"role": "user", "content": [{"type": "text", "text": "Compare these three images."},
                                  {"type": "image", "asset": "photo"},
                                  {"type": "image", "asset": "mirror"},
                                  {"type": "image", "asset": "strip"}]}],
]
# Extra strings for the real tokenizer: digits, contractions, whitespace runs, scripts.
TOKENIZER_STRINGS = [
    "In 2025, 1,234.56 units cost $78.90 (≈ €72).", "I'm sure they'll say it's fine; we'd've known.",
    "tabs\tand  spaces   and\n\nnewlines \n", "naïve café — 東京タワー, Москва, مرحبا, ① ² Ⅻ ١٢٣",
    "emoji 🙂👍🏽 and <global-img><row_2_col_3> literal tokens", "",
]

CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for case in manifest["testcases"]:
        CASES[case["name"]] = (manifest, case)


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
    explicit = os.environ.get("TRTMC_SMOLVLM_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "model.safetensors").is_file(), root
        return root
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(manifest["hf_id"], revision=manifest["hf_revision"],
                                  allow_patterns=["*.json", "*.txt", "*.safetensors"]))


def _helper(runtime: Path, name: str) -> Path:
    roots = [runtime]
    configured = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if configured:
        roots.insert(0, Path(configured))
    library = runtime / "libtrtmc_model_smolvlm.so"
    if library.is_file():
        roots.append(library.resolve().parent)
    for root in roots:
        for configuration in ("", "Release", "Debug"):
            candidate = root / "families/smolvlm" / configuration / name
            if candidate.is_file():
                return candidate
    raise AssertionError(f"build the owning {name} CMake target")


def _images(directory: Path) -> dict:
    from PIL import Image, ImageOps

    photo = Image.open(ROOT / "data" / PHOTO).convert("RGB")
    images = {}
    for name, picture in (("photo", photo), ("mirror", ImageOps.mirror(photo)),
                          ("strip", photo.crop(STRIP_BOX))):
        raw = directory / f"{name}.rgb"
        np.asarray(picture).tofile(raw)
        images[name] = {"image": picture, "path": str(raw), "height": picture.height, "width": picture.width}
    return images


def _runner_requests(images: dict) -> list:
    def part(item):
        if item["type"] == "text":
            return {"type": "text", "text": item["text"]}
        image = images[item["asset"]]
        return {"type": "image", "path": image["path"], "height": image["height"], "width": image["width"]}

    return [{"messages": [{"role": m["role"], "content": [part(p) for p in m["content"]]} for m in messages],
             "config": {"max_new_tokens": MAX_NEW_TOKENS}} for messages in REQUESTS]


def _template_messages(messages):
    return [{"role": m["role"], "content": [{"type": p["type"], **({"text": p["text"]} if p["type"] == "text" else {})}
                                            for p in m["content"]]} for m in messages]


def _reference(checkpoint: Path, images: dict, actual: list) -> dict:
    import torch
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration


    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    # The checkpoint's processor resizes with Pillow Lanczos (the slow processor).
    assert type(processor.image_processor).__name__ == "SmolVLMImageProcessor"
    pixels = {}
    for name, image in images.items():
        output = processor.image_processor([[image["image"]]], return_row_col_info=True, return_tensors="np")
        pixels[name] = {"pixel_values": np.asarray(output["pixel_values"], dtype=np.float32)[0],
                        "rows": int(output["rows"][0][0]), "cols": int(output["cols"][0][0])}
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    # Some images enable TF32 by default (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1); the reference is FP32.
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert torch.get_float32_matmul_precision() == "highest"
    model = None
    try:
        model, info = SmolVLMForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
            output_loading_info=True)
        assert not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")), info
        model = model.to("cuda").eval()
        results = []
        for messages, native in zip(REQUESTS, actual):
            prompt = processor.apply_chat_template(_template_messages(messages), add_generation_prompt=True,
                                                   tokenize=False)
            request_images = [images[p["asset"]]["image"] for m in messages for p in m["content"]
                              if p["type"] == "image"]
            media = {"images": [request_images]} if request_images else {}
            inputs = processor(text=[prompt], return_tensors="pt", **media).to("cuda")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
                # Teacher forcing: the reference's next-token logits on the prompt plus the
                # native tokens, one row per native token.
                native_ids = torch.tensor([native["token_ids"]], device="cuda")
                forced = model(input_ids=torch.cat([inputs["input_ids"], native_ids], dim=1),
                               attention_mask=torch.ones_like(torch.cat([inputs["input_ids"], native_ids], dim=1)),
                               pixel_values=inputs["pixel_values"],
                               pixel_attention_mask=inputs["pixel_attention_mask"],
                               use_cache=False, logits_to_keep=native_ids.shape[1] + 1)
            tokens = output[0, inputs["input_ids"].shape[1]:].cpu().tolist()
            step_logits = list(forced.logits[0, :-1].float().cpu().numpy())
            results.append({"text": processor.decode(tokens, skip_special_tokens=True), "token_ids": tokens,
                            "input_ids": inputs["input_ids"][0].cpu().tolist(),
                            "disagreements": reference_disagreements(native["token_ids"], step_logits)})
            del inputs, output, forced
        strings = [processor.tokenizer(text, add_special_tokens=True)["input_ids"] for text in TOKENIZER_STRINGS]
        return {"pixels": pixels, "results": results, "strings": strings}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _run(command, env, timeout=1800):
    command = [str(argument) for argument in command]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout)
    record_evidence("commands", {"argv": command, "returncode": completed.returncode})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _host_parity(probe: Path, checkpoint: Path, images: dict, reference: dict, directory: Path, env) -> dict:
    """Native image tiles, prompt ids and string ids against the reference processor."""
    from families.smolvlm.config import ModelConfig

    processing = {"image_processing": ModelConfig.from_dir(checkpoint).processing.as_json()}
    runtime_json = directory / "runtime_processing.json"
    runtime_json.write_text(json.dumps(processing))
    grids, report = {}, {"images": {}, "prompts": [], "strings": []}
    for name, image in images.items():
        pixels, summary = directory / "probe.f32", directory / "probe.json"
        _run([probe, "image", runtime_json, image["path"], image["height"], image["width"], pixels, summary], env)
        native = json.loads(summary.read_text())
        expected = reference["pixels"][name]
        tiles = np.fromfile(pixels, dtype=np.float32).reshape(expected["pixel_values"].shape)
        report["images"][name] = {
            "size": [image["height"], image["width"]], "rows": native["rows"], "cols": native["cols"],
            "frames": native["frames"], "max_abs_pixel_difference": float(
                np.max(np.abs(tiles - expected["pixel_values"]))),
            "bit_exact": bool(np.array_equal(tiles, expected["pixel_values"]))}
        assert (native["rows"], native["cols"]) == (expected["rows"], expected["cols"]), name
        assert report["images"][name]["bit_exact"], report["images"][name]
        grids[name] = (native["rows"], native["cols"])
    tokenizer = checkpoint / "tokenizer.json"
    for messages, expected in zip(REQUESTS, reference["results"]):
        request = {"image_seq_len": 64, "messages": [
            {"role": m["role"], "content": [
                {"type": "text", "text": p["text"]} if p["type"] == "text"
                else {"type": "image", "rows": grids[p["asset"]][0], "cols": grids[p["asset"]][1]}
                for p in m["content"]]} for m in messages]}
        source, target = directory / "prompt.json", directory / "prompt_ids.json"
        source.write_text(json.dumps(request, ensure_ascii=False))
        _run([probe, "prompt", tokenizer, source, target], env)
        native = json.loads(target.read_text())["ids"]
        report["prompts"].append({"tokens": len(native), "exact": native == expected["input_ids"]})
        assert native == expected["input_ids"], report["prompts"][-1]
    source, target = directory / "strings.json", directory / "string_ids.json"
    source.write_text(json.dumps(TOKENIZER_STRINGS, ensure_ascii=False))
    _run([probe, "tokens", tokenizer, source, target], env)
    native = json.loads(target.read_text())
    report["strings"] = [ids == expected for ids, expected in zip(native, reference["strings"])]
    assert all(report["strings"]) and len(native) == len(reference["strings"]), report["strings"]
    return report


def _ned(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = current
    return row[-1] / max(len(a), len(b), 1)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name, request, tmp_path):
    manifest, case = CASES[case_name]
    _selected(request.config, manifest, case)
    assert case["reference_backend"] == "hf_transformers"
    assert case["reference_precision"] == "fp32"
    assert case["oracle"] in {"token_exact", "teacher_forced_near_tie"}
    import torch
    assert torch.cuda.is_available(), "selected SmolVLM E2E requires CUDA"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    runner, probe = _helper(runtime, "smolvlm_inference_test"), _helper(runtime, "smolvlm_runtime_probe")
    assert (runtime / "libtrtmc_model_smolvlm.so").is_file()
    checkpoint = _checkpoint(manifest)
    images = _images(tmp_path)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(model_dir=checkpoint, output_path=bundle, family=FAMILY, task=manifest["task"],
                           precision=manifest["precision"], max_sequence_length=manifest["max_sequence_length"],
                           tensor_parallel_size=manifest["tensor_parallel_size"]))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["TMPDIR"] = str(tmp_path)
    source, destination = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(_runner_requests(images), ensure_ascii=False))
    with evidence_stage("native"):
        completed = _run([runner, bundle, runtime, source, destination], env)
        record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    actual = json.loads(destination.read_text())
    assert len(actual) == len(REQUESTS)
    with evidence_stage("reference"):
        reference = _reference(checkpoint, images, actual)
        record_evidence("reference", {"backend": "hf_transformers", "precision": "fp32",
                                      "results": reference["results"]})
    with evidence_stage("host_parity"):
        record_evidence("host_parity", _host_parity(probe, checkpoint, images, reference, tmp_path, env))
    expected = reference["results"]
    assert len(expected) == len(REQUESTS)
    # The longest prompt must exceed one prefill chunk so that chunked prefill is exercised.
    from families.smolvlm.model import MAX_PREFILL_CHUNK
    assert max(len(b["input_ids"]) for b in expected) > MAX_PREFILL_CHUNK
    rows = []
    for a, b in zip(actual, expected):
        assert a["text"] and 0 < len(a["token_ids"]) <= MAX_NEW_TOKENS
        rows.append({"actual": a, "reference": {"text": b["text"], "token_ids": b["token_ids"]},
                     "prompt_tokens": len(b["input_ids"]),
                     "first_divergence": first_divergence(a["token_ids"], b["token_ids"]),
                     "teacher_forced_disagreements": b["disagreements"],
                     "normalized_text_edit_distance": _ned(a["text"], b["text"])})
    record_evidence("metrics", {"oracle": case["oracle"], "cases": rows})
    if case["oracle"] == "token_exact":
        assert all(row["first_divergence"] is None for row in rows), rows
    else:
        sidecar = json.loads((ROOT / "thresholds" / f"{case_name}.json").read_text())
        tolerance = sidecar["threshold_overrides"]["near_tie_logit_gap"]
        assert all(d["logit_gap"] <= tolerance for row in rows for d in row["teacher_forced_disagreements"]), rows
