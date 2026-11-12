# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof for MiniCPM-OCR.

Every case builds the manifest bundle, runs the native CLI (and, for premerge cases, the public C and C++ SDK
consumers) and checks the generated tokens against the official Transformers model run in FP32:

* ``token_exact``: the native greedy tokens equal the reference greedy tokens (FP32 bundles).
* ``teacher_forced_near_tie``: the FP32 reference, fed the prompt plus the native tokens, ranks every native token
  within ``near_tie_logit_tolerance`` of its top logit (reduced-precision bundles). The tolerance (``thresholds/``)
  is 2^-6 = 0.015625, about 23 times the largest departure of the official model run in FP16 from its own FP32
  argmax on 25 image / prompt pairs (0.00069, once in 2,638 tokens); a native token further below the
  FP32 argmax fails.

In both cases the native text must equal the official processor's decoding of the native tokens.
"""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import gc
import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

_TEST_DIR = Path(__file__).resolve().parent
_FAMILY = _TEST_DIR.parent.name
_DATA = _TEST_DIR / "data"
_ORACLES = {"token_exact", "teacher_forced_near_tie"}
_SECTIONS = {"vision.plan", "decoder.plan", "runtime.json", "tokenizer.json"}


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "images_text_to_text", path
        assert manifest["precision"] in {"fp16", "fp32"}, path
        assert isinstance(manifest["hf_id"], str) and isinstance(manifest["hf_revision"], str), path
        assert isinstance(manifest["bundle"], str), path
        assert isinstance(manifest["max_sequence_length"], int), path
        assert manifest["tensor_parallel_size"] == 1, path
        for case in manifest["testcases"]:
            name = case["name"]
            assert name not in cases, name
            assert case["oracle"] in _ORACLES, name
            assert case["images"] and all((_DATA / image).is_file() for image in case["images"]), name
            assert isinstance(case["prompt"], str) and case["prompt"], name
            assert isinstance(case["max_new_tokens"], int) and case["max_new_tokens"] > 0, name
            cases[name] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()
_BUNDLES: dict[str, Path] = {}
_REFERENCE: dict[str, object] = {}


def _csv_values(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in str(value).split(",") if item.strip()}


def _selection(config) -> set[str]:
    selected = _csv_values(config.getoption("--e2e-model", default=[]) or [])
    selected |= _csv_values(config.getoption("--e2e-testcase", default=[]) or [])
    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file:
        path = Path(models_file)
        assert path.is_file(), f"E2E models file does not exist: {path}"
        selected |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return selected


def _require_selected(case_name: str, manifest: dict, config) -> None:
    selected = _selection(config)
    if os.environ.get("TRTMC_E2E") != "1" and not selected:
        pytest.skip("real family E2E requires TRTMC_E2E=1 or an explicit E2E selection")
    if selected and not ({_FAMILY, manifest["name"], case_name} & selected):
        pytest.skip(f"{case_name} was not selected")


def _environment():
    binary = Path(os.environ.get("TRTMC_BINARY", ""))
    runtime_root = Path(os.environ.get("TRTMC_RUNTIME_ROOT", ""))
    assert binary.is_file() and os.access(binary, os.X_OK), "selected E2E requires TRTMC_BINARY"
    assert runtime_root.is_dir(), "selected E2E requires TRTMC_RUNTIME_ROOT"
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file(), runtime_root
    assert (runtime_root / f"libtrtmc_model_{_FAMILY}.so").is_file(), runtime_root
    import torch

    assert torch.cuda.is_available(), "selected E2E requires CUDA"
    return binary, runtime_root, torch


def _checkpoint(manifest: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"]))
    assert (path / "config.json").is_file(), path
    return path


def _bundle(manifest: dict, model_dir: Path, request) -> Path:
    """Build each manifest bundle once per session; its cases share it."""
    name = manifest["name"]
    if name not in _BUNDLES:
        root = request.getfixturevalue("tmp_path_factory").mktemp(name)
        bundle = root / manifest["bundle"]
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family=_FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=manifest["max_sequence_length"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
        assert bundle.is_file() and bundle.stat().st_size > 0, bundle
        _BUNDLES[name] = bundle
    return _BUNDLES[name]


def _inspect(binary: Path, bundle: Path) -> None:
    inspected = subprocess.run([str(binary), "inspect", str(bundle)], check=True, capture_output=True, text=True,
                               timeout=60)
    record_evidence("commands", {"argv": inspected.args})
    payload = json.loads(inspected.stdout)
    assert payload["family"] == _FAMILY
    assert payload["task"] == "images_text_to_text"
    assert _SECTIONS <= set(payload["sections"])


def _thresholds(case_name: str) -> dict[str, float]:
    path = _TEST_DIR / "thresholds" / f"{case_name}.json"
    if not path.is_file():
        return {}
    thresholds = json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]
    assert isinstance(thresholds, dict) and thresholds, path
    return thresholds


def _image_paths(case: dict) -> list[Path]:
    return [_DATA / image for image in case["images"]]


def _run_native(binary: Path, runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> dict:
    command = [
        str(binary), "run", str(bundle), "--runtime-root", str(runtime_root),
        "--prompt", case["prompt"], "--max-new-tokens", str(case["max_new_tokens"]),
        "--images", json.dumps([str(path) for path in _image_paths(case)]),
    ]
    record_evidence("commands", {"argv": command})
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900)
    (tmp_path / "native.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (tmp_path / "native.stderr.log").write_text(completed.stderr, encoding="utf-8")
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    completed.check_returncode()
    return json.loads(completed.stdout)


def _run_sdk_consumers(runtime_root: Path, bundle: Path, case: dict, tmp_path: Path) -> list[dict]:
    """Run the public C and C++ SDK consumers on the same request (RGB8 pixels decoded here)."""
    import numpy as np
    from PIL import Image

    native_build = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", ""))
    assert native_build.is_dir(), "premerge E2E requires TRTMC_NATIVE_BUILD_DIR for the SDK consumers"
    arguments = []
    for index, path in enumerate(_image_paths(case)):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        raw = tmp_path / f"image_{index}.rgb"
        raw.write_bytes(pixels.tobytes())
        arguments += [str(raw), str(pixels.shape[0]), str(pixels.shape[1])]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), environment.get("LD_LIBRARY_PATH", "")) if value)
    outputs = []
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{_FAMILY}_sdk_{language}"
        assert consumer.is_file(), f"build the family-owned SDK consumer: {consumer.name}"
        completed = subprocess.run(
            [str(consumer), str(bundle), str(runtime_root), str(case["max_new_tokens"]), case["prompt"], *arguments],
            check=True, capture_output=True, text=True, env=environment, timeout=900)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        assert len(lines) == 1, f"SDK {language} consumer must print one result"
        result = json.loads(lines[0])
        record_evidence(f"sdk_{language}", result)
        assert result["task"] == "images_text_to_text"
        outputs.append(result)
    return outputs


def _reference_model(model_dir: Path, torch):
    """The official model in FP32 with TF32 disabled, loaded once per session."""
    key = str(model_dir)
    if key not in _REFERENCE:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.get_float32_matmul_precision() == "highest"
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32)
        _REFERENCE.clear()
        gc.collect()
        _REFERENCE[key] = (processor, model.to("cuda").eval())
    return _REFERENCE[key]


def _reference_inputs(processor, case: dict):
    from PIL import Image

    images = []
    for path in _image_paths(case):
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": case["prompt"]}]
    rendered = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True)
    inputs = processor(text=[rendered], images=images, return_tensors="pt")
    record_evidence("inputs", {"images": case["images"], "target_sizes": inputs["target_sizes"].tolist(),
                               "prompt_tokens": int(inputs["input_ids"].shape[1])})
    return inputs


def _reference_greedy(model, inputs, case: dict, torch) -> list[int]:
    with torch.inference_mode():
        generated = model.generate(**inputs.to("cuda"), max_new_tokens=int(case["max_new_tokens"]),
                                   do_sample=False)
    return generated[0, inputs["input_ids"].shape[1]:].tolist()


def _teacher_forced_gaps(model, inputs, native_ids: list[int], torch) -> list[float]:
    """FP32 top logit minus the logit of each native token, given the native prefix."""
    prompt = inputs["input_ids"].to("cuda")
    continuation = torch.tensor([native_ids], dtype=prompt.dtype, device="cuda")
    ids = torch.cat([prompt, continuation], dim=1)
    feed = {key: value.to("cuda") for key, value in inputs.items() if key not in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), **feed).logits[0].float()
    rows = logits[prompt.shape[1] - 1: ids.shape[1] - 1]
    chosen = rows.gather(1, continuation[0][:, None])[:, 0]
    return (rows.max(dim=1).values - chosen).tolist()


def _record_diagnostics(native_ids: list[int], reference_ids: list[int]) -> None:
    common = min(len(native_ids), len(reference_ids))
    prefix = next((i for i in range(common) if native_ids[i] != reference_ids[i]), common)
    record_evidence("diagnostics", {
        "matching_prefix_tokens": prefix,
        "native_token_count": len(native_ids),
        "reference_token_count": len(reference_ids),
    })


@pytest.mark.parametrize("case_name", sorted(_CASES))
def test_e2e(case_name: str, request, tmp_path: Path) -> None:
    manifest, case = _CASES[case_name]
    _require_selected(case_name, manifest, request.config)
    record_evidence("inputs", {"manifest": manifest, "case": case})
    binary, runtime_root, torch = _environment()
    model_dir = _checkpoint(manifest)
    record_evidence("checkpoint", {"model_dir": str(model_dir), "hf_id": manifest["hf_id"],
                                   "hf_revision": manifest["hf_revision"]})
    with evidence_stage("build"):
        bundle = _bundle(manifest, model_dir, request)
    with evidence_stage("compare"):
        _inspect(binary, bundle)
    with evidence_stage("native"):
        payload = _run_native(binary, runtime_root, bundle, case, tmp_path)
    native_ids = payload["token_ids"]
    assert isinstance(native_ids, list) and native_ids and all(isinstance(i, int) for i in native_ids)
    assert len(native_ids) <= case["max_new_tokens"]
    assert str(payload["text"]).strip(), "the native run produced no text"
    if case.get("premerge"):
        with evidence_stage("native"):
            for result in _run_sdk_consumers(runtime_root, bundle, case, tmp_path):
                assert result["token_ids"] == native_ids
                assert result["text"] == payload["text"]

    with evidence_stage("reference"):
        processor, model = _reference_model(model_dir, torch)
        inputs = _reference_inputs(processor, case)
        reference_ids = _reference_greedy(model, inputs, case, torch)
        reference_text = processor.post_process_image_text_to_text([reference_ids])[0]
        decoded_native_text = processor.post_process_image_text_to_text([native_ids])[0]
    record_evidence("reference", {"reference_ids": reference_ids, "reference_text": reference_text,
                                  "native_text": payload["text"]})
    _record_diagnostics(native_ids, reference_ids)

    with evidence_stage("compare"):
        assert payload["text"] == decoded_native_text, "the native text is not the official decoding of its tokens"
        if case["oracle"] == "token_exact":
            assert native_ids == reference_ids
            assert payload["text"] == reference_text
            return
        tolerance = float(_thresholds(case_name)["near_tie_logit_tolerance"])
        record_evidence("thresholds", {"near_tie_logit_tolerance": tolerance})
        gaps = _teacher_forced_gaps(model, inputs, native_ids, torch)
        record_evidence("diagnostics", {"max_teacher_forced_gap": max(gaps),
                                        "near_tie_positions": sum(gap > 0 for gap in gaps)})
        assert max(gaps) <= tolerance, f"a native token is {max(gaps):.4f} logits below the FP32 argmax"
        eos = set(json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"])
        assert native_ids[-1] in eos or len(native_ids) == case["max_new_tokens"]
