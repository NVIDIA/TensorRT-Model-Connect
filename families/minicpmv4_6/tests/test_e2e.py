# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned checkpoint-to-native-runtime proof for MiniCPM-V 4.6.

Every case builds (once per manifest) a bundle from the pinned checkpoint,
runs the native CLI and compares the generated tokens with the Transformers
reference, run in FP32 on the same decoded pixels:

``token_exact``     the native greedy tokens equal the reference's.
``teacher_forced``  the reference re-scores the native tokens; each must be the
                    reference's argmax or within ``logit_tolerance`` (default
                    ``_LOGIT_TOLERANCE``) of it.

Image fixtures are decoded once with Pillow and written as lossless PNG, so the
native loader and the reference read identical RGB values. Image cases marked
``sdk`` also run the public C and C++ SDK consumers on the same input and
require their tokens to equal the CLI's.

Each case also checks the content of both outputs: a short ``expected_answer``
is the whole text (case and white space aside), and each of the
``expected_mentions`` of a longer output appears as a whole word.

The real E2E runs with ``TRTMC_E2E=1`` or an explicit selection
(``--e2e-model``, ``--e2e-testcase``, ``--e2e-models-file``); a selection that
names this family but matches none of its cases is a usage error, and a
testcase-only selection fails unless some collected E2E case owns it.
"""

from __future__ import annotations

from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

import gc
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

_TEST_DIR = Path(__file__).resolve().parent
_REPO = _TEST_DIR.parents[2]
_FAMILY = _TEST_DIR.parent.name
_ORACLES = {"token_exact", "teacher_forced"}
# The single parameter of a testcase-only selection that names none of this family's cases.
_SELECTION_CHECK = "selection-check"
# FP16 bundle tokens may trail the FP32 reference argmax by at most this many logits.
# Calibration, every output scored by the FP32 reference (TF32 off) on the E2E inputs
# plus a 160-token transcription:
#   healthy   - the reference's own FP16 and BF16 runs (10 runs, 470 tokens): at most
#               0.215 (FP16) and 0.0555 (BF16); the FP16 bundle (4 cases, 75 tokens):
#               at most 0.0555;
#   defective - 12 known-wrong answers (3 per input on 4 inputs: one word changed or
#               another answer): each has a token at least 10.6 below the argmax.
_LOGIT_TOLERANCE = 0.25


def _load_cases() -> dict[str, tuple[dict, dict]]:
    cases: dict[str, tuple[dict, dict]] = {}
    for path in sorted((_TEST_DIR / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == _FAMILY, path
        assert manifest["task"] == "images_text_to_text", path
        assert manifest["precision"] in {"fp16", "fp32"}, path
        assert isinstance(manifest["max_sequence_length"], int), path
        assert manifest["tensor_parallel_size"] == 1, path
        for case in manifest["testcases"]:
            assert case["name"] not in cases and case["name"] != _SELECTION_CHECK, case["name"]
            assert case["oracle"] in _ORACLES, case["name"]
            # One content check per case: a whole short answer, or whole-word mentions.
            assert ("expected_answer" in case) != ("expected_mentions" in case), case["name"]
            texts = [case["expected_answer"]] if "expected_answer" in case else case["expected_mentions"]
            assert texts and all(isinstance(text, str) and text.strip() for text in texts), case["name"]
            cases[case["name"]] = (manifest, case)
    assert cases, f"{_FAMILY} has no E2E cases"
    return cases


_CASES = _load_cases()


def _csv_values(values) -> set[str]:
    return {item.strip() for value in values or [] for item in str(value).split(",") if item.strip()}


def _filters(config) -> tuple[set[str], set[str]]:
    models = _csv_values(config.getoption("--e2e-model"))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        path = Path(models_file)
        if not path.is_file():
            raise pytest.UsageError(f"E2E models file does not exist: {path}")
        models |= {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return models, _csv_values(config.getoption("--e2e-testcase"))


def _selected_cases(config) -> tuple[list[str], bool]:
    """The cases to run, and whether the real E2E runs at all.

    ``--e2e-model`` (this family, a manifest or a case name) and
    ``--e2e-testcase`` (case names) narrow each other. A model selection that
    names this family, one of its manifests or one of its cases but matches
    none of its cases is an error; a model selection of other families leaves
    this file empty, whatever ``--e2e-testcase`` names.
    """
    models, testcases = _filters(config)
    if not models and not testcases:
        return sorted(_CASES), os.environ.get("TRTMC_E2E") == "1"
    selected = [
        name
        for name, (manifest, _) in sorted(_CASES.items())
        if (not models or {_FAMILY, manifest["name"], name} & models)
        and (not testcases or name in testcases)
    ]
    own = {_FAMILY, *(manifest["name"] for manifest, _ in _CASES.values()), *_CASES}
    if not selected and models & own:
        raise pytest.UsageError(
            f"the E2E selection names {_FAMILY} but matches none of its cases "
            f"(--e2e-model {sorted(models)}, --e2e-testcase {sorted(testcases)}); "
            f"its cases: {', '.join(sorted(_CASES))}"
        )
    if not selected and testcases and not models:
        # Only the whole session knows whether another family owns these cases (test_e2e checks).
        return [_SELECTION_CHECK], True
    return selected, True


def _foreign_testcases(request) -> None:
    """Skip when another collected family's E2E owns the selected testcases; fail when none does."""
    _, testcases = _filters(request.config)
    owned = set()
    for item in request.session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        name = getattr(item, "originalname", "")
        if name.startswith("test_") and name.endswith("e2e") and "case_name" in params:
            owned.add(params["case_name"])
    owned.discard(_SELECTION_CHECK)
    if testcases & owned:
        pytest.skip(f"--e2e-testcase {sorted(testcases & owned)} belongs to another family")
    pytest.fail(f"--e2e-testcase {sorted(testcases)} matches none of the session's E2E cases")


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = names if enabled else [
        pytest.param(name, marks=pytest.mark.skip(
            reason="real family E2E requires TRTMC_E2E=1 or an explicit E2E selection"))
        for name in names
    ]
    metafunc.parametrize("case_name", parameters, ids=names)


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
    """The pinned snapshot from the local Hugging Face cache, offline as in CI.

    ``local_files_only=True`` takes a commit's snapshot from the cache alone. Without it, huggingface_hub lists the
    revision's files first unless the cache holds that listing, which a cache staged by an older client lacks, and
    offline the listing fails.
    """
    from huggingface_hub import snapshot_download

    assert os.environ.get("HF_HUB_OFFLINE") == "1", "selected E2E resolves the checkpoint offline, as CI does"
    try:
        path = Path(snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest["hf_revision"], local_files_only=True))
    except Exception as error:
        raise AssertionError(
            f"selected {_FAMILY} E2E requires the exact cached checkpoint "
            f"{manifest['hf_id']} at {manifest['hf_revision']}"
        ) from error
    assert path.name == manifest["hf_revision"] and (path / "config.json").is_file(), path
    return path


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    assert len(set(keys)) == len(keys), f"duplicate keys: {keys}"
    return dict(pairs)


def _thresholds(case_name: str) -> dict:
    """The case's optional threshold sidecar; ``logit_tolerance`` is its only key."""
    path = _TEST_DIR / "thresholds" / f"{case_name}.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    assert set(payload) == {"threshold_overrides"}, path
    thresholds = payload["threshold_overrides"]
    assert isinstance(thresholds, dict) and thresholds, path
    assert set(thresholds) <= {"logit_tolerance"}, (path, sorted(thresholds))
    return thresholds


def _tolerance(thresholds: dict) -> float:
    value = thresholds.get("logit_tolerance", _LOGIT_TOLERANCE)
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0, value
    return float(value)


_BUNDLES: dict[str, Path] = {}


def _bundle(manifest: dict, model_dir: Path, directory: Path) -> Path:
    """Build each manifest's bundle once per session; its cases share it."""
    if manifest["name"] in _BUNDLES:
        return _BUNDLES[manifest["name"]]
    bundle = directory / manifest["bundle"]
    with evidence_stage("build"):
        build(BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=_FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest["max_sequence_length"],
            tensor_parallel_size=manifest["tensor_parallel_size"],
        ))
    assert bundle.is_file() and bundle.stat().st_size > 0, bundle
    _BUNDLES[manifest["name"]] = bundle
    return bundle


def _images(case: dict, directory: Path) -> list[Path]:
    """Decode each fixture with Pillow once and store it as lossless PNG."""
    from PIL import Image

    paths = []
    for index, source in enumerate(case.get("images", [])):
        target = directory / f"input-{index}.png"
        with Image.open(_REPO / source) as image:
            image.convert("RGB").save(target)
        paths.append(target)
    return paths


def _run(command: list[str], tmp_path: Path, label: str) -> dict:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900)
    record_evidence(f"{label}_command", {
        "argv": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    })
    (tmp_path / f"{label}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (tmp_path / f"{label}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    completed.check_returncode()
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run_cli(binary: Path, runtime_root: Path, bundle: Path, case: dict, images: list[Path],
             tmp_path: Path) -> dict:
    command = [str(binary), "run", str(bundle), "--runtime-root", str(runtime_root),
               "--prompt", case["prompt"], "--max-new-tokens", str(case["max_new_tokens"])]
    if len(images) == 1:
        command += ["--image", str(images[0])]
    elif images:
        command += ["--images", json.dumps([str(path) for path in images])]
    else:
        command += ["--use-chat-template", "true"]
    return _run(command, tmp_path, "cli")


def _run_sdk_consumers(runtime_root: Path, bundle: Path, case: dict, image: Path,
                       tmp_path: Path) -> list[dict]:
    import numpy as np
    from PIL import Image

    native_build = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", runtime_root))
    with Image.open(image) as decoded:
        pixels = np.asarray(decoded.convert("RGB"), dtype=np.float32) / np.float32(255)
    raw = tmp_path / "sdk-input.rgb.f32"
    raw.write_bytes(np.ascontiguousarray(pixels, dtype="<f4").tobytes())
    outputs = []
    for language in ("c", "cpp"):
        consumer = native_build / f"test_{_FAMILY}_sdk_{language}"
        assert consumer.is_file() and os.access(consumer, os.X_OK), consumer
        outputs.append(_run([str(consumer), str(bundle), str(runtime_root), str(raw),
                             str(pixels.shape[0]), str(pixels.shape[1]), case["prompt"],
                             str(case["max_new_tokens"])], tmp_path, f"sdk_{language}"))
    return outputs


def _reference_inputs(processor, case: dict, images: list[Path]):
    from PIL import Image

    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": case["prompt"]}]
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )
    decoded = []
    for path in images:
        with Image.open(path) as image:
            decoded.append(image.convert("RGB"))
    arguments = {"images": decoded} if decoded else {}
    return processor(text=[rendered], return_tensors="pt", **arguments)


def _reference(model_dir: Path, case: dict, images: list[Path], native_ids: list[int], torch) -> dict:
    """FP32 Transformers reference: greedy tokens and the scores of the native tokens."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Exact FP32: TF32 off, and both global switches restored afterwards, also when loading or inference fails.
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        assert torch.get_float32_matmul_precision() == "highest"
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32).eval().to("cuda")
        try:
            return _reference_run(processor, model, case, images, native_ids, torch)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _reference_run(processor, model, case: dict, images: list[Path], native_ids: list[int], torch) -> dict:
    """Greedy reference tokens and the scores of the native tokens."""
    inputs = _reference_inputs(processor, case, images).to("cuda")
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=int(case["max_new_tokens"]), do_sample=False)
        reference_ids = generated[0, prompt_length:].tolist()
        # Re-score the native continuation: logits[i] predicts native token i.
        forced = torch.tensor([native_ids], dtype=inputs["input_ids"].dtype, device="cuda")
        scored = dict(inputs)
        scored["input_ids"] = torch.cat([inputs["input_ids"], forced], dim=1)
        scored["attention_mask"] = torch.ones_like(scored["input_ids"])
        logits = model(**scored).logits[0, prompt_length - 1: prompt_length - 1 + len(native_ids)].float()
        native = logits.gather(1, forced[0].unsqueeze(1)).squeeze(1)
        best = logits.max(dim=1)
        gaps = (best.values - native).tolist()
    eos = model.generation_config.eos_token_id
    return {
        "prompt_tokens": prompt_length,
        "reference_ids": reference_ids,
        "reference_text": processor.tokenizer.decode(reference_ids, skip_special_tokens=True),
        "teacher_forced_argmax_ids": best.indices.tolist(),
        "native_logit_gaps": gaps,
        "native_ids_decoded": processor.tokenizer.decode(native_ids, skip_special_tokens=True),
        "eos_token_ids": [eos] if isinstance(eos, int) else list(eos or []),
    }


def _record_teacher_forced(native_ids: list[int], reference: dict, worst: float, tolerance: float,
                           tmp_path: Path) -> None:
    """Record the teacher-forced comparison so that the report can verify it."""
    if not evidence_enabled():
        return
    native_file = tmp_path / "teacher-forced-native.json"
    reference_file = tmp_path / "teacher-forced-reference.json"
    native_file.write_text(json.dumps({"token_ids": native_ids}) + "\n", encoding="utf-8")
    reference_file.write_text(json.dumps({
        "argmax_ids": reference["teacher_forced_argmax_ids"],
        "native_logit_gaps": reference["native_logit_gaps"],
    }, allow_nan=False) + "\n", encoding="utf-8")
    record_evidence("reference_comparison", {
        "label": "Native tokens re-scored by the FP32 Transformers reference (teacher forcing)",
        "scope": "independent_reference",
        "enforced": True,
        "native": native_file,
        "reference": reference_file,
        "checks": [{
            "name": "max_logit_gap",
            "label": "Largest logit gap of a native token below the reference argmax",
            "scope": "independent_reference",
            "actual": worst,
            "operator": "<=",
            "expected": tolerance,
            "passed": worst <= tolerance,
        }],
    })


def _assert_oracle(case: dict, thresholds: dict[str, float], payload: dict, reference: dict,
                   tmp_path: Path) -> None:
    native_ids = payload["token_ids"]
    budget = int(case["max_new_tokens"])
    assert native_ids and len(native_ids) <= budget
    eos_ids = set(reference["eos_token_ids"])
    assert eos_ids, "the reference declares no end-of-sequence token"
    # Generation ends at its first end-of-sequence token, or when the budget is spent.
    assert not eos_ids & set(native_ids[:-1]), f"native output continues after an end token: {native_ids}"
    assert native_ids[-1] in eos_ids or len(native_ids) == budget, f"native output stopped early: {native_ids}"
    # The runtime's text is the reference tokenizer's decoding of the same ids, whole.
    native_text = str(payload["text"])
    assert native_text == reference["native_ids_decoded"], (native_text, reference["native_ids_decoded"])
    if case["oracle"] == "token_exact":
        assert not thresholds, "token_exact cases take no threshold"
        assert native_ids == reference["reference_ids"], (native_ids, reference["reference_ids"])
    else:
        tolerance = _tolerance(thresholds)
        record_evidence("thresholds", {"logit_tolerance": tolerance})
        gaps = reference["native_logit_gaps"]
        assert len(gaps) == len(native_ids) == len(reference["teacher_forced_argmax_ids"])
        assert all(math.isfinite(gap) for gap in gaps), gaps
        worst = max(gaps)
        _record_teacher_forced(native_ids, reference, worst, tolerance, tmp_path)
        assert worst <= tolerance, f"native token {gaps.index(worst)} trails the reference argmax by {worst}"
    # Content of both runs: a short answer is the whole text; a longer output names each mention.
    for label, text in (("native", native_text), ("reference", reference["reference_text"])):
        if "expected_answer" in case:
            assert _normalized(text) == _normalized(case["expected_answer"]), (label, text)
        for mention in case.get("expected_mentions", []):
            assert _mentions(text, mention), (label, text, mention)


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _mentions(text: str, mention: str) -> bool:
    """The mention appears as a whole word: not inside a longer word or hyphenated name; any case."""
    return re.search(rf"(?<![\w-]){re.escape(mention)}(?![\w-])", text, re.IGNORECASE) is not None


def test_e2e(case_name: str, tmp_path: Path, tmp_path_factory, request) -> None:
    if case_name == _SELECTION_CHECK:
        _foreign_testcases(request)
    manifest, case = _CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case, "prompt": case["prompt"]})
    binary, runtime_root, torch = _environment()
    model_dir = _checkpoint(manifest)
    record_evidence("checkpoint", {"hf_id": manifest["hf_id"], "hf_revision": manifest["hf_revision"]})
    bundle = _bundle(manifest, model_dir, tmp_path_factory.mktemp(manifest["name"]))
    images = _images(case, tmp_path)
    if images:
        record_evidence("inputs", {"images": images})

    with evidence_stage("native"):
        payload = _run_cli(binary, runtime_root, bundle, case, images, tmp_path)
    record_evidence("native", payload)
    if case.get("sdk", False):
        assert len(images) == 1, "SDK consumer cases use exactly one image"
        with evidence_stage("native"):
            for output in _run_sdk_consumers(runtime_root, bundle, case, images[0], tmp_path):
                assert output["token_ids"] == payload["token_ids"]
                assert output["text"] == payload["text"]

    with evidence_stage("reference"):
        reference = _reference(model_dir, case, images, payload["token_ids"], torch)
    record_evidence("reference", reference)
    with evidence_stage("compare"):
        _assert_oracle(case, _thresholds(case_name), payload, reference, tmp_path)
