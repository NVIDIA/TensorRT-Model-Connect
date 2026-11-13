# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Kolibri1 MoE parity through native incremental text continuation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build

FAMILY = "kolibri1"
TASKS = frozenset({"text_continuation"})
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


def test_official_checkpoint_e2e(case_name, tmp_path, monkeypatch):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    runner = runtime_root / "test_kolibri1_runtime"
    assert runner.is_file()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Expert packing must complete sequentially, so asynchronous source tensors
    # cannot occupy the GPU alongside their dequantized packed replacements.
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    tokenizer = AutoTokenizer.from_pretrained(directory)
    # Keep the oracle, builder, and native runtime in separate memory phases.
    # Dequantization is the official FP8 loader path, with FP16 execution.
    reference = AutoModelForCausalLM.from_pretrained(
        directory,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        quantization_config=FineGrainedFP8Config(dequantize=True),
        attn_implementation="eager",
        experts_implementation="eager",
    ).eval()
    # The loader preserves storage dtype for renamed source tensors. Set the
    # requested compute dtype on the renamed norms and the router's FP32 bias.
    for layer in reference.model.layers:
        layer.post_self_attn_layernorm.to(dtype=torch.float16)
        layer.post_mlp_layernorm.to(dtype=torch.float16)
        layer.mlp.gate.e_score_correction_bias = layer.mlp.gate.e_score_correction_bias.float()
    del layer
    torch.cuda.empty_cache()
    expected = []
    for prompt in case["prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = reference.generate(
                **inputs, do_sample=False, max_new_tokens=case["max_new_tokens"]
            )
        ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        expected.append({"token_ids": ids, "text": tokenizer.decode(ids, skip_special_tokens=True)})
        print(f"Official Kolibri1: {expected[-1]!r}", flush=True)
    del reference, generated, inputs
    gc.collect()
    torch.cuda.empty_cache()
    bundle = tmp_path / manifest["bundle"]
    _build(directory, bundle, manifest)
    gc.collect()
    torch.cuda.empty_cache()
    requests = [
        {"prompt": prompt, "max_new_tokens": case["max_new_tokens"]} for prompt in case["prompts"]
    ]
    completed = subprocess.run(
        [str(runner), str(bundle), str(runtime_root)],
        input=json.dumps(requests),
        text=True,
        capture_output=True,
        check=False,
        timeout=case.get("runtime_timeout_s", 3600),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload == expected
    assert all(item["text"].strip() for item in payload)
    print(f"Kolibri1 exact native greedy generations: {payload!r}", flush=True)


def test_native_tokenizer(case_name):
    from transformers import AutoTokenizer

    _, manifest, _ = CASES[case_name]
    directory = _model_dir(manifest)
    runner = (
        _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
        / "test_kolibri1_tokenizer"
    )
    reference = AutoTokenizer.from_pretrained(directory)
    for text in [
        "The capital of Germany is",
        "1234567890 42",
        "def f(x):\n    return x + 1\n",
        "Grüße 世界 Αλφα 😀",
        "We're ready.\t\n",
        "<|im_start|>user\nHello<|im_end|>",
    ]:
        completed = subprocess.run(
            [str(runner), str(directory / "tokenizer.json"), text],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert json.loads(completed.stdout) == reference.encode(text)
