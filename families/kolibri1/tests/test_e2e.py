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


def test_official_checkpoint_e2e(case_name, tmp_path):
    import gc
    import torch

    _, manifest, case = CASES[case_name]
    directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    runner = runtime_root / "test_kolibri1_runtime"
    assert runner.is_file()
    import sys

    case_path, expected_path = tmp_path / "reference-case.json", tmp_path / "reference.json"
    case_path.write_text(json.dumps(case))
    oracle = subprocess.run(
        [
            sys.executable,
            str(TEST_ROOT / "reference.py"),
            str(directory),
            str(case_path),
            str(expected_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=case.get("runtime_timeout_s", 3600),
    )
    assert oracle.returncode == 0, oracle.stderr + oracle.stdout
    print(oracle.stdout, end="", flush=True)
    expected = json.loads(expected_path.read_text())
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


def test_fp16_scalar_and_silu_match_pytorch(case_name, tmp_path):
    import numpy as np
    import tensorrt as trt
    import torch
    from families.kolibri1.model import Graph

    logger = trt.Logger(trt.Logger.WARNING)
    graph = Graph(None, {}, logger)
    value = graph.net.add_input("value", trt.float16, (1, 65536))
    graph.output(graph.silu(value), "silu")
    graph.output(
        graph.cast(
            graph.mul(graph.cast(value, trt.float32), graph.const([[128**-0.5]], np.float32)),
            trt.float16,
        ),
        "scaled",
    )
    path = tmp_path / "scalar-math.plan"
    with path.open("wb") as stream:
        graph.save(stream)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = torch.linspace(-16, 16, 65536, device="cuda").half().reshape(1, -1)
        activated, scaled = torch.empty_like(inputs), torch.empty_like(inputs)
        context.set_tensor_address("value", inputs.data_ptr())
        context.set_tensor_address("silu", activated.data_ptr())
        context.set_tensor_address("scaled", scaled.data_ptr())
        expected_activation = torch.nn.functional.silu(inputs)
        expected_scale = inputs * (128**-0.5)
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(scaled, expected_scale)
    assert (activated.float() - expected_activation.float()).abs().max().item() <= 2e-4


def test_fp16_rope_matches_pytorch(case_name, tmp_path):
    import tensorrt as trt
    import torch
    from families.kolibri1.model import Graph

    logger = trt.Logger(trt.Logger.WARNING)
    graph = Graph(None, {}, logger)
    hidden = graph.net.add_input("hidden", trt.float16, (1, 4, 128, 128))
    cosine = graph.net.add_input("cos", trt.float16, (1, 1, 128, 128))
    sine = graph.net.add_input("sin", trt.float16, (1, 1, 128, 128))
    graph.output(graph.rope(hidden, cosine, sine), "result")
    path = tmp_path / "rope.plan"
    with path.open("wb") as stream:
        graph.save(stream)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.cuda.stream(stream):
        inputs = torch.randn((1, 4, 128, 128), generator=generator, device="cuda").half()
        angles = torch.randn((1, 1, 128, 128), generator=generator, device="cuda")
        cosine, sine = angles.cos().half(), angles.sin().half()
        first, second = inputs.chunk(2, dim=-1)
        expected = inputs * cosine + torch.cat((-second, first), dim=-1) * sine
        actual = torch.empty_like(inputs)
        for name, value in (("hidden", inputs), ("cos", cosine), ("sin", sine), ("result", actual)):
            context.set_tensor_address(name, value.data_ptr())
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(actual, expected)
