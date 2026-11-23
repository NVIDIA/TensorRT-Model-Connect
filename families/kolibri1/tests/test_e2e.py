# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Kolibri1 MoE parity through native incremental text continuation."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

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


@cache
def _native_test_binary(name: str) -> Path:
    """Build source-only helpers in the configured native test tree."""
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


def _record_comparison(tmp_path, label, native, reference, checks):
    """Retain the output pair and the result of already enforced comparisons."""
    if not evidence_enabled():
        return
    import numpy as np

    paths = []
    for role, value in (("native", native), ("reference", reference)):
        if isinstance(value, np.ndarray):
            path = tmp_path / f"{label}-{role}.npz"
            np.savez_compressed(path, output=value)
        else:
            path = tmp_path / f"{label}-{role}.json"
            path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    record_evidence(
        "reference_comparison",
        {
            "label": label,
            "scope": "independent_reference",
            "enforced": True,
            "native": paths[0],
            "reference": paths[1],
            "checks": [dict(check, scope="independent_reference") for check in checks],
        },
    )


def _checkpoint_source(manifest: dict, path: Path, *, override: bool) -> Path:
    """An arbitrary local directory does not establish the manifest's pinned identity."""
    record_evidence(
        "checkpoint",
        {
            "repo_id": None if override else manifest["hf_id"],
            "revision": None if override else manifest["hf_revision"],
            "source": "local_override" if override else "pinned_snapshot",
            "path": str(path.resolve()),
            "revision_verified": not override,
            "expected_repo_id": manifest["hf_id"],
            "expected_revision": manifest["hf_revision"],
            "precision": manifest["precision"],
        },
    )
    return path


def _model_dir(manifest: dict) -> Path:
    explicit = os.environ.get(f"TRTMC_{FAMILY.upper()}_MODEL_DIR")
    if explicit:
        return _checkpoint_source(
            manifest, _required_path(explicit, f"TRTMC_{FAMILY.upper()}_MODEL_DIR"), override=True
        )
    from huggingface_hub import snapshot_download

    try:
        snapshot = snapshot_download(
            repo_id=manifest["hf_id"], revision=manifest.get("hf_revision"), local_files_only=True
        )
    except Exception as error:
        raise AssertionError(
            f"selected {FAMILY} E2E requires the exact cached checkpoint {manifest['hf_id']}"
        ) from error
    return _checkpoint_source(manifest, Path(snapshot), override=False)


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
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "thresholds",
        {
            key: value
            for key, value in case.items()
            if any(
                word in key
                for word in ("error", "cosine", "atol", "rtol", "snr", "rmse", "max_abs")
            )
        },
    )
    for asset_key in ("image", "test_image", "audio"):
        if asset_key in case:
            for asset_path in (TEST_ROOT / case[asset_key], TEST_ROOT / "data" / case[asset_key]):
                if asset_path.is_file():
                    record_evidence("inputs", {asset_key: asset_path})
                    break
    import gc
    import torch

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    runner = _native_test_binary("test_kolibri1_runtime")
    with evidence_stage("comparison"):
        assert runner.is_file()
    import sys

    case_path, expected_path = tmp_path / "reference-case.json", tmp_path / "reference.json"
    case_path.write_text(json.dumps(case))
    with evidence_stage("reference"):
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
        record_evidence(
            "reference_process",
            {
                "argv": oracle.args,
                "returncode": oracle.returncode,
                "stdout": (oracle.stdout or "")[-4096:],
                "stderr": (oracle.stderr or "")[-16384:],
            },
        )
        assert oracle.returncode == 0, oracle.stderr + oracle.stdout
    print(oracle.stdout, end="", flush=True)
    expected = json.loads(expected_path.read_text())
    gc.collect()
    torch.cuda.empty_cache()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    gc.collect()
    torch.cuda.empty_cache()
    requests = [
        {"prompt": prompt, "max_new_tokens": case["max_new_tokens"]} for prompt in case["prompts"]
    ]
    with evidence_stage("runtime"):
        completed = subprocess.run(
            [str(runner), str(bundle), str(runtime_root)],
            input=json.dumps(requests),
            text=True,
            capture_output=True,
            check=False,
            timeout=case.get("runtime_timeout_s", 3600),
        )
        record_evidence(
            "runtime_process",
            {
                "argv": completed.args,
                "returncode": completed.returncode,
                "stdout": (completed.stdout or "")[-4096:],
                "stderr": (completed.stderr or "")[-16384:],
            },
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.splitlines()[-1])
    record_evidence("inputs", {"requests": requests})
    record_evidence("thresholds", {"tokens_and_text": "exact equality for every prompt"})
    record_evidence("native", {"generations": payload})
    record_evidence("reference", {"mode": "hf_transformers", "generations": expected})
    with evidence_stage("comparison"):
        assert payload == expected
    with evidence_stage("comparison"):
        assert all(item["text"].strip() for item in payload)
    _record_comparison(
        tmp_path,
        "greedy-generations",
        payload,
        expected,
        [
            {
                "name": "generation_mismatch",
                "actual": int(payload != expected),
                "operator": "==",
                "expected": 0,
                "passed": payload == expected,
            },
        ],
    )
    print(f"Kolibri1 exact native greedy generations: {payload!r}", flush=True)


def test_native_tokenizer(case_name):
    from transformers import AutoTokenizer

    _, manifest, _ = CASES[case_name]
    directory = _model_dir(manifest)
    runner = _native_test_binary("test_kolibri1_tokenizer")
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
    import tensorrt as trt
    import torch
    from families.kolibri1.model import Graph

    logger = trt.Logger(trt.Logger.WARNING)
    graph = Graph(None, {}, logger)
    value = graph.net.add_input("value", trt.float16, (1, 65536))
    graph.output(graph.silu(value), "silu")
    graph.output(graph.scale_attention(value, 128), "scaled")
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
