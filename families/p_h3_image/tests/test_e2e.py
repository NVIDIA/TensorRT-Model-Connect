# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned full-image parity through the native text-to-image Task."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "p_h3_image"
TASKS = frozenset({"text_to_image"})
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


def test_native_tokenizer(case_name):
    from transformers import AutoTokenizer

    _, manifest, _ = CASES[case_name]
    directory = _model_dir(manifest)
    reference = AutoTokenizer.from_pretrained(directory / "tokenizer")
    _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    runner = _native_test_binary("test_p_h3_image_tokenizer")
    for text in (
        "",
        "A watercolor painting of Berlin.",
        "Grüße 世界 😀",
        "café cafe\u0301",
        "1234567890",
        "We're ready.\t\n",
        "<|im_start|>user\nHello<|im_end|>",
    ):
        completed = subprocess.run(
            [str(runner), str(directory / "tokenizer/tokenizer.json"), text],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert json.loads(completed.stdout) == reference.encode(text, add_special_tokens=False)


def test_official_checkpoint_e2e(case_name, tmp_path):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    record_evidence(
        "checkpoint",
        {
            "repo_id": manifest["hf_id"],
            "revision": manifest["hf_revision"],
            "precision": manifest["precision"],
        },
    )
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
    import numpy as np
    import torch

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    _, runtime_root = _runtime()
    import sys

    height, width = manifest["image_height"], manifest["image_width"]
    record_evidence(
        "inputs",
        {
            "seed": case["seed"],
            "num_inference_steps": case["num_inference_steps"],
            "prompt_upsampler": case["prompt_upsampler"],
        },
    )
    input_path = tmp_path / "reference-input.json"
    input_path.write_text(json.dumps([manifest, case]))
    with evidence_stage("reference"):
        oracle = subprocess.run(
            [
                sys.executable,
                str(TEST_ROOT / "reference.py"),
                str(directory),
                str(input_path),
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=case["runtime_timeout_s"],
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
    with evidence_stage("comparison"):
        assert oracle.returncode == 0, oracle.stderr + oracle.stdout
    print(oracle.stdout, end="", flush=True)
    requests = json.loads((tmp_path / "reference-requests.json").read_text())
    expected = [
        np.fromfile(tmp_path / f"reference-{index}.f32", dtype="<f4").reshape(height, width, 3)
        for index in range(len(case["prompts"]))
    ]
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    gc.collect()
    torch.cuda.empty_cache()
    with evidence_stage("runtime"):
        completed = subprocess.run(
            [str(_native_test_binary("test_p_h3_image_runtime")), str(bundle), str(runtime_root)],
            input=json.dumps(requests),
            capture_output=True,
            text=True,
            check=False,
            timeout=case["runtime_timeout_s"],
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
    with evidence_stage("comparison"):
        assert completed.returncode == 0, completed.stderr + completed.stdout
    results = json.loads(completed.stdout.splitlines()[-1])
    with evidence_stage("comparison"):
        assert len(results) == len(expected)
    for index, (result, oracle) in enumerate(zip(results, expected)):
        with evidence_stage("comparison"):
            assert result == dict(height=height, width=width, channels=3, count=height * width * 3)
        actual = np.fromfile(requests[index]["output"], dtype="<f4").reshape(oracle.shape)
        with evidence_stage("comparison"):
            assert np.isfinite(actual).all()
        with evidence_stage("comparison"):
            assert actual.min() >= 0 and actual.max() <= 1
        difference = actual.astype(np.float64) - oracle.astype(np.float64)
        rmse = float(np.sqrt(np.mean(difference**2)))
        maximum = float(np.abs(difference).max())
        a, b = actual.astype(np.float64).ravel(), oracle.astype(np.float64).ravel()
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(
            f"P-H3-Image prompt {index}: RMSE={rmse}, max_abs={maximum}, cosine={cosine}",
            flush=True,
        )
        record_evidence(
            "inputs",
            {"sample": index, "prompt": case["prompts"][index], "request": requests[index]},
        )
        record_evidence(
            "metrics", {"sample": index, "rmse": rmse, "max_abs_error": maximum, "cosine": cosine}
        )
        if evidence_enabled():
            from PIL import Image

            views = []
            for role, pixels in (("native", actual), ("reference", oracle)):
                image_path = tmp_path / f"{role}-{index}.png"
                Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(
                    image_path
                )
                record_evidence(
                    role,
                    {
                        "sample": index,
                        "image": image_path,
                        "mode": "diffusers" if role == "reference" else "native",
                        "shape": list(pixels.shape),
                    },
                )
                views.append(
                    {
                        "title": f"Sample {index} / {role}",
                        "image": image_path,
                        "caption": case["prompts"][index],
                    }
                )
            record_evidence("views", views)
        with evidence_stage("comparison"):
            assert rmse <= case["rmse_max"]
        with evidence_stage("comparison"):
            assert maximum <= case["max_abs_error"]
        with evidence_stage("comparison"):
            assert cosine >= case["cosine_min"]
        _record_comparison(
            tmp_path,
            f"pixels-{index}",
            actual,
            oracle,
            [
                {
                    "name": "rmse",
                    "actual": rmse,
                    "operator": "<=",
                    "expected": case["rmse_max"],
                    "passed": rmse <= case["rmse_max"],
                },
                {
                    "name": "max_abs_error",
                    "actual": maximum,
                    "operator": "<=",
                    "expected": case["max_abs_error"],
                    "passed": maximum <= case["max_abs_error"],
                },
                {
                    "name": "cosine",
                    "actual": cosine,
                    "operator": ">=",
                    "expected": case["cosine_min"],
                    "passed": cosine >= case["cosine_min"],
                },
            ],
        )


@pytest.mark.parametrize("rotary_dim", [96, 128])
def test_bf16_rope_matches_pytorch(case_name, rotary_dim, tmp_path):
    import tensorrt as trt
    import torch
    from families.p_h3_image.graph import Graph

    logger = trt.Logger(trt.Logger.WARNING)
    graph = Graph(None, logger)
    hidden = graph.net.add_input("hidden", trt.bfloat16, (1, 19, 32, 128))
    cosine = graph.net.add_input("cos", trt.bfloat16, (1, 19, 1, rotary_dim))
    sine = graph.net.add_input("sin", trt.bfloat16, (1, 19, 1, rotary_dim))
    graph.output(graph.rope(hidden, cosine, sine, rotary_dim), "result")
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
        inputs = torch.randn((1, 19, 32, 128), generator=generator, device="cuda").bfloat16()
        angles = torch.randn((1, 19, 1, rotary_dim), generator=generator, device="cuda")
        cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
        rotated, tail = inputs[..., :rotary_dim], inputs[..., rotary_dim:]
        first, second = rotated.chunk(2, dim=-1)
        swapped = torch.cat((-second, first), dim=-1)
        expected = torch.cat((rotated * cosine + swapped * sine, tail), dim=-1)
        actual = torch.empty_like(inputs)
        for name, value in (("hidden", inputs), ("cos", cosine), ("sin", sine), ("result", actual)):
            context.set_tensor_address(name, value.data_ptr())
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(actual, expected)


def test_bf16_modulation_residual_and_affine(case_name, tmp_path):
    import numpy as np
    import tensorrt as trt
    import torch
    from families.p_h3_image.graph import Graph

    class Weights:
        index = {"linear.weight": None, "linear.bias": None}
        adapter = None

        def read(self, name):
            return np.array(
                [[1.0078125]] if name.endswith("weight") else [-1.015625], dtype=np.float32
            )

    logger = trt.Logger(trt.Logger.WARNING)
    graph = Graph(Weights(), logger)
    generator = torch.Generator(device="cuda").manual_seed(42)
    inputs = {
        name: torch.randn((1, 65536), generator=generator, device="cuda").bfloat16()
        for name in ("hidden", "scale", "shift", "residual", "gate")
    }
    inputs["linear_input"] = torch.tensor([[1.0078125]], device="cuda", dtype=torch.bfloat16)
    tensors = {
        name: graph.net.add_input(name, trt.bfloat16, tuple(value.shape))
        for name, value in inputs.items()
    }
    hidden, scale, shift, residual, gate = [
        tensors[name] for name in ("hidden", "scale", "shift", "residual", "gate")
    ]
    graph.output(graph.add(graph.mul(hidden, graph.add(scale, 1)), shift), "modulated")
    graph.output(graph.add(residual, graph.mul(gate, hidden)), "residual_output")
    graph.output(graph.linear(tensors["linear_input"], "linear"), "affine")
    expected = {
        "modulated": inputs["hidden"] * (1 + inputs["scale"]) + inputs["shift"],
        "residual_output": inputs["residual"] + inputs["gate"] * inputs["hidden"],
        "affine": torch.tensor([[2**-14]], device="cuda", dtype=torch.bfloat16),
    }
    path = tmp_path / "operator-boundaries.plan"
    with path.open("wb") as stream:
        graph.save(stream)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None
    context = engine.create_execution_context()
    actual = {name: torch.empty_like(value) for name, value in expected.items()}
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    for name, value in {**inputs, **actual}.items():
        context.set_tensor_address(name, value.data_ptr())
    assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), name
