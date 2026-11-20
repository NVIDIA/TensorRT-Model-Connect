# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict default quality gate has no short-answer or prefix relaxation."""

import json
from pathlib import Path
import pytest
from families.qwen3_5_moe.tests.quality import assert_quality, normalized_edit_distance


def test_normalization_and_fixed_threshold():
    assert normalized_edit_distance(" A  B ", "a b") == 0
    assert normalized_edit_distance("abc", "abd") == 1 / 3
    assert_quality("abcdefghij", "abcdefghiX")
    for actual, expected in (
        ("abcdefghij", "abcdefghXX"),
        ("Paris", "The answer is Paris"),
        ("", ""),
    ):
        with pytest.raises(AssertionError):
            assert_quality(actual, expected)


@pytest.mark.parametrize("gpu_passed", [True, False, None])
@pytest.mark.parametrize("reference_passed", [True, False, None])
def test_independent_quality_requires_both_gates(gpu_passed, reference_passed):
    """A passing oracle never repairs failed/missing GPU proof or mutates prior evidence."""
    from families.qwen3_5_moe.tests.quality import combined_evidence

    gpu = {"task": "vision_language_generation", "max_sequence_length": 4096}
    reference = {}
    if gpu_passed is not None:
        gpu["passed"] = gpu_passed
    if reference_passed is not None:
        reference["passed"] = reference_passed
    original_gpu = dict(gpu)
    result = combined_evidence(gpu, reference)
    assert result["passed"] is (gpu_passed is True and reference_passed is True)
    assert result["gpu_passed"] is (gpu_passed is True)
    assert result["hf_quality"] == ("passed" if reference_passed is True else "failed")
    assert result["task"] == gpu["task"]
    assert result["max_sequence_length"] == 4096
    assert gpu == original_gpu


@pytest.mark.parametrize("task", [
    "text_generation", "vision_language_generation", "images_text_to_text",
])
@pytest.mark.parametrize("separate_build", [False, True])
def test_native_harness_reads_metadata_without_copying_engines(
    tmp_path, task, separate_build, monkeypatch,
):
    """Native E2E uses its actual binding and the same existing raw/image/recovery cases."""
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from families.qwen3_5_moe.tests.edge_validation import extract_bundle, requests

    bundle = tmp_path / "native.bundle"
    writer = BundleWriter(bundle)
    media = task != "text_generation"
    writer.set_header(family="qwen3_5_moe", task=task, backend="trt")
    writer.add_bytes("engine.plan", b"decoder")
    writer.add_json("runtime.json", {"max_cache_length": 1024})
    if media:
        writer.add_bytes("native_visual.plan", b"visual")
        writer.add_json("native_visual.json", {"capacity": 1024, "input_limit": 1024})
    writer.finish()
    with pytest.raises(ValueError, match="native, not Edge"):
        extract_bundle(bundle, tmp_path / "edge-only")
    extracted = tmp_path / "native"
    marker = extract_bundle(bundle, extracted, allow_native=True)
    assert marker["backend"] == "native" and marker["task"] == task
    assert marker["max_sequence_length"] == 1024 and not list(extracted.iterdir())
    actual, _ = requests(1024, tmp_path / "image.rgb", 16, 16, task, backend=marker["backend"])
    expected, _ = requests(1024, tmp_path / "image.rgb", 16, 16,
                           "vision_language_generation" if media else "text_generation",
                           backend="edge_llm")
    native_cases = {case["id"]: case for case in actual}
    edge_cases = {case["id"]: case for case in expected}
    assert edge_cases["reject_control"]["expect_error"] is True
    seeded = native_cases["seeded_control"]
    repeated = native_cases["seeded_control_repeat"]
    assert seeded["config"] == edge_cases["reject_control"]["config"]
    assert seeded["config"]["seed"] == 42
    assert seeded["equal_to"] == "raw" and not seeded.get("expect_error", False)
    assert repeated["config"] == seeded["config"]
    assert repeated["equal_to"] == "seeded_control" and not repeated.get("expect_error", False)
    assert [case for case in actual if not case["id"].startswith("seeded_control")] == [
        case for case in expected if case["id"] != "reject_control"
    ]
    rejection = "reject_image_capacity" if media else "reject_capacity"
    recovery = "image_after_rejection" if media else "raw_after_capacity"
    assert native_cases[rejection]["expect_error"] is True
    assert native_cases[recovery]["equal_to"] == ("image" if media else "raw")

    # Check path wiring only; do not mock successful inference or quality evidence.
    from types import SimpleNamespace
    from PIL import Image
    from families.qwen3_5_moe.tests import edge_validation

    runtime = tmp_path / "installed-runtime"
    build = tmp_path / "native-build" if separate_build else None
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image)

    def check_driver(command, log):
        assert Path(command[0]) == (
            (build or runtime) / "families/qwen3_5_moe/qwen3_5_moe_edge_inference"
        )
        assert command[command.index("--runtime-root") + 1] == str(runtime)
        assert command[command.index("--expected-backend") + 1] == "native"
        raise RuntimeError("driver path verified")

    monkeypatch.setattr(edge_validation, "run_driver", check_driver)
    with pytest.raises(RuntimeError, match="driver path verified"):
        edge_validation.gpu(SimpleNamespace(
            bundle=bundle, runtime_root=runtime, native_build_dir=build,
            output=tmp_path / "proof", image=image,
            checkpoint_id="fixture", checkpoint_revision="fixture",
        ))


@pytest.mark.parametrize("task", ["vision_language_generation", "images_text_to_text"])
def test_native_harness_rejects_inconsistent_visual_capacity(tmp_path, task):
    from tensorrt_model_connect.bundle_writer import BundleWriter
    from families.qwen3_5_moe.tests.edge_validation import extract_bundle

    bundle = tmp_path / "native.bundle"
    writer = BundleWriter(bundle)
    writer.set_header(family="qwen3_5_moe", task=task, backend="trt")
    writer.add_bytes("engine.plan", b"decoder")
    writer.add_json("runtime.json", {"max_cache_length": 1024})
    writer.add_bytes("native_visual.plan", b"visual")
    writer.add_json("native_visual.json", {"capacity": 512, "input_limit": 512})
    writer.finish()
    with pytest.raises(ValueError, match="capacity differ"):
        extract_bundle(bundle, tmp_path / "extracted", allow_native=True)


@pytest.mark.parametrize("name,available,expected", [
    ("cuda", True, "cuda:0"),
    ("cuda", False, None),
    ("cpu", True, "cpu"),
    ("cpu", False, "cpu"),
])
def test_packed_reference_preserves_explicit_device(name, available, expected, monkeypatch):
    """The selected GPU oracle must not silently switch to a different reference device."""
    import torch
    from families.qwen3_5_moe.tests.hf_reference import _reference_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    if expected is None:
        with pytest.raises(RuntimeError, match="requires an available CUDA device"):
            _reference_device(name)
    else:
        assert str(_reference_device(name)) == expected


def test_packed_reference_rejects_unknown_device():
    from families.qwen3_5_moe.tests.hf_reference import _reference_device

    with pytest.raises(ValueError, match="reference device"):
        _reference_device("automatic")


@pytest.mark.parametrize("dtype,expected", [("bfloat16", "bf16"), ("float16", "fp16")])
def test_packed_reference_honors_checkpoint_precision(tmp_path, dtype, expected):
    """The official BF16 fixture must not be silently narrowed to FP16."""
    from families.qwen3_5_moe.tests.hf_reference import _reference_precision

    fixture = Path(__file__).parent / "data/gptq-config.json"
    config = json.loads(fixture.read_text())
    assert config["text_config"]["dtype"] == "bfloat16"
    config["text_config"]["dtype"] = dtype
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _reference_precision(tmp_path) == expected


@pytest.mark.parametrize("dtype", [None, "float32", "int8"])
def test_packed_reference_rejects_undeclared_or_unsupported_precision(tmp_path, dtype):
    """The official Torch packed backend supports FP16/BF16, not a guessed dtype."""
    from families.qwen3_5_moe.tests.hf_reference import _reference_precision

    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"dtype": dtype}}))
    with pytest.raises(ValueError, match="checkpoint dtype"):
        _reference_precision(tmp_path)


@pytest.mark.parametrize("times_out", [False, True])
def test_failed_validation_preserves_diagnostics(tmp_path, capsys, times_out):
    """A failing child remains failed while its bounded diagnostic reaches CI evidence."""
    import os
    import subprocess
    import sys
    from families.qwen3_5_moe.tests.edge_validation import run_driver

    log = tmp_path / "reference.log"
    command = [sys.executable, "-c", (
        "import os, sys, time; "
        "print('x' * 70000); "
        "print(os.environ['TRTMC_DIAGNOSTIC_FIXTURE'], flush=True); "
        + ("time.sleep(10)" if times_out else "sys.exit(7)")
    )]
    error = subprocess.TimeoutExpired if times_out else subprocess.CalledProcessError
    with pytest.raises(error) as failure:
        run_driver(command, log, env={**os.environ, "TRTMC_DIAGNOSTIC_FIXTURE": "reference failure"},
                   timeout=1 if times_out else 30)
    assert failure.value.cmd == command
    assert json.loads(log.with_suffix(".command.json").read_text()) == command
    if not times_out:
        assert failure.value.returncode == 7
    captured = capsys.readouterr().out
    assert "reference failure" in captured and len(captured) <= 65536 + 256
    assert log.stat().st_size > 65536 and log.read_text().endswith("reference failure\n")


@pytest.mark.parametrize("zeros,declared,expected", [
    ([7, 7], None, "gptq"),
    ([8, 8], None, "gptq_v2"),
    ([8, 8], "gptq_v2", "gptq_v2"),
    ([7, 8], None, None),
    ([9, 9], None, None),
    ([], None, None),
    ([8, 8], "gptq", None),
])
def test_packed_reference_requires_consistent_source_format(tmp_path, zeros, declared, expected):
    """Literal zero8 must not receive the v1 zero-minus-one conversion."""
    import numpy as np
    from safetensors.numpy import save_file
    from families.qwen3_5_moe.tests.hf_reference import _reference_format

    quant = {"quant_method": "gptq", "bits": 4, "sym": True}
    if declared is not None:
        quant["checkpoint_format"] = declared
    config = json.dumps({"quantization_config": quant})
    (tmp_path / "config.json").write_text(config)
    tensors = {
        f"expert.{i}.qzeros": np.full((1, 1), zero * 0x11111111, np.uint32).view(np.int32)
        for i, zero in enumerate(zeros)
    }
    if tensors:
        save_file(tensors, tmp_path / "weights.safetensors")
    index = dict.fromkeys(tensors, "weights.safetensors")
    if expected is None:
        with pytest.raises(ValueError):
            _reference_format(tmp_path, index)
    else:
        assert _reference_format(tmp_path, index) == expected
    assert (tmp_path / "config.json").read_text() == config
