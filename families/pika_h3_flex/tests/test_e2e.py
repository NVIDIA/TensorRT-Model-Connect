# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned native video and stereo-audio parity through the public Task API."""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_enabled, evidence_stage, record_evidence

FAMILY = "pika_h3_flex"
TASKS = frozenset({"text_to_audio_video"})
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
    build_dir = _required_path(os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR")
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", "2", "--target", name],
        check=True,
        timeout=600,
    )
    return _required_path(str(build_dir / name), name)


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
    for text in (
        "",
        "A quiet mountain lake.",
        "Grüße 世界 😀",
        "café cafe\u0301",
        "1234567890",
        "We're ready.\t\n",
        "<|im_start|>user\nHello<|im_end|>",
    ):
        result = subprocess.run(
            [
                str(_native_test_binary("test_pika_h3_flex_tokenizer")),
                str(directory / "tokenizer/tokenizer.json"),
                text,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert json.loads(result.stdout) == reference.encode(text, add_special_tokens=False)


def test_cpu_generator_stream(case_name, tmp_path):
    import numpy as np
    import torch

    _, _, case = CASES[case_name]
    counts = [24 * 37 * 30 * 52, 414 * 32, 24 * 37 * 30 * 52, 414 * 32]
    for seed in (42, 43):
        path = tmp_path / f"noise-{seed}.f32"
        result = subprocess.run(
            [str(_native_test_binary("test_pika_h3_flex_random")), str(path)],
            input=json.dumps(dict(seed=seed, counts=counts)),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        generator = torch.Generator("cpu").manual_seed(seed)
        expected = np.concatenate(
            [torch.randn(count, generator=generator).numpy() for count in counts]
        )
        actual = np.fromfile(path, dtype="<f4")
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        assert np.max(np.abs(actual - expected)) <= case["noise_max_abs"]


def _video_metrics(actual, expected):
    import numpy as np

    square, dot, a2, b2, maximum = 0.0, 0.0, 0.0, 0.0, 0.0
    for a, b in zip(actual, expected):
        assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
        a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        difference = a - b
        square += float(difference @ difference)
        dot += float(a @ b)
        a2 += float(a @ a)
        b2 += float(b @ b)
        maximum = max(maximum, float(np.abs(difference).max()))
    return float(np.sqrt(square / actual.size)), maximum, dot / np.sqrt(a2 * b2)


def test_official_checkpoint_e2e(case_name, tmp_path):
    _, manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    if evidence_enabled():
        for key in ("image", "test_image", "audio"):
            if key in case:
                for source in (TEST_ROOT / case[key], TEST_ROOT / "data" / case[key]):
                    if source.is_file():
                        record_evidence("inputs", {key: source})
                        break
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
            if any(word in key for word in ("error", "cosine", "atol", "rtol", "snr", "rmse"))
        },
    )
    import gc
    import sys
    import numpy as np
    import torch

    _, manifest, case = CASES[case_name]
    with evidence_stage("checkpoint"):
        directory = _model_dir(manifest)
    _, runtime_root = _runtime()
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
        reference_log = tmp_path / "reference-process.log"
        reference_log.write_text(oracle.stderr)
        record_evidence(
            "reference_process",
            {
                "argv": oracle.args,
                "stdout": oracle.stdout,
                "stderr": reference_log,
                "returncode": oracle.returncode,
            },
        )
        assert oracle.returncode == 0, oracle.stderr + oracle.stdout
        print(oracle.stdout, end="", flush=True)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(directory, bundle, manifest)
    record_evidence("bundle", {"path": str(bundle), "manifest": manifest})
    gc.collect()
    torch.cuda.empty_cache()
    requests = [
        dict(
            prompt=prompt,
            seed=case["seed"] + index,
            video=str(tmp_path / f"native-video-{index}.f32"),
            audio=str(tmp_path / f"native-audio-{index}.f32"),
        )
        for index, prompt in enumerate(case["prompts"])
    ]
    with evidence_stage("runtime"):
        native = subprocess.run(
            [str(_native_test_binary("test_pika_h3_flex_runtime")), str(bundle), str(runtime_root)],
            input=json.dumps(requests),
            capture_output=True,
            text=True,
            check=False,
            timeout=case["runtime_timeout_s"],
        )
        native_log = tmp_path / "native-process.log"
        native_log.write_text(native.stderr)
        record_evidence(
            "native_process",
            {
                "argv": native.args,
                "stdout": native.stdout,
                "stderr": native_log,
                "stdin": json.dumps(requests),
                "returncode": native.returncode,
            },
        )
        assert native.returncode == 0, native.stderr + native.stdout
        results = json.loads(native.stdout.splitlines()[-1])
        assert len(results) == len(requests)
    with evidence_stage("comparison"):
        for index, result in enumerate(results):
            record_evidence(
                "inputs", {"prompt": case["prompts"][index], "seed": case["seed"] + index}
            )
            assert result["frames"] == 124 and result["height"] == 480 and result["width"] == 832
            assert (
                result["sample_rate"] == 32000
                and result["channels"] == 2
                and result["audio_count"] == 165333 * 2
            )
            assert result["audio_start_seconds"] == 0 and result["conditioned_prefix_frames"] == 0
            np.testing.assert_allclose(
                result["timestamps"], np.arange(124, dtype=np.float64) / 24, atol=1e-12, rtol=0
            )
            expected_video = np.memmap(
                tmp_path / f"reference-video-{index}.f32",
                dtype="<f4",
                mode="r",
                shape=(124, 480, 832, 3),
            )
            actual_video = np.memmap(
                requests[index]["video"], dtype="<f4", mode="r", shape=expected_video.shape
            )
            expected_audio = np.fromfile(
                tmp_path / f"reference-audio-{index}.f32", dtype="<f4"
            ).astype(np.float64)
            actual_audio = np.fromfile(requests[index]["audio"], dtype="<f4").astype(np.float64)
            if evidence_enabled():
                import struct
                from PIL import Image

                views = []
                selected = [0, 62, 123]
                preview_indices = [0, 31, 62, 93, 123]
                for role, video, audio in (
                    ("native", actual_video, actual_audio),
                    ("reference", expected_video, expected_audio),
                ):
                    # The full-resolution oracle above still compares all 124 frames.
                    # Retain lossless representative tensors and label the GIF as a preview.
                    frames_path = tmp_path / f"{role}-{index}-selected-frames.npz"
                    np.savez_compressed(frames_path, frames=np.stack([video[i] for i in selected]))
                    frames = [
                        Image.fromarray(
                            np.rint(np.clip(video[i], 0, 1) * 255).astype(np.uint8)
                        ).resize((416, 240))
                        for i in preview_indices
                    ]
                    preview = tmp_path / f"{role}-{index}-video-preview.gif"
                    frames[0].save(
                        preview, save_all=True, append_images=frames[1:], duration=500, loop=0
                    )
                    audio_path = tmp_path / f"{role}-{index}-stereo.wav"
                    pcm = audio.astype("<f4").tobytes()
                    audio_path.write_bytes(
                        struct.pack(
                            "<4sI4s4sIHHIIHH4sI",
                            b"RIFF",
                            36 + len(pcm),
                            b"WAVE",
                            b"fmt ",
                            16,
                            3,
                            2,
                            32000,
                            256000,
                            8,
                            32,
                            b"data",
                            len(pcm),
                        )
                        + pcm
                    )
                    record_evidence(
                        role,
                        {
                            "sample": index,
                            "prompt": case["prompts"][index],
                            **({"mode": "official_diffusers"} if role == "reference" else {}),
                            "metadata": result
                            if role == "native"
                            else {
                                "video_shape": list(video.shape),
                                "sample_rate": 32000,
                                "channels": 2,
                            },
                            "selected_frames": frames_path,
                            "retained_frame_indices": selected,
                            "full_video_shape": list(video.shape),
                            "waveform": audio.astype(np.float32),
                            "video_preview_scope": {
                                "frame_indices": preview_indices,
                                "width": 416,
                                "height": 240,
                            },
                        },
                    )
                    views.extend(
                        [
                            {
                                "title": f"Sample {index} / {role}",
                                "image": preview,
                                "caption": "Half-resolution preview of frames 0,31,62,93,123; metrics cover all 124 frames.",
                            },
                            {
                                "title": f"Sample {index} / {role}",
                                "audio": audio_path,
                                "caption": "Complete 32 kHz stereo waveform.",
                            },
                        ]
                    )
                record_evidence("views", views)

            rmse, maximum, cosine = _video_metrics(actual_video, expected_video)
            print(
                f"Pika video {index}: RMSE={rmse}, max_abs={maximum}, cosine={cosine}", flush=True
            )
            video_metrics = {"rmse": rmse, "max_abs_error": maximum, "cosine": cosine}
            record_evidence("metrics", {"sample": index, "video": video_metrics})
            assert (
                rmse <= case["video_rmse_max"]
                and maximum <= case["video_max_abs"]
                and cosine >= case["video_cosine_min"]
            )
            assert actual_audio.shape == expected_audio.shape and np.isfinite(actual_audio).all()
            assert actual_audio.min() >= -1 and actual_audio.max() <= 1
            difference = actual_audio - expected_audio
            energy, error = float(expected_audio @ expected_audio), float(difference @ difference)
            snr = float(10 * np.log10(energy / error)) if error else float("inf")
            cosine = float(
                actual_audio
                @ expected_audio
                / (np.linalg.norm(actual_audio) * np.linalg.norm(expected_audio))
            )
            maximum = float(np.abs(difference).max())
            print(
                f"Pika audio {index}: SNR={snr} dB, max_abs={maximum}, cosine={cosine}", flush=True
            )
            record_evidence(
                "metrics",
                {
                    "sample": index,
                    "video": video_metrics,
                    "audio": {"snr_db": snr, "cosine": cosine, "max_abs_error": maximum},
                },
            )
            assert (
                snr >= case["audio_snr_min_db"]
                and cosine >= case["audio_cosine_min"]
                and maximum <= case["audio_max_abs"]
            )


@pytest.mark.parametrize("rotary_dim", [96, 128])
def test_bf16_rope_matches_pytorch(case_name, rotary_dim, tmp_path):
    import tensorrt as trt
    import torch
    from families.pika_h3_flex.graph import Graph

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
    from families.pika_h3_flex.graph import Graph

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
