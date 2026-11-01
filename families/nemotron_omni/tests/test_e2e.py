# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual public build/task inference against a pinned independent vLLM oracle.

Local full-checkpoint execution requires explicit selection. The native profile
is the declared premerge case; small graph/frontend tests also run independently. An executed accuracy
failure remains a failure, not an xfail or a reason to suppress the adapter.
"""

from pathlib import Path
import json
import os
import subprocess
import wave

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

ROOT = Path(__file__).resolve().parent
FAMILY = "nemotron_omni"
REFERENCE = json.loads((ROOT / "data/reference.json").read_text())
CASES = {}
for path in sorted((ROOT / "manifests").glob("*.json")):
    manifest = json.loads(path.read_text())
    for case in manifest["testcases"]:
        CASES[case["name"]] = manifest, case


def _selection(config, name):
    return {
        part.strip()
        for value in (config.getoption(name, default=[]) or [])
        for part in value.split(",")
        if part.strip()
    }


def _selected(config, manifest, case):
    models = _selection(config, "--e2e-model")
    path = config.getoption("--e2e-models-file", default=None)
    if path:
        models.update(
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    cases = _selection(config, "--e2e-testcase")
    if not models and not cases and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("full-checkpoint Omni E2E requires explicit selection")
    if models and not models.intersection({FAMILY, manifest["name"], case["name"]}):
        pytest.skip("model profile was not selected")
    if cases and case["name"] not in cases:
        pytest.skip("testcase was not selected")


def _checkpoint(manifest):
    explicit = os.environ.get("TRTMC_NEMOTRON_OMNI_MODEL_DIR")
    if explicit:
        root = Path(explicit)
        assert (root / "model.safetensors.index.json").is_file()
        return root
    from huggingface_hub import hf_hub_download, snapshot_download

    index = hf_hub_download(
        manifest["hf_id"], "model.safetensors.index.json", revision=manifest["hf_revision"]
    )
    shards = set(json.loads(Path(index).read_text())["weight_map"].values())
    return Path(
        snapshot_download(
            manifest["hf_id"],
            revision=manifest["hf_revision"],
            allow_patterns=["*.json", "*.jinja", *sorted(shards)],
        )
    )


def _asset(name):
    spec = REFERENCE["assets"][name]
    if "local" in spec:
        path = ROOT / "data" / spec["local"]
    elif os.environ.get("TRTMC_NEMOTRON_OMNI_ASSET_DIR"):
        path = Path(os.environ["TRTMC_NEMOTRON_OMNI_ASSET_DIR"]) / spec["filename"]
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(spec["repo_id"], spec["filename"], revision=spec["revision"]))
    assert path.is_file(), name
    return path


def make_inputs(ids, directory):
    """Preserve the exact public fixture pixels/PCM, frame sampling and timing."""
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    expected = [row for row in REFERENCE["cases"] if row["id"] in ids]
    kinds = {kind for row in expected for kind in row["modalities"]}
    media = {}
    if "image" in kinds:
        image = Image.open(_asset("example_reasoning_input.png")).convert("RGB")
        path = directory / "image.rgb"
        np.asarray(image).tofile(path)
        media["image"] = {"path": str(path), "height": image.height, "width": image.width}
    if "audio" in kinds:
        with wave.open(str(_asset("speech_16k_mono.wav")), "rb") as source:
            assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (
                1,
                2,
                16000,
            )
            pcm = (
                np.frombuffer(source.readframes(source.getnframes()), "<i2").astype("<f4") / 32768.0
            )
        path = directory / "audio.f32"
        pcm.tofile(path)
        media["audio"] = {"path": str(path), "sample_rate": 16000, "channels": 1}
    if "video" in kinds:
        spec = REFERENCE["video"]
        decoded = {
            i: np.asarray(Image.open(_asset(f"video-frame-{i}.png")).convert("RGB"))
            for i in spec["frame_indices"]
        }
        frames = []
        for i, timestamp in zip(spec["frame_indices"], spec["timestamps_seconds"], strict=True):
            pixels = decoded[i]
            assert abs(i / spec["original_fps"] - timestamp) < 1e-6
            path = directory / f"frame-{i}.rgb"
            pixels.tofile(path)
            frames.append({"path": str(path), "height": pixels.shape[0], "width": pixels.shape[1]})
        media["video"] = {"frames": frames, "timestamps": spec["timestamps_seconds"]}
    requests = []
    for row in expected:
        item = {
            "id": row["id"],
            "prompt": row["prompt"],
            "max_new_tokens": 96,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "use_chat_template": True,
            "enable_thinking": False,
            **{kind: media[kind] for kind in row["modalities"]},
        }
        if row["id"] == "audio_video":
            item["audio_start_seconds"] = 0.0
        requests.append(item)
    return requests, expected


def _ned(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = current
    return row[-1] / max(len(a), len(b), 1)


def _runner(runtime, target="nemotron_omni_inference_test"):
    roots = [runtime]
    configured = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    if configured:
        roots.insert(0, Path(configured))
    for name in ("libtrtmc_model_nemotron_omni.so", "trtmc_model_nemotron_omni.dll"):
        library = runtime / name
        if library.is_file():
            roots.append(library.resolve().parent)
    suffix = ".exe" if os.name == "nt" else ""
    candidates = []
    for root in roots:
        for prefix in (root, root / "families/nemotron_omni"):
            for configuration in ("", "Release", "Debug"):
                candidates.append(prefix / configuration / (target + suffix))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # A wheel-backed run may have a configured test tree but no family DSO build.
    # Build only this owning driver, never provision or reconfigure an environment.
    if configured and (Path(configured) / "CMakeCache.txt").is_file():
        subprocess.run(
            [
                "cmake",
                "--build",
                configured,
                "--target",
                target,
                "--config",
                "Release",
                "--parallel",
                "2",
            ],
            check=True,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise AssertionError(f"build the owning {target} CMake target")


@pytest.mark.parametrize("target", ["nemotron_omni_inference_test", "nemotron_omni_decoder_probe"])
def test_runner_builds_missing_declared_helper(tmp_path, monkeypatch, target):
    native = tmp_path / "native"
    native.mkdir()
    (native / "CMakeCache.txt").touch()
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(native))
    suffix = ".exe" if os.name == "nt" else ""
    executable = native / "families/nemotron_omni/Release" / (target + suffix)
    calls = []

    def build_driver(command, *, check):
        assert check
        assert command == [
            "cmake",
            "--build",
            str(native),
            "--target",
            target,
            "--config",
            "Release",
            "--parallel",
            "2",
        ]
        calls.append(command)
        executable.parent.mkdir(parents=True)
        executable.touch()

    monkeypatch.setattr(subprocess, "run", build_driver)
    assert _runner(tmp_path / "wheel", target) == executable
    assert _runner(tmp_path / "wheel", target) == executable
    assert len(calls) == 1


@pytest.mark.parametrize("target", ["nemotron_omni_inference_test", "nemotron_omni_decoder_probe"])
def test_runner_does_not_configure_an_undeclared_tree(tmp_path, monkeypatch, target):
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected build"))
    with pytest.raises(AssertionError, match="build the owning"):
        _runner(tmp_path / "wheel", target)


def test_reference_integrity(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **kw: pytest.fail("fixture download")
    )
    assert REFERENCE["reference_backend"] == "vllm==0.20.0"
    assert {row["id"] for row in REFERENCE["cases"]} == {
        "text",
        "image",
        "audio",
        "video",
        "image_audio",
        "audio_video",
    }
    assert REFERENCE["sampling"] == {"temperature": 0, "top_p": 1, "max_tokens": 96, "seed": 0}
    for name, asset in REFERENCE["assets"].items():
        if "local" in asset:
            _asset(name)
    ids = [row["id"] for row in REFERENCE["cases"]]
    requests, expected = make_inputs(ids, tmp_path / "fixtures")
    assert [row["id"] for row in requests] == [row["id"] for row in expected] == ids
    for manifest, case in CASES.values():
        assert manifest["hf_id"] == REFERENCE["model"]
        assert manifest["hf_revision"] == REFERENCE["revision"]
        assert case["reference_backend"] == "golden_snapshot"
        assert manifest["trust_remote_code"] is False
        assert case["metadata"]["golden_snapshot_path"] == "data/reference.json"
        for row in REFERENCE["cases"]:
            assert row["text"] and 0 < len(row["token_ids"]) <= 96


def _progress(capture, case_name, stage):
    # The outer CI timeout can terminate pytest before captured output or JUnit
    # is written. Keep the last entered stage visible without changing capture
    # for model output, evidence, execution limits, or acceptance criteria.
    with capture.disabled():
        print(f"Nemotron Omni E2E [{case_name}]: {stage}", flush=True)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_e2e(case_name, request, tmp_path, capfd):
    manifest, case = CASES[case_name]
    _selected(request.config, manifest, case)
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    _progress(capfd, case_name, "native helper setup")
    runner = _runner(runtime)
    _progress(capfd, case_name, "checkpoint acquisition")
    checkpoint = _checkpoint(manifest)
    _progress(capfd, case_name, "fixture preparation")
    requests, expected = make_inputs(case["modalities"], tmp_path / "fixtures")
    bundle = tmp_path / manifest["bundle"]
    _progress(capfd, case_name, "build started")
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=checkpoint,
                output_path=bundle,
                family=FAMILY,
                task=manifest["task"],
                precision=manifest["precision"],
                quantization=manifest["quantization"],
                max_sequence_length=manifest["max_sequence_length"],
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
    _progress(capfd, case_name, "build completed")
    source, destination = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(json.dumps(requests))
    command = [str(runner), str(bundle), str(runtime), str(source), str(destination)]
    env = dict(os.environ)
    library_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    env[library_path] = str(runtime) + os.pathsep + env.get(library_path, "")
    env["TMPDIR"] = str(tmp_path)
    _progress(capfd, case_name, "native inference started")
    with evidence_stage("native"):
        completed = subprocess.run(command, env=env, text=True, capture_output=True, timeout=3600)
        record_evidence("commands", {"argv": command, "returncode": completed.returncode})
        record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
        assert completed.returncode == 0, completed.stdout + completed.stderr
    _progress(capfd, case_name, "native inference completed; accuracy validation started")
    actual = json.loads(destination.read_text())
    assert len(actual) == len(expected) == len(requests)
    threshold = json.loads((ROOT / "thresholds" / f"{case_name}.json").read_text())[
        "threshold_overrides"
    ]["normalized_text_edit_distance"]
    results = []
    for a, b in zip(actual, expected, strict=True):
        assert a["id"] == b["id"] and a["status"] == "passed"
        assert a["text"] and 0 < len(a["token_ids"]) <= 96
        results.append(
            {
                "id": a["id"],
                "actual": a,
                "reference": b,
                "normalized_text_edit_distance": _ned(a["text"], b["text"]),
            }
        )
    record_evidence(
        "reference", {"backend": REFERENCE["reference_backend"], "revision": REFERENCE["revision"]}
    )
    record_evidence("metrics", {"cases": results, "normalized_text_edit_distance_max": threshold})
    assert all(row["normalized_text_edit_distance"] <= threshold for row in results), results
    _progress(capfd, case_name, "accuracy validation passed")


@pytest.mark.parametrize("fail_second", [False, True])
def test_reference_replay_is_complete_and_honest(tmp_path, monkeypatch, fail_second):
    """Mock only the replay plumbing; this is not independent model validation."""
    import sys
    from types import SimpleNamespace
    from . import record_reference

    output = tmp_path / "replay.json"
    checkpoint = tmp_path / "arbitrary-local-checkpoint"
    checkpoint.mkdir()
    reference = {**REFERENCE, "cases": [{"id": "first"}, {"id": "second"}]}
    monkeypatch.setattr(record_reference, "REFERENCE", reference)
    monkeypatch.setattr(
        record_reference,
        "make_inputs",
        lambda ids, _: ([{"id": name, "prompt": name} for name in ids], []),
    )
    calls = []

    class LLM:
        def __init__(self, **kwargs):
            assert kwargs["model"] == str(checkpoint)

        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt")

        def generate(self, *args, **kwargs):
            assert not output.exists(), "partial replay must not be published"
            calls.append(1)
            if fail_second and len(calls) == 2:
                raise RuntimeError("reference fixture failure")
            return [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="fixture", token_ids=[1], finish_reason="stop")]
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(__version__="0.20.0", LLM=LLM, SamplingParams=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(
        sys, "argv", ["record_reference", "--model-dir", str(checkpoint), "--output", str(output)]
    )
    if fail_second:
        with pytest.raises(RuntimeError, match="reference fixture failure"):
            record_reference.main()
        assert not output.exists()
    else:
        record_reference.main()
        result = json.loads(output.read_text())
        assert [row["id"] for row in result["cases"]] == ["first", "second"]
        assert result["checkpoint"] == {
            "path": str(checkpoint.resolve()),
            "identity_status": "unverified_local_directory",
        }
        assert result["expected_checkpoint"] == {
            "model": REFERENCE["model"],
            "revision": REFERENCE["revision"],
        }
        assert "revision" not in result and "model" not in result
        before = output.read_bytes()
        with pytest.raises(SystemExit):
            record_reference.main()
        assert output.read_bytes() == before
    assert len(calls) == 2
