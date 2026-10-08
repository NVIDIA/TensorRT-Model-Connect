# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for dispatch and bundle composition, not TensorRT graph proof."""

import importlib
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter


def request(tmp_path, **changes):
    base = BuildRequest(
        model_dir=tmp_path / "checkpoint", output_path=tmp_path / "model.bundle",
        family="parakeet_tdt", task="speech_transcription", precision="fp16",
    )
    return replace(base, **changes)


@pytest.mark.parametrize(("option", "value"), [
    ("family", "whisper"), ("task", "transcription"), ("backend", "trt_rtx"),
    ("precision", "bf16"), ("tensor_parallel_size", 2),
    ("context_parallel_size", 2), ("max_batch_size", 2),
    ("max_sequence_length", 128), ("image_height", 224), ("image_width", 224),
    ("video_num_frames", 8), ("quantization", "int8"),
    ("fp32_layers", (0,)),
    ("graph_transform", lambda *args: None),
])
def test_unsupported_options_fail_before_checkpoint_or_engine_loading(tmp_path, option, value):
    build = importlib.import_module("families.parakeet_tdt.model").build
    with pytest.raises(ValueError, match=option):
        build(request(tmp_path, **{option: value}), BundleWriter(tmp_path / "bad.bundle"))
    assert not (tmp_path / "bad.bundle").exists()


def test_missing_native_checkpoint_does_not_fall_back_to_nemo(tmp_path):
    build = importlib.import_module("families.parakeet_tdt.model").build
    req = request(tmp_path)
    req.model_dir.mkdir()
    (req.model_dir / "legacy.nemo").write_bytes(b"not a supported input")
    with pytest.raises(FileNotFoundError, match="config.json"):
        build(req, BundleWriter(req.output_path))


def test_dynamic_kv_cache_is_explicitly_unsupported(tmp_path):
    build = importlib.import_module("families.parakeet_tdt.model").build
    req = request(tmp_path, dynamic_kv_cache=True)
    with pytest.raises(NotImplementedError, match="does not support dynamic_kv_cache"):
        build(req, BundleWriter(req.output_path))


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_build_packages_owned_sections_without_mutating_checkpoint(tmp_path, monkeypatch, precision):
    build = importlib.import_module("families.parakeet_tdt.model").build
    req = request(tmp_path, precision=precision)
    req.model_dir.mkdir()
    fixture = Path(__file__).with_name("config.json")
    (req.model_dir / "config.json").write_bytes(fixture.read_bytes())
    (req.model_dir / "model.safetensors").write_bytes(b"weights validated in separate tests")
    (req.model_dir / "tokenizer.json").write_text('{"model":{"type":"Unigram"}}')
    before = {p.name: p.read_bytes() for p in req.model_dir.iterdir()}
    seen = []

    def compile_engines(model_dir, *, precision, verbose):
        seen.append((model_dir, precision, verbose))
        return {"encoder.plan": b"encoder", "predictor.plan": b"predictor",
                "joint.plan": b"joint", "mel_filterbank": b"mel"}, {"tdt_blank_id": 8192}

    monkeypatch.setitem(sys.modules, "families.parakeet_tdt.engines",
                        SimpleNamespace(compile_engines=compile_engines))
    writer = BundleWriter(req.output_path)
    build(req, writer)
    writer.finish()
    data = req.output_path.read_bytes()
    size = struct.unpack_from("<Q", data, 8)[0]
    header = json.loads(data[16:16 + size])
    assert header["family"] == "parakeet_tdt"
    assert header["task"] == "speech_transcription"
    assert header["backend"] == "trt"
    sections = set(header["sections"])
    assert sections == {"runtime.json", "encoder.plan", "predictor.plan", "joint.plan",
                        "mel_filterbank", "tokenizer.json"}
    assert seen == [(req.model_dir, precision, False)]
    assert before == {p.name: p.read_bytes() for p in req.model_dir.iterdir()}


def test_engine_composition_uses_all_three_plans_and_requires_frontend(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    engines = importlib.import_module("families.parakeet_tdt.engines")
    config = Path(__file__).with_name("config.json").read_bytes()
    (tmp_path / "config.json").write_bytes(config)
    weights = {"example": 1}
    monkeypatch.setattr(engines, "_load_weights", lambda *a, **kw: (weights, {"tdt_blank_id": 8192}))
    seen = []

    def graph(name):
        def compile(*args, **kwargs):
            assert args[-1] is weights
            assert kwargs == {"precision": "fp16", "verbose": True}
            seen.append(name)
            return name.encode()
        return compile

    for name in ("encoder", "predictor", "joint"):
        monkeypatch.setattr(engines, "_build_" + name, graph(name))
    monkeypatch.setattr(engines, "_build_mel_filterbank", lambda *a, **kw: b"mel")
    plans, runtime = engines.compile_engines(tmp_path, precision="fp16", verbose=True)
    assert plans == {"encoder.plan": b"encoder", "predictor.plan": b"predictor",
                     "joint.plan": b"joint", "mel_filterbank": b"mel"}
    assert set(seen) == {"encoder", "predictor", "joint"}
    assert runtime["tdt_blank_id"] == 8192
    assert runtime["mel_sampling_rate"] == 16000
    monkeypatch.setattr(engines, "_build_mel_filterbank", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="mel filterbank"):
        engines.compile_engines(tmp_path, precision="fp16", verbose=True)


@pytest.mark.parametrize(("key", "error", "message"), [
    ("unrelated", KeyError, "missing normalized tensors"),
    ("prompt_kernel.0.weight", ValueError, "prompt_kernel"),
])
def test_native_checkpoint_loader_rejects_unsupported_weights(tmp_path, monkeypatch, key, error, message):
    import numpy as np
    from safetensors.numpy import save_file

    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    engines = importlib.import_module("families.parakeet_tdt.engines")
    (tmp_path / "config.json").write_bytes(Path(__file__).with_name("config.json").read_bytes())
    save_file({key: np.zeros(1, dtype=np.float32)}, tmp_path / "model.safetensors")
    with pytest.raises(error, match=message):
        engines._load_hf_as_nemo(str(tmp_path))


def test_config_rejects_changed_duration_semantics(tmp_path):
    from families.parakeet_tdt.config import ParakeetTDTConfig

    raw = json.loads(Path(__file__).with_name("config.json").read_text())
    raw["durations"] = [0, 1, 2, 4]
    with pytest.raises(ValueError, match="durations"):
        ParakeetTDTConfig.from_json(json.dumps(raw)).validate_supported_checkpoint()


def test_weight_transpose_preserves_values_and_rejects_vectors():
    import numpy as np
    from families.parakeet_tdt.checkpoint import _transpose_2d

    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = _transpose_2d(matrix, "test")
    np.testing.assert_array_equal(result, matrix.T)
    assert result.flags.c_contiguous
    with pytest.raises(ValueError, match="rank-2"):
        _transpose_2d(np.zeros(3), "bad")
