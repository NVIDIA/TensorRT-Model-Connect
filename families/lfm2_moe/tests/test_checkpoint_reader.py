# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit safetensors reader and model-build argument wiring."""

import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

from families.lfm2_moe import checkpoint_mapper


@pytest.mark.parametrize("sharded", [False, True])
def test_numpy_safetensors_preserves_loaded_values(tmp_path, monkeypatch, sharded):
    arrays = {
        "model.weight": np.array([[1.5, -2.25], [0.125, 16]], dtype=ml_dtypes.bfloat16),
        "model.bias": np.array([0.03125, -0.2], dtype=np.float32),
    }
    if sharded:
        names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
        mapping = dict(zip(arrays, names, strict=True))
        for tensor, filename in mapping.items():
            save_file({tensor: arrays[tensor]}, str(tmp_path / filename))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_file(arrays, str(tmp_path / "model.safetensors"))
    original = checkpoint_mapper.safe_open
    frameworks = []

    def recording_open(path, *, framework):
        frameworks.append(framework)
        return original(path, framework=framework)

    monkeypatch.setattr(checkpoint_mapper, "safe_open", recording_open)
    readers = checkpoint_mapper._open_safetensors(tmp_path)
    assert frameworks == ["numpy"] * (2 if sharded else 1)
    for name, values in arrays.items():
        actual = checkpoint_mapper._load_tensor(readers, name)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, values.astype(np.float32))


def test_reader_does_not_guess_another_checkpoint_format(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a supported checkpoint")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        checkpoint_mapper._open_safetensors(tmp_path)


def test_build_wires_checkpoint_precision_and_capacity(tmp_path, monkeypatch):
    pytest.importorskip("tensorrt")  # Bindings only; graph construction is replaced below.
    from families.lfm2_moe import model

    raw = json.loads((Path(__file__).parent / "fixtures/lfm25-8b-a1b-config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(raw))
    sentinel = object()
    calls = []

    def load(directory, config, *, precision):
        assert Path(directory) == tmp_path and config.raw == raw
        assert precision == "bf16"
        calls.append("weights")
        return sentinel

    def engine(config, weights, capacity, **options):
        assert config.raw == raw and weights is sentinel and capacity == 128000
        assert options == {"precision": "bf16", "verbose": True, "debug_layer_outputs": False}
        calls.append("engine")
        return b"fixture plan"

    class Writer:
        def set_header(self, **values):
            self.header = values

        def add_bytes(self, name, value):
            assert (name, value) == ("engine.plan", b"fixture plan")

        def add_json(self, name, value):
            assert name == "runtime.json"
            self.runtime = value

    monkeypatch.setattr(model, "load_lfm2_moe_weights", load)
    monkeypatch.setattr(model, "build_lfm2_moe_engine", engine)
    request = SimpleNamespace(
        model_dir=tmp_path,
        task="text_generation",
        backend="trt",
        precision="bf16",
        max_sequence_length=128000,
        verbose=True,
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        tensor_parallel_size=1,
        quantization=None,
        fp32_layers=(),
        mtp_seq_len=None,
    )
    writer = Writer()
    model.build(request, writer)
    assert calls == ["weights", "engine"]
    assert writer.header == {"family": "lfm2_moe", "task": "text_generation", "backend": "trt"}
    assert writer.runtime["max_cache_length"] == 128000
    assert writer.runtime["precision"] == "bf16"
