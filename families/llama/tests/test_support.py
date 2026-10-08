# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of build memory lifetime and bundle publication."""

import gc
import importlib.util
import json
from pathlib import Path
import struct
import sys
import types
import weakref

import numpy as np
import pytest
from safetensors.numpy import save_file

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter

from .. import checkpoint_mapper
from ..config import ModelConfig

@pytest.fixture
def request_and_writer(tmp_path, monkeypatch):
    # Jiaxin Deng: stub only GPU compilation; exercise the real build and writer on CPU.
    with monkeypatch.context() as imports:
        for module_name, function_name in (
            ("dual_profile_decoder_builder", "build_dual_profile_decoder_engine"),
            ("standard_decoder_builder", "build_standard_decoder_engine"),
        ):
            module = types.ModuleType(f"families.llama.{module_name}")
            setattr(module, function_name, None)
            imports.setitem(sys.modules, module.__name__, module)
        spec = importlib.util.spec_from_file_location(
            "families.llama._plan_lifetime_model", Path(__file__).parents[1] / "model.py"
        )
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "vocab_size": 32,
        "max_position_embeddings": 32, "eos_token_id": 2,
    }))
    monkeypatch.setattr(model, "load_standard_weights", lambda *args, **kwargs: {})
    destination = tmp_path / "model.bundle"
    request = BuildRequest(
        model_dir=checkpoint, output_path=destination, family="llama",
        task="text_generation", precision="fp16", max_sequence_length=16,
    )
    writer = BundleWriter(destination)
    yield model, request, writer
    writer.abort()


def test_prefill_plan_is_released_before_decode_build(request_and_writer, monkeypatch):
    model, request, writer = request_and_writer
    released = []

    class Plan(bytes):
        def __del__(self):
            released.append(True)

    def build_engine(config, *args, **kwargs):
        if config.raw["_decoder_engine_role"] == "prefill":
            return Plan(b"prefill bytes")
        gc.collect()
        assert released, "prefill plan is still resident during decode build"
        return b"decode bytes"

    monkeypatch.setattr(model, "_build_engine", build_engine)
    model.build(request, writer)
    writer.finish()
    data = request.output_path.read_bytes()
    header_size = struct.unpack("<Q", data[8:16])[0]
    header = json.loads(data[16:16 + header_size])
    payload = data[16 + header_size:]
    sections = {
        name: payload[entry["offset"]:entry["offset"] + entry["length"]]
        for name, entry in header["sections"].items()
    }
    assert sections["engine.plan"] == b"decode bytes"
    assert sections["prefill.plan"] == b"prefill bytes"
    assert json.loads(sections["runtime.json"])["decoder_engine_layout"] == "split"


def test_decode_failure_preserves_published_bundle(request_and_writer, monkeypatch):
    model, request, writer = request_and_writer
    request.output_path.write_bytes(b"previous bundle")

    def build_engine(config, *args, **kwargs):
        if config.raw["_decoder_engine_role"] == "prefill":
            return b"prefill bytes"
        raise RuntimeError("decode build failed")

    monkeypatch.setattr(model, "_build_engine", build_engine)
    with pytest.raises(RuntimeError, match="decode build failed"):
        model.build(request, writer)
    writer.abort()
    assert request.output_path.read_bytes() == b"previous bundle"
    assert not list(request.output_path.parent.glob(".model.bundle.sections.*"))


@pytest.mark.parametrize("precision,fp32_layers", [("fp16", ()), ("fp32", ()), ("fp16", (1,))])
@pytest.mark.parametrize("tied", [False, True])
def test_projection_scratch_is_released_without_changing_weights(
    tmp_path, monkeypatch, precision, fp32_layers, tied
):
    config = ModelConfig(
        model_type="llama", hidden_size=4, vocab_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, intermediate_size=6,
    )
    embedding = np.arange(32, dtype=np.float32).reshape(8, 4) / 7 - 2
    tensors = {"model.embed_tokens.weight": embedding}
    if not tied:
        tensors["lm_head.weight"] = embedding + 3
    shapes = {
        "self_attn.q_proj": (4, 4), "self_attn.k_proj": (2, 4),
        "self_attn.v_proj": (2, 4), "self_attn.o_proj": (4, 4),
        "mlp.gate_proj": (6, 4), "mlp.up_proj": (6, 4), "mlp.down_proj": (4, 6),
    }
    output_names = ("w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down")
    expected = {}
    for index in range(2):
        for projection_index, (source, output) in enumerate(zip(shapes, output_names)):
            shape = shapes[source]
            values = (
                np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 7
                + index * 3 + projection_index / 9 - 2
            ).astype(np.float16)
            tensors[f"model.layers.{index}.{source}.weight"] = values
            dtype = np.float32 if precision == "fp32" or index in fp32_layers else np.float16
            expected[f"layer.{index}.{output}"] = values.astype(np.float32).T.astype(dtype)
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"model.layers.{index}.{name}.weight"] = np.ones(4, dtype=np.float16)
    save_file(tensors, tmp_path / "model.safetensors")
    load_tensor = checkpoint_mapper._load_tensor
    previous = {}
    embedding_source = None

    def tracked_load(readers, name):
        nonlocal embedding_source
        if name == "model.embed_tokens.weight":
            value = load_tensor(readers, name)
            embedding_source = weakref.ref(value)
            return value
        assert embedding_source() is None, "embedding scratch is still live"
        if ".layers." in name and name.endswith("_proj.weight"):
            layer = name.split(".")[2]
            if layer in previous:
                assert previous[layer]() is None, "previous projection scratch is still live"
            value = load_tensor(readers, name)
            previous[layer] = weakref.ref(value)
            return value
        return load_tensor(readers, name)

    monkeypatch.setattr(checkpoint_mapper, "_load_tensor", tracked_load)
    weights = checkpoint_mapper.load_standard_weights(
        tmp_path, config, precision=precision, fp32_layers=fp32_layers
    )
    dtype = np.float32 if precision == "fp32" else np.float16
    np.testing.assert_array_equal(weights["embedding"], embedding.astype(dtype))
    output = embedding if tied else tensors["lm_head.weight"]
    np.testing.assert_array_equal(weights["w_out"], output.T.astype(dtype))
    assert weights["w_out"].dtype == dtype
    assert weights["w_out"].flags.c_contiguous
    for name, values in expected.items():
        np.testing.assert_array_equal(weights[name], values)
        assert weights[name].dtype == values.dtype
        assert weights[name].flags.c_contiguous
    assert weights["_attention_size"] == 4
    assert weights["_kv_attention_size"] == 2
    assert weights["_mlp_size"] == 6
