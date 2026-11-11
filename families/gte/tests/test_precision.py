# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GTE-v1.5 build precision: fp32 (default) and fp16 are accepted, other precisions are not."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from families.gte import config as config_module, model as model_module
from tensorrt_model_connect import BuildRequest

_HIDDEN, _INTERMEDIATE, _LAYERS, _HEADS, _VOCAB = 64, 128, 2, 4, 50


def _tiny_weights(seed: int = 0) -> dict:
    """A small random GTE-v1.5 checkpoint with every tensor ``build_engine`` reads."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def norm():
        return (1.0 + rng.standard_normal(_HIDDEN) * 0.05).astype(np.float32)

    weights = {
        "embeddings.word_embeddings.weight": mat(_VOCAB, _HIDDEN) * 20,
        "embeddings.LayerNorm.weight": norm(),
        "embeddings.LayerNorm.bias": mat(_HIDDEN),
    }
    for index in range(_LAYERS):
        prefix = f"encoder.layer.{index}."
        weights[f"{prefix}attention.qkv_proj.weight"] = mat(3 * _HIDDEN, _HIDDEN)
        weights[f"{prefix}attention.qkv_proj.bias"] = mat(3 * _HIDDEN)
        weights[f"{prefix}attention.o_proj.weight"] = mat(_HIDDEN, _HIDDEN)
        weights[f"{prefix}attention.o_proj.bias"] = mat(_HIDDEN)
        weights[f"{prefix}mlp.up_gate_proj.weight"] = mat(2 * _INTERMEDIATE, _HIDDEN)
        weights[f"{prefix}mlp.down_proj.weight"] = mat(_HIDDEN, _INTERMEDIATE)
        weights[f"{prefix}mlp.down_proj.bias"] = mat(_HIDDEN)
        for name in ("attn_ln", "mlp_ln"):
            weights[f"{prefix}{name}.weight"] = norm()
            weights[f"{prefix}{name}.bias"] = mat(_HIDDEN)
    return weights


def _write_checkpoint(path: Path) -> Path:
    config = {
        "model_type": "new",
        "architectures": ["NewModel"],
        "hidden_act": "gelu",
        "layer_norm_type": "layer_norm",
        "position_embedding_type": "rope",
        "num_attention_heads": _HEADS,
        "type_vocab_size": 0,
        "max_position_embeddings": 8192,
        "rope_theta": 500000,
        "rope_scaling": {"type": "ntk", "factor": 2.0},
        "pack_qkv": True,
        "layer_norm_eps": 1e-12,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(_tiny_weights(), str(path / "model.safetensors"))
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


class _Writer:
    """Collects what ``build`` writes."""

    def __init__(self):
        self.files = {}

    def set_header(self, **_):
        pass

    def add_bytes(self, name, data):
        self.files[name] = data

    def add_json(self, name, data):
        self.files[name] = data


def _request(model_dir: Path, precision: str) -> BuildRequest:
    return BuildRequest(
        model_dir=model_dir,
        output_path=model_dir / "gte.bundle",
        family="gte",
        task="text_to_embedding",
        precision=precision,
        max_sequence_length=16,
    )


def _capture_precision(monkeypatch) -> list[str]:
    seen: list[str] = []

    def fake_engine(weights, cfg, *, seq_len, precision="fp32", verbose=False, quant_ctx=None):
        seen.append(precision)
        return b"plan"

    monkeypatch.setattr(model_module, "build_engine", fake_engine)
    return seen


@pytest.mark.parametrize("precision", ["fp32", "fp16", "FP16"])
def test_fp32_and_fp16_reach_the_engine_builder(tmp_path, monkeypatch, precision) -> None:
    """The requested work type reaches ``build_engine``; the bundle contents do not change."""
    seen = _capture_precision(monkeypatch)
    writer = _Writer()
    model_module.build(_request(_write_checkpoint(tmp_path), precision), writer)
    assert seen == [precision.lower()]
    assert sorted(writer.files) == ["model.plan", "runtime.json", "tokenizer.json"]
    assert writer.files["runtime.json"] == {
        "max_sequence_length": 16,
        "hidden_size": _HIDDEN,
        "pad_token_id": 0,
    }


@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8"])
def test_other_precisions_are_refused(tmp_path, monkeypatch, precision) -> None:
    """Only fp32 and fp16 are measured; anything else is refused before the checkpoint is read."""
    seen = _capture_precision(monkeypatch)
    with pytest.raises(NotImplementedError, match="fp32 or fp16"):
        model_module.build(_request(tmp_path, precision), _Writer())
    assert seen == []


def test_the_default_precision_is_fp32() -> None:
    from families.gte.support import describe
    from tensorrt_model_connect.model_support import ModelMetadata

    support = describe(ModelMetadata(config={"model_type": "new"}, model_index={}))
    assert support.default_precision == "fp32"


def _network_casts(trt, weights, cfg, precision) -> int:
    """Number of Cast layers in the network ``build_engine`` hands to the builder."""
    counted = []
    original = trt.Builder.build_serialized_network

    def count(self, network, config):
        counted.append(
            sum(
                1
                for i in range(network.num_layers)
                if network.get_layer(i).type == trt.LayerType.CAST
            )
        )
        return original(self, network, config)

    trt.Builder.build_serialized_network = count
    try:
        model_module.build_engine(weights, cfg, seq_len=16, precision=precision)
    finally:
        trt.Builder.build_serialized_network = original
    return counted[0]


def _gpu_trt():
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    return trt, torch


def test_fp16_casts_only_around_the_float32_parts(tmp_path) -> None:
    """fp32 keeps its one Cast (the mask). fp16 adds a pair around every layer norm, gelu and
    attention softmax, and one at the output."""
    trt, _ = _gpu_trt()
    weights = _tiny_weights()
    cfg = config_module.resolve(_write_checkpoint(tmp_path), weights)
    norms = 1 + 2 * _LAYERS
    assert _network_casts(trt, weights, cfg, "fp32") == 1
    assert _network_casts(trt, weights, cfg, "fp16") == 1 + 2 * (norms + 2 * _LAYERS) + 1


def test_an_fp16_engine_keeps_the_io_and_the_values(tmp_path) -> None:
    """fp16 changes only the work type: int32 in, float32 out, close to the fp32 engine."""
    trt, torch = _gpu_trt()
    weights = _tiny_weights()
    cfg = config_module.resolve(_write_checkpoint(tmp_path), weights)
    window, length = 16, 11
    ids = torch.zeros(window, dtype=torch.int32, device="cuda")
    ids[:length] = torch.tensor([1, 5, 6, 32, 7, 8, 9, 32, 10, 11, 2], dtype=torch.int32)
    mask = torch.zeros(window, dtype=torch.int32, device="cuda")
    mask[:length] = 1

    def run(precision):
        plan = model_module.build_engine(weights, cfg, seq_len=window, precision=precision)
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        io = {
            name: (engine.get_tensor_dtype(name), tuple(engine.get_tensor_shape(name)))
            for name in names
        }
        hidden = torch.empty((window, _HIDDEN), dtype=torch.float32, device="cuda")
        context = engine.create_execution_context()
        for name, tensor in (
            ("input_ids", ids),
            ("attention_mask", mask),
            ("last_hidden_state", hidden),
        ):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return io, hidden[:length].cpu()

    io32, hidden32 = run("fp32")
    io16, hidden16 = run("fp16")
    assert (
        io16
        == io32
        == {
            "input_ids": (trt.int32, (window,)),
            "attention_mask": (trt.int32, (window,)),
            "last_hidden_state": (trt.float32, (window, _HIDDEN)),
        }
    )
    assert torch.isfinite(hidden16).all()
    cls32, cls16 = hidden32[0], hidden16[0]
    assert float((hidden16 - hidden32).abs().max()) < 2e-2
    assert float(torch.nn.functional.cosine_similarity(cls16, cls32, dim=0)) > 0.9999
