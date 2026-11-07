# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeBERTa-v2/v3 build precision: fp32 (default) and fp16 are accepted, other precisions are not."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from tensorrt_model_connect import BuildRequest

from .. import config as config_module, model as model_module

HIDDEN, INTERMEDIATE, LAYERS, HEADS, VOCAB, SPAN, WINDOW = 128, 256, 2, 4, 64, 16, 32


def _tiny_checkpoint(seed: int = 0) -> dict:
    """A small random DeBERTa-v3 checkpoint with every tensor ``build_engine`` reads."""
    rng = np.random.default_rng(seed)

    def mat(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def norm(prefix):
        return {f"{prefix}.weight": 1.0 + mat(HIDDEN), f"{prefix}.bias": mat(HIDDEN)}

    weights = {
        "deberta.embeddings.word_embeddings.weight": rng.standard_normal((VOCAB, HIDDEN)).astype(
            np.float32
        ),
        "deberta.encoder.rel_embeddings.weight": rng.standard_normal((2 * SPAN, HIDDEN)).astype(
            np.float32
        ),
        **norm("deberta.embeddings.LayerNorm"),
        **norm("deberta.encoder.LayerNorm"),
    }
    for index in range(LAYERS):
        layer = f"deberta.encoder.layer.{index}."
        for name, rows, columns in (
            ("attention.self.query_proj", HIDDEN, HIDDEN),
            ("attention.self.key_proj", HIDDEN, HIDDEN),
            ("attention.self.value_proj", HIDDEN, HIDDEN),
            ("attention.output.dense", HIDDEN, HIDDEN),
            ("intermediate.dense", INTERMEDIATE, HIDDEN),
            ("output.dense", HIDDEN, INTERMEDIATE),
        ):
            weights[f"{layer}{name}.weight"] = mat(rows, columns)
            weights[f"{layer}{name}.bias"] = mat(rows)
        weights.update(norm(f"{layer}attention.output.LayerNorm"))
        weights.update(norm(f"{layer}output.LayerNorm"))
    return weights


_CONFIG = {
    "hidden_size": HIDDEN,
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "intermediate_size": INTERMEDIATE,
    "vocab_size": VOCAB,
    "relative_attention": True,
    "position_biased_input": False,
    "pos_att_type": ["p2c", "c2p"],
    "share_att_key": True,
    "norm_rel_ebd": "layer_norm",
    "position_buckets": SPAN,
    "max_position_embeddings": 512,
    "type_vocab_size": 0,
    "layer_norm_eps": 1e-7,
}


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    save_file(_tiny_checkpoint(), str(tmp_path / "model.safetensors"))
    return tmp_path


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
        output_path=model_dir / "deberta.bundle",
        family="deberta_v2",
        task="text_to_token_features",
        precision=precision,
        max_sequence_length=WINDOW,
    )


def _capture(monkeypatch) -> list[str]:
    seen: list[str] = []

    def fake_engine(weights, cfg, *, seq_len, precision, verbose):
        seen.append(precision)
        return b"plan"

    monkeypatch.setattr(model_module, "build_engine", fake_engine)
    monkeypatch.setattr(
        model_module,
        "_tokenizer",
        lambda directory: (b"{}", {"pad_token_id": 0, "cls_token_id": 1, "sep_token_id": 2}),
    )
    return seen


@pytest.mark.parametrize("precision", ["fp32", "fp16", "FP16"])
def test_fp32_and_fp16_reach_the_engine_builder(tiny, monkeypatch, precision) -> None:
    """The requested work type is passed to the engine builder; the bundle layout is unchanged."""
    seen = _capture(monkeypatch)
    writer = _Writer()
    model_module.build(_request(tiny, precision), writer)
    assert seen == [precision.lower()]
    assert writer.files["model.plan"] == b"plan"
    assert writer.files["runtime.json"] == {
        "max_sequence_length": WINDOW,
        "hidden_size": HIDDEN,
        "pad_token_id": 0,
        "cls_token_id": 1,
        "sep_token_id": 2,
    }


@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8"])
def test_other_precisions_are_refused(tiny, monkeypatch, precision) -> None:
    """Only fp32 and fp16 builds are measured; anything else is refused before building."""
    seen = _capture(monkeypatch)
    with pytest.raises(NotImplementedError, match="fp32 or fp16"):
        model_module.build(_request(tiny, precision), _Writer())
    assert seen == []


def test_the_default_precision_is_fp32() -> None:
    from tensorrt_model_connect.model_support import ModelMetadata

    from ..support import describe

    support = describe(ModelMetadata(config={"model_type": "deberta-v2"}, model_index={}))
    assert support.default_precision == "fp32"


@pytest.mark.gpu
@pytest.mark.trt
def test_an_fp16_engine_keeps_the_io_and_the_values(tiny) -> None:
    """fp16 changes only the work type: int32 in, float32 features out, close to fp32."""
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("engine build needs a GPU")
    weights = model_module._open_checkpoint(tiny)
    cfg = config_module.resolve(tiny, weights, model_module._prefix(weights))
    length = 27  # past position_buckets / 2, so the log bucketing is in play
    ids = torch.zeros(WINDOW, dtype=torch.int32, device="cuda")
    ids[:length] = torch.arange(3, 3 + length, dtype=torch.int32) % VOCAB
    mask = torch.zeros(WINDOW, dtype=torch.int32, device="cuda")
    mask[:length] = 1

    def run(precision):
        plan = model_module.build_engine(weights, cfg, seq_len=WINDOW, precision=precision)
        engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
        io = {
            engine.get_tensor_name(i): (
                engine.get_tensor_dtype(engine.get_tensor_name(i)),
                tuple(engine.get_tensor_shape(engine.get_tensor_name(i))),
            )
            for i in range(engine.num_io_tensors)
        }
        hidden = torch.empty((WINDOW, HIDDEN), dtype=torch.float32, device="cuda")
        context = engine.create_execution_context()
        for name, tensor in (
            ("input_ids", ids),
            ("attention_mask", mask),
            ("last_hidden_state", hidden),
        ):
            context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return io, hidden.cpu()

    io32, hidden32 = run("fp32")
    io16, hidden16 = run("fp16")
    assert (
        io16
        == io32
        == {
            "input_ids": (trt.int32, (WINDOW,)),
            "attention_mask": (trt.int32, (WINDOW,)),
            "last_hidden_state": (trt.float32, (WINDOW, HIDDEN)),
        }
    )
    assert torch.isfinite(hidden16).all()
    real16, real32 = hidden16[:length], hidden32[:length]
    error = float((real16 - real32).norm() / real32.norm())
    assert error < 1e-2
