# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.modernbert.config import ModelConfig
from families.modernbert.tests.test_local_attention import _weights
from families.modernbert.tests.test_local_attention import graph as graph
from tensorrt_model_connect.build import BuildRequest


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("sequence_length", [1, 3, 8])
def test_serial_graph_preserves_requested_precision(graph, precision, sequence_length):
    network, encoder, _, _ = graph(sequence_length)
    config = ModelConfig.from_json(
        json.dumps(
            {
                "model_type": "modernbert",
                "hidden_size": 4,
                "vocab_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 4,
            }
        )
    )
    values = [2**20, -(2**20), 2**-30, -(2**-30)] if precision != "fp16" else [1, -1, 0.5, -0.5]
    assert (
        encoder._ModernBertModel().build_engine(config, _weights(values), 8, precision=precision)
        == b"typed-graph"
    )
    table = network.gathers[0]
    assert table.dtype == precision
    expected = np.asarray(
        values,
        dtype={"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}[precision],
    ).astype(np.float32)
    np.testing.assert_array_equal(table.values, np.tile(expected, (4, 1)))
    assert len(network.ropes) == 4
    assert all(tensor.dtype == precision for rope in network.ropes for tensor in rope)
    assert network.output.dtype == "fp32"
    assert network.output.shape == (sequence_length, 4)
    assert np.isfinite(network.output.values).all()
    if precision == "bf16":
        assert not any(values.dtype == np.float16 for values in network.constants)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_tp_only_dispatches_its_implemented_fp32_precision(graph, monkeypatch, precision):
    _, encoder, _, _ = graph(1)
    parallel = SimpleNamespace(enabled=True, validate=lambda: None)
    encoder._ModernBertModel.build_engine.__globals__["normalize_parallel_config"] = lambda config: (
        parallel
    )
    monkeypatch.setitem(
        sys.modules,
        "families.modernbert.parallel",
        SimpleNamespace(build_tp_modernbert_engine=lambda *args, **kwargs: b"fp32-only"),
    )
    if precision == "fp32":
        assert (
            encoder._ModernBertModel().build_engine(
                None, {}, 8, precision=precision, parallel_config=parallel
            )
            == b"fp32-only"
        )
    else:
        with pytest.raises(NotImplementedError, match="require fp32"):
            encoder._ModernBertModel().build_engine(
                None, {}, 8, precision=precision, parallel_config=parallel
            )


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_public_tp_request_fails_before_loading_weights(graph, tmp_path, precision):
    _, encoder, _, _ = graph(1)
    namespace = encoder.build.__globals__
    namespace["_ModernBertModel"] = lambda: pytest.fail(
        "weights loaded for unsupported TP precision"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "modernbert"}), encoding="utf-8"
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="modernbert",
        task="text_to_pooled_features",
        precision=precision,
        tensor_parallel_size=2,
    )
    with pytest.raises(NotImplementedError, match="require fp32"):
        encoder.build(request, None)


def test_public_serial_bf16_request_reaches_typed_graph(graph, tmp_path, monkeypatch):
    network, encoder, _, _ = graph(3)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "modernbert",
                "hidden_size": 4,
                "vocab_size": 4,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "intermediate_size": 4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        encoder._ModernBertModel, "load_weights", lambda *args: _weights([1, -1, 0.5, -0.5])
    )
    namespace = encoder.build.__globals__
    namespace["_tokenizer_runtime_contract"] = lambda path: {}
    namespace["_BUNDLE_FILES"] = ()
    engines = {}
    writer = SimpleNamespace(
        set_header=lambda **kwargs: None,
        add_json=lambda *args: None,
        add_bytes=lambda name, data: engines.update({name: data}),
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "model.bundle",
        family="modernbert",
        task="text_to_pooled_features",
        precision="bf16",
        max_sequence_length=8,
    )
    encoder.build(request, writer)
    assert engines["engine.plan"] == b"typed-graph"
    assert network.gathers[0].dtype == "bf16"
