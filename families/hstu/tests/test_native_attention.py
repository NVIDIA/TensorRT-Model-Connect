# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission and checkpoint-row semantics for the original attention adapter."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from families.hstu.config import parse_config
from families.hstu.native_attention_build import compile_target, unsupported_reason
from families.hstu.dense_graph import DenseGraph
from families.hstu.paged_graph import PagedGraph
from families.hstu.tests.fixtures import tiny_config


def supported():
    config = tiny_config()
    config["embedding_tables"] = [table for table in config["embedding_tables"] if table["role"] != "context"]
    return parse_config({**config, "num_heads": 4, "head_dim": 64,
                         "max_sequence_length": 1024, "scaling_seqlen": 1024,
                         "enable_history_cache": True, "time_buckets": 0})


@pytest.mark.parametrize("changes,precision", [
    ({}, "fp32"), ({}, "fp16"),
    ({"target_group_size": 2}, "bf16"),
    ({"is_causal": False}, "bf16"),
    ({"scaling_seqlen": -1}, "bf16"),
    ({"max_sequence_length": 2048}, "bf16"),
    ({"num_heads": 8}, "bf16"),
    ({"head_dim": 128}, "bf16"),
    ({"mode": "retrieval"}, "bf16"),
    ({"time_buckets": 2048}, "bf16"),
])
def test_unsupported_semantics_keep_ordinary_path(changes, precision):
    assert unsupported_reason({**supported(), **changes}, precision)


def test_context_is_not_silently_treated_as_history():
    config = supported()
    config["embedding_tables"].append({"name": "context", "role": "context", "num_embeddings": 3})
    assert unsupported_reason(config, "bf16")


def test_supported_semantics_have_no_batch_or_candidate_special_case():
    config = supported()
    assert unsupported_reason(config, "bf16") is None
    for batch in (1, 2, 4, 8):
        assert unsupported_reason({**config, "max_batch_size": batch}, "bf16") is None


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (9, 0), (10, 0), (10, 3), (12, 0)])
def test_compiler_target_changes_only_artifact_compatibility(capability):
    expected = "sm_" + "".join(str(value) for value in capability)
    assert compile_target(capability, [expected]) == expected
    with pytest.raises(ValueError, match="compiler does not support"):
        compile_target(capability, [])


@pytest.mark.parametrize("capability", [(7, 5), (8, -1), (10, 13), (10.0, 3), (True, 0), (8,)])
def test_invalid_compiler_target_is_rejected(capability):
    with pytest.raises(ValueError):
        compile_target(capability, ["sm_75", "sm_80", "sm_103"])


def test_uncached_native_admission_preserves_shared_block_math():
    config = {**supported(), "enable_history_cache": False}
    assert unsupported_reason(config, "bf16") is None
    assert DenseGraph.block is PagedGraph.block
    assert DenseGraph.linear is PagedGraph.linear
    assert unsupported_reason({**config, "target_group_size": 2}, "bf16")


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_model_graph_choice_never_queries_the_gpu(monkeypatch, tmp_path, cache, batch):
    from families.hstu import dense_graph, model, native_attention_build, paged_graph

    # GPU target discovery belongs solely to compiling the native artifact.
    # Block the GPU module at this boundary, while substituting only build I/O.
    monkeypatch.setitem(sys.modules, "cuda.bindings", None)
    selected = []
    config = {**supported(), "enable_history_cache": cache}
    library = tmp_path / "native.so"
    library.write_bytes(b"native-library")
    monkeypatch.setattr(native_attention_build, "source_directory", lambda hint: tmp_path)

    def build_library(output, source, *, attention_mode, verbose):
        selected.append(attention_mode)
        (output / "attention_native.NOTICE").write_bytes(b"upstream license notices")
        return library, {"namespace": "model-local"}

    def engine(kind):
        def build(config, weights, max_batch, path, namespace, verbose):
            selected.append((kind, max_batch, namespace))
            return b"plan"
        return build

    monkeypatch.setattr(native_attention_build, "build_attention_library", build_library)
    monkeypatch.setattr(dense_graph, "build_dense_engine", engine("dense"))
    monkeypatch.setattr(paged_graph, "build_paged_engine", engine("paged"))
    result = model._native_attention(config, {}, "bf16", batch, False, tmp_path, "trt")
    mode = "paged" if cache else "dense"
    assert selected == [mode, (mode, batch, "model-local")]
    assert result == (
        b"plan", b"native-library", {"namespace": "model-local"}, b"upstream license notices"
    )


def test_dense_placeholder_is_one_element_and_reused_without_cache_operations():
    graph = object.__new__(DenseGraph)
    graph.dtype = "bf16"
    constants = []

    def constant(data, name, dtype):
        constants.append((data, name, dtype))
        return object()

    graph.constant = constant
    first = graph.updated_pages(None, 0, None)
    second = graph.updated_pages(None, 1, None)
    assert first is second and len(constants) == 1
    assert constants[0][0].shape == (1,) and not np.any(constants[0][0])
    assert constants[0][2] == "bf16"


def test_original_projection_fields_preserve_every_head_and_channel():
    config = supported()
    h, d, e = config["num_heads"], config["head_dim"], config["hidden_size"]
    weights = {}
    for layer in range(config["num_layers"]):
        weights[f"blocks.{layer}.uvqk.weight"] = np.arange(h * 4 * d * e, dtype=np.float32).reshape(h * 4 * d, e)
        weights[f"blocks.{layer}.uvqk.bias"] = np.arange(h * 4 * d, dtype=np.float32)
    original = copy.deepcopy(weights)
    types = SimpleNamespace(float32="fp32", float16="fp16", bfloat16="bf16")
    graph = PagedGraph(types, None, config, weights, "bf16", creator=None)
    for layer in range(config["num_layers"]):
        prefix = f"blocks.{layer}.uvqk"
        for output_field, canonical_field in enumerate((0, 2, 3, 1)):
            for head in range(h):
                for channel in range(d):
                    canonical = (head * 4 + canonical_field) * d + channel
                    physical = (output_field * h + head) * d + channel
                    np.testing.assert_array_equal(graph.weights[f"{prefix}.weight"][physical],
                                                  original[f"{prefix}.weight"][canonical])
                    assert graph.weights[f"{prefix}.bias"][physical] == original[f"{prefix}.bias"][canonical]
        np.testing.assert_array_equal(weights[f"{prefix}.weight"], original[f"{prefix}.weight"])


def test_unknown_attention_choice_is_rejected():
    with pytest.raises(ValueError, match="attention_implementation"):
        parse_config({**tiny_config(), "attention_implementation": "unspecified_kernel"})
