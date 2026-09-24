# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise request rejection and bundle emission without a TensorRT installation."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
import pytest
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.qwen.embedding import validate_request, build_embedding
from families.qwen.tests.test_embedding_contract import _write_pooling_config, _qwen3_config


def request_at(root):
    _write_pooling_config(root)
    config = _qwen3_config(root)
    (root / "config.json").write_text(json.dumps(config.raw))
    (root / "tokenizer.json").write_text("{}")
    return BuildRequest(
        model_dir=root,
        output_path=root / "test.bundle",
        family="qwen",
        task="embedding",
        precision="bf16",
        max_sequence_length=256,
    )


@pytest.mark.parametrize(
    "changes,pattern",
    [
        ({"tensor_parallel_size": 2}, "parallel"),
        ({"context_parallel_size": 2}, "parallel"),
        ({"max_batch_size": 2}, "one text"),
        ({"dynamic_kv_cache": True}, "KV cache"),
        ({"quantization": "fp8"}, "quantization"),
        ({"fp32_layers": (0,)}, "mixed precision"),
        ({"image_height": 32}, "text only"),
        ({"precision": "int8"}, "precision"),
        ({"precision": "fp32"}, "precision"),
        ({"task": "text_generation"}, "task"),
        ({"max_sequence_length": 32769}, "context"),
    ],
)
def test_rejects_unsupported_build_before_weights(tmp_path, changes, pattern):
    request = replace(request_at(tmp_path), **changes)
    with pytest.raises(ValueError, match=pattern):
        validate_request(request)


def test_generation_checkpoint_is_not_embedding(tmp_path):
    request = request_at(tmp_path)
    (tmp_path / "modules.json").unlink()
    with pytest.raises(ValueError, match="sentence-transformers"):
        validate_request(request)


def test_bundle_contains_task_contract_and_no_generation_head(tmp_path, monkeypatch):
    request = request_at(tmp_path)
    calls = []

    def weights(path, config, **kwargs):
        calls.append(kwargs)
        return {"embedding": "fixture"}

    def engine(config, values, length, **kwargs):
        assert values == {"embedding": "fixture"}
        assert length == 256 and kwargs["precision"] == "bf16"
        return b"fixture-plan-not-gpu-evidence"

    monkeypatch.setitem(
        sys.modules,
        "families.qwen.embedding_weights",
        SimpleNamespace(load_standard_weights=weights),
    )
    monkeypatch.setitem(
        sys.modules,
        "families.qwen.embedding_builder",
        SimpleNamespace(build_qwen3_embedding_engine=engine),
    )
    writer = BundleWriter(request.output_path)
    build_embedding(request, writer)
    writer.finish()
    data = request.output_path.read_bytes()
    size = int.from_bytes(data[8:16], "little")
    header = json.loads(data[16 : 16 + size])
    assert (header["family"], header["task"], header["backend"]) == ("qwen", "embedding", "trt")
    assert set(header["sections"]) == {"engine.plan", "runtime.json", "tokenizer.json"}
    section = header["sections"]["runtime.json"]
    start = 16 + size + section["offset"]
    runtime = json.loads(data[start : start + section["length"]])
    assert runtime["embedding_dimension"] == 1024
    assert runtime["embedding_pooling"] == "last_token"
    assert runtime["embedding_normalize"] is True
    assert runtime["embedding_eos_token_id"] == 151643
    assert calls == [{"precision": "bf16", "include_lm_head": False}]
