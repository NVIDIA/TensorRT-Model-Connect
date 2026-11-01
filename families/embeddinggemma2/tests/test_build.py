# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata
from families.embeddinggemma2 import model, support
from families.embeddinggemma2.tests.test_config import tiny_config


@pytest.mark.parametrize(
    "option,value",
    [
        ("task", "encoding"),
        ("precision", "fp16"),
        ("max_batch_size", 2),
        ("tensor_parallel_size", 2),
        ("context_parallel_size", 2),
        ("image_height", 224),
        ("video_num_frames", 1),
        ("quantization", "int8"),
        ("dynamic_kv_cache", True),
        ("mtp_seq_len", 4),
        ("fp32_layers", (1,)),
    ],
)
def test_unsupported_options_fail_before_checkpoint_or_gpu(tmp_path, option, value):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="embeddinggemma2",
        task="embedding",
        precision="bf16",
    )
    request = replace(request, **{option: value})
    with pytest.raises(
        NotImplementedError if option in {"dynamic_kv_cache", "mtp_seq_len"} else ValueError
    ):
        model.build(request, None)


def test_bundle_contract_and_capacity_are_family_owned(tmp_path, monkeypatch):
    raw = tiny_config()
    raw["image_token_id"] = 14
    (tmp_path / "config.json").write_text(json.dumps(raw))
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    monkeypatch.setattr(model, "load_weights", lambda *args: {})
    calls = []
    monkeypatch.setattr(model, "build_engine", lambda *args: calls.append(args) or b"plan")
    sections, headers = {}, []
    writer = SimpleNamespace(
        set_header=lambda **kw: headers.append(kw),
        add_bytes=sections.__setitem__,
        add_json=sections.__setitem__,
    )
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "out.bundle",
        family="embeddinggemma2",
        task="embedding",
        precision="bf16",
        max_sequence_length=16,
    )
    model.build(request, writer)
    assert calls[0][2:4] == (16, "bf16")
    assert headers == [{"family": "embeddinggemma2", "task": "embedding", "backend": "trt"}]
    assert sections["engine.plan"] == b"plan"
    assert sections["runtime.json"]["pooling"] == "mean"
    assert sections["runtime.json"]["embedding_dimension"] == 6
    assert sections["runtime.json"]["unsupported_token_ids"] == [14]
    for invalid in (0, 1, 33, True):
        with pytest.raises(ValueError):
            request = replace(request, max_sequence_length=invalid)
            model.build(request, writer)


def test_support_does_not_claim_gemma_or_multimodal_tasks():
    assert support.describe(ModelMetadata({"model_type": "gemma4"}, {})) is None
    actual = support.describe(ModelMetadata({"model_type": "embedding_gemma2"}, {}))
    assert actual.tasks == ("embedding",)
    assert actual.default_precision == "bf16"
