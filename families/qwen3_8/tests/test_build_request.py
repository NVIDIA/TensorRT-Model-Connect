# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""prefill_chunk_len build input: validation and compatibility with the core request."""

from __future__ import annotations

import pytest

from families.qwen3_8.build_request import BuildRequest, coerce_request


def _request(**overrides) -> BuildRequest:
    values = dict(
        model_dir="m", output_path="o", family="qwen3_8", task="text_generation",
        precision="fp16")
    values.update(overrides)
    return BuildRequest(**values)


@pytest.mark.parametrize("value", [None, 0, 2, 8, 32, 64, 128, 192])
def test_prefill_chunk_len_accepts_supported_values(value) -> None:
    from dataclasses import replace
    from families.qwen3_8 import model
    from families.qwen3_8.edge_llm import dispatch

    request = _request(prefill_chunk_len=value)
    assert request.prefill_chunk_len == value
    assert model._native_prefill_chunk_len(request, 256) == (
        min(model._DEFAULT_PREFILL_SEQ_LEN, 256) if value is None else value)
    assert model._native_prefill_chunk_len(replace(request, prefill_chunk_len=128), 100) == 64
    for mode in (dict(visual=True), dict(external_draft=True), dict(fp8_cache=True),
                 dict(prepared_mtp=True)):
        if value in (None, 0):
            assert model._native_prefill_chunk_len(request, 256, **mode) == 0
        else:
            with pytest.raises(NotImplementedError, match="prefill_chunk_len"):
                model._native_prefill_chunk_len(request, 256, **mode)
    if value is not None:
        # No checkpoint exists: explicit native prefill must bypass Edge probing.
        seen = []
        dispatch.build(request, None, lambda req, writer: seen.append((req, writer)))
        assert seen == [(request, None)]


@pytest.mark.parametrize("value", [1, 3, 48, 65, 100, -64])
def test_prefill_chunk_len_rejects_unsupported_values(value) -> None:
    with pytest.raises(ValueError, match="prefill_chunk_len"):
        _request(prefill_chunk_len=value)


def test_coerce_request_defaults_family_only_fields_for_core_requests() -> None:
    """The shared core request has no prefill_chunk_len; coercion must not require it."""

    class CoreRequest:
        model_dir = "m"
        output_path = "o"
        family = "qwen3_8"
        task = "text_generation"
        precision = "fp16"
        backend = "trt"
        max_sequence_length = None
        tensor_parallel_size = 1
        quantization = None
        fp32_layers = ()
        mtp_seq_len = None
        verbose = False
        graph_transform = None

    assert coerce_request(CoreRequest()).prefill_chunk_len is None
