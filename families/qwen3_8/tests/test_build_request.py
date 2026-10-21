# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill build inputs (chunk length, dynamic engine): validation and core-request compatibility."""

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
    assert _request(prefill_chunk_len=value).prefill_chunk_len == value


@pytest.mark.parametrize("value", [1, 3, 48, 65, 100, -64])
def test_prefill_chunk_len_rejects_unsupported_values(value) -> None:
    with pytest.raises(ValueError, match="prefill_chunk_len"):
        _request(prefill_chunk_len=value)


def test_prefill_dynamic_defaults_off() -> None:
    request = _request()
    assert request.prefill_dynamic is False
    assert request.prefill_opt_len is None


@pytest.mark.parametrize("chunk", [None, 64, 1024])
@pytest.mark.parametrize("opt", [None, 1, 256, 1024])
def test_prefill_dynamic_accepts_opt_len_up_to_chunk(chunk, opt) -> None:
    if opt is not None and chunk is not None and opt > chunk:
        pytest.skip("opt above the chunk is rejected, see the next test")
    request = _request(prefill_dynamic=True, prefill_chunk_len=chunk, prefill_opt_len=opt)
    assert request.prefill_dynamic and request.prefill_opt_len == opt


def test_prefill_dynamic_rejects_bad_combinations() -> None:
    with pytest.raises(ValueError, match="prefill_opt_len needs prefill_dynamic"):
        _request(prefill_opt_len=256)
    with pytest.raises(ValueError, match="multi-token prefill"):
        _request(prefill_dynamic=True, prefill_chunk_len=0)
    with pytest.raises(ValueError, match="prefill_opt_len"):
        _request(prefill_dynamic=True, prefill_chunk_len=512, prefill_opt_len=513)
    with pytest.raises(ValueError, match="prefill_opt_len"):
        _request(prefill_dynamic=True, prefill_opt_len=0)


def test_coerce_request_defaults_family_only_fields_for_core_requests() -> None:
    """The shared core request has no prefill fields; coercion must not require them."""

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

    coerced = coerce_request(CoreRequest())
    assert coerced.prefill_chunk_len is None
    assert coerced.prefill_dynamic is False
    assert coerced.prefill_opt_len is None
