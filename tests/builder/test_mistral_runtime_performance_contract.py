# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the Mistral generation runtime."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE = REPOSITORY_ROOT / "src" / "runtime" / "models" / "mistral" / "pipeline.cpp"


def _function_body(source: str, start: str, end: str) -> str:
    return source.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def test_gpu_greedy_generation_uses_batched_prefill_device_logits() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    prefill = _function_body(
        source,
        "bool MistralTextGenerationPipeline::run_prefill_batched(",
        "void MistralTextGenerationPipeline::prime_decoder_after_batched_prefill(",
    )
    dispatch = _function_body(
        source,
        "void MistralTextGenerationPipeline::run_prefill(",
        "TrtModule& MistralTextGenerationPipeline::require_block_prefill(",
    )

    assert "bool retain_device_logits" in prefill
    assert "const auto logits_offset" in prefill
    assert "d_logits_ptr_ = device_logits + logits_offset" in prefill
    assert "run_prefill_batched(input_ids, logits, gpu_sampling)" in dispatch
    assert "!gpu_sampling && run_prefill_batched" not in dispatch


def test_decoder_graph_is_only_primed_before_its_first_capture() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    prime = _function_body(
        source,
        "void MistralTextGenerationPipeline::prime_decoder_after_batched_prefill(",
        "void MistralTextGenerationPipeline::run_prefill(",
    )

    assert "decoder.cuda_graph_active()" in prime
    assert "decoder.cuda_graph_captured()" in prime
    assert prime.index("decoder.cuda_graph_captured()") < prime.index(
        "decoder.forward_async(inputs)"
    )
