# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the Nemotron generation runtime."""

from pathlib import Path


RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "runtime" / "models" / "nemotron" / "pipeline.cpp"
)


def _function_body(source: str, start: str, end: str) -> str:
    return source.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def test_gpu_greedy_generation_uses_batched_prefill_device_logits() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    prefill = _function_body(
        source,
        "bool NemotronTextGenerationPipeline::run_prefill_batched(",
        "void NemotronTextGenerationPipeline::run_prefill(",
    )
    dispatch = _function_body(
        source,
        "void NemotronTextGenerationPipeline::run_prefill(",
        "TrtModule& NemotronTextGenerationPipeline::require_block_prefill(",
    )

    assert "bool retain_device_logits" in prefill
    assert "if (retain_device_logits)" in prefill
    assert "TensorMap outputs = prefill_->forward(inputs)" in prefill
    assert "logits_tensor.numel()) - vocab" in prefill
    assert "d_logits_ptr_ = device_logits + logits_offset" in prefill
    assert "run_prefill_batched(input_ids, logits, gpu_sampling)" in dispatch
    assert "!gpu_sampling && run_prefill_batched" not in dispatch


def test_batched_prefill_does_not_launch_a_discarded_decoder_prime() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")

    assert "prime_decoder_after_batched_prefill" not in source


def test_generation_does_not_run_decoder_after_final_sample() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    decode = _function_body(
        source,
        "int32_t NemotronTextGenerationPipeline::run_decode_loop(",
        "int32_t NemotronTextGenerationPipeline::select_decoder_index(",
    )

    final_sample_guard = "if (step + 1 >= max_new_tokens)"
    assert final_sample_guard in decode
    assert decode.index(final_sample_guard) < decode.index("run_step_device(result.token_id)")
