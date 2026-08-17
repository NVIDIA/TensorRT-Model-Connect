# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the Qwen generation runtime."""

from pathlib import Path


RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "runtime" / "models" / "qwen" / "pipeline.cpp"
)


def test_final_requested_token_does_not_trigger_an_unused_decoder_step() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    decode_loop = source.split("int32_t QwenTextGenerationPipeline::run_decode_loop(", maxsplit=1)[
        1
    ].split("int32_t QwenTextGenerationPipeline::select_decoder_index", maxsplit=1)[0]

    final_token_guard = "if (step + 1 >= max_new_tokens)"
    assert final_token_guard in decode_loop
    assert decode_loop.index(final_token_guard) < decode_loop.index(
        "run_step_device(result.token_id)"
    )


def test_one_token_generation_does_not_prime_an_unused_decoder_graph() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    generate = source.split(
        "QwenTextGenerationPipeline::TimedGenResult QwenTextGenerationPipeline::generate_from_ids(",
        maxsplit=1,
    )[1].split(
        "QwenTextGenerationPipeline::TimedGenResult "
        "QwenTextGenerationPipeline::generate_diffusion_from_ids(",
        maxsplit=1,
    )[0]

    assert "run_prefill(input_ids, logits, gpu_sampling, max_new_tokens > 1)" in generate
