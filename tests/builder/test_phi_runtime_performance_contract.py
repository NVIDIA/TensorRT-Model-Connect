# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the Phi generation runtime."""

from pathlib import Path


RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "runtime" / "models" / "phi" / "pipeline.cpp"
)


def test_decoder_graph_is_only_primed_before_its_first_capture() -> None:
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")
    prime = source.split(
        "void PhiTextGenerationPipeline::prime_decoder_after_batched_prefill(", maxsplit=1
    )[1].split(
        "void PhiTextGenerationPipeline::run_prefill(", maxsplit=1
    )[0]

    assert "decoder.cuda_graph_active()" in prime
    assert "decoder.cuda_graph_captured()" in prime
    assert prime.index("decoder.cuda_graph_captured()") < prime.index("decoder.forward_async(inputs)")
