# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static performance contracts for the Phi generation runtime."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_SOURCE = REPOSITORY_ROOT / "src" / "runtime" / "models" / "phi" / "pipeline.cpp"
TRT_MODULE_SOURCE = REPOSITORY_ROOT / "src" / "runtime" / "backend" / "trt_module_impl.cpp"


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


def test_decoder_capture_state_delegates_to_cuda_graph_readiness() -> None:
    source = TRT_MODULE_SOURCE.read_text(encoding="utf-8")
    implementation = source.split(
        "bool TrtModuleImpl::cuda_graph_captured() const", maxsplit=1
    )[1].split(
        "bool TrtModuleImpl::begin_timing_event", maxsplit=1
    )[0]

    assert "use_cuda_graph_" in implementation
    assert "cuda_graph_" in implementation
    assert "cuda_graph_->ready()" in implementation
