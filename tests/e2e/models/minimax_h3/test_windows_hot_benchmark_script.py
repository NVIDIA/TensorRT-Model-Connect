# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPOSITORY_ROOT / "scripts" / "run_windows_h3_hot_benchmark.ps1"
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build_windows_h3.ps1"
BENCHMARK_DOC = (
    REPOSITORY_ROOT / "website" / "docs" / "reference" / "minimax-h3-windows-hot-benchmark.md"
)


def test_windows_hot_benchmark_is_parameterized_and_revision_bound() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT.read_text(encoding="utf-8")

    for parameter in ("$Bundle", "$CudaRoot", "$TensorRtRtxRoot", "$OutputDirectory"):
        assert parameter in runner
    for fixed_setting in (
        'video_num_frames = 124',
        'num_inference_steps = 50',
        '"minimax_h3.retain_engines" = $true',
        '"minimax_h3.retained_tail_weight_budget_gib"',
        'cuda_graphs = $false',
    ):
        assert fixed_setting in runner

    assert "git -C $RepositoryRoot status --porcelain" in runner
    assert "$WorkerMetadata.build.source_revision -ne $SourceRevision" in runner
    assert '$WorkerMetadata.build.configuration -ne "Release"' in runner
    assert "Read-MiniMaxBundleConfig" in runner
    assert '[ \\t]*\\r?$"' in runner
    assert "validate_native_bundle_config" in runner
    assert "BundleProvenanceVerified" in runner
    assert "PromptMatchesBaseline" in runner
    assert "-not $SkipBundleHash" in runner
    assert "cudart_runtime_sha256" in runner
    assert "OutputDirectory must be outside the source checkout" in runner
    assert "generated_pixels -ne $ExpectedElements" in runner
    assert "Test-FinitePositive" in runner
    assert "C:\\Users\\" not in runner

    assert "[switch]$BuildBenchmarks" in build_script
    assert '"-DTRTMC_SOURCE_REVISION=$SourceRevision"' in build_script
    assert '$BuildTargets += "trtmc_benchmark_worker"' in build_script
    assert "requires a clean source checkout" in build_script
    assert "TRTMC_MINIMAX_H3_SOURCE_REVISION" in build_script


def test_windows_hot_benchmark_document_has_a_copyable_contract() -> None:
    documentation = BENCHMARK_DOC.read_text(encoding="utf-8")

    assert ".\\scripts\\build_windows_h3.ps1" in documentation
    assert ".\\scripts\\run_windows_h3_hot_benchmark.ps1" in documentation
    assert "one untimed pipeline warmup and two measured calls" in documentation
    assert "portable latency" in documentation
    assert "guarantee" in documentation
    assert "does not redistribute checkpoints" in documentation
