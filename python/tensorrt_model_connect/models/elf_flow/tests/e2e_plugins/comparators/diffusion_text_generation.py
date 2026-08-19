# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comparator for diffusion text generation outputs.

Most ELF checks are handled by the contract plugin, but this lightweight
comparator keeps the task strategy covered when no external reference is used.
"""

from __future__ import annotations

from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


class DiffusionTextGenerationComparator:
    @property
    def task_strategy(self) -> str:
        return "diffusion_text_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        del ref, threshold
        if stage.name not in {"decoded_text", "end_to_end", "full_generation"}:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.PASSED.value,
                metrics={"stage_ok": MetricResult(1.0, 1.0, "==", True)},
                message=f"{stage.name} invariant stage",
            )
        samples = trt.data.get("generated_samples", []) if trt.data else []
        non_empty = 0
        if isinstance(samples, list):
            for sample in samples:
                if isinstance(sample, dict) and str(sample.get("generated", "")).strip():
                    non_empty += 1
        passed = non_empty > 0
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics={
                "non_empty_generated_text": MetricResult(
                    float(non_empty), 1.0, ">=", passed
                )
            },
            message="diffusion text generation produced text" if passed else "no generated text",
        )


plugin = DiffusionTextGenerationComparator()
