# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference backend for deterministic artifacts exported by upstream ELF JAX."""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class ElfUpstreamReplayReference:
    @property
    def backend_name(self) -> str:
        return "upstream_replay"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        del ctx
        artifact_value = (
            case.inputs.get("upstream_replay_artifact")
            or case.inputs.get("elf_replay_artifact")
        )
        if not artifact_value:
            raise RuntimeError(f"ELF case {case.name} has no upstream replay artifact")
        artifact_path = Path(str(artifact_value))
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        samples = artifact.get("expected_generated_samples", [])
        if not samples:
            files = artifact.get("files", {})
            expected_value = (
                files.get("expected_generated_jsonl_path")
                or files.get("expected_jsonl_path")
                if isinstance(files, dict)
                else None
            )
            if isinstance(expected_value, str) and expected_value:
                expected_path = Path(expected_value)
                if not expected_path.is_absolute():
                    expected_path = artifact_path.parent / expected_path
                samples = [
                    json.loads(line)
                    for line in expected_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        if not isinstance(samples, list) or not samples:
            raise RuntimeError(
                f"ELF replay artifact {artifact_path} has no expected samples")
        text = "\n".join(str(sample.get("generated", "")) for sample in samples)
        return StageOutput(
            stage_name=stage.name,
            data={"expected_generated_samples": samples},
            text=text,
            metadata={
                "source": "official_elf_jax_replay",
                "artifact": str(artifact_path),
            },
        )


plugin = ElfUpstreamReplayReference()
