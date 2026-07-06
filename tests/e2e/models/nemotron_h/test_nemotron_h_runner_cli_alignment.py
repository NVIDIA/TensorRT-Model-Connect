# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H E2E runner to CLI alignment tests."""

from __future__ import annotations

import json
import subprocess

from tests.e2e.models.nemotron_h.e2e_plugins.runners import text_generation
from tests.e2e_harness.contracts import E2ECase, RunContext


def test_text_runner_maps_runtime_config_to_set_flags(monkeypatch, tmp_path):
    case = E2ECase(
        name="nemotron-h-case",
        hf_id="dummy/nemotron-h",
        family="nemotron_h",
        runtime_strategy="nemotron_h_hybrid_mamba_attention",
        task_strategy="text_generation_causal",
        bundle="nemotron-h-case.trtfb",
        inputs={},
        metadata={
            "runtime_config": {
                "runtime": {"disable_cuda_graph": True},
            },
        },
    )
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("-o") + 1]
        with open(output_path, "w", encoding="utf-8") as output:
            output.write(json.dumps({"generated": "Paris", "token_ids": [1]}))
            output.write("\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(text_generation.subprocess, "run", _fake_run)

    text_generation.TextGenerationCausalRunner()._run_cpp_binary(
        ctx,
        str(tmp_path / case.bundle),
        "The capital of France is",
        4,
        case=case,
    )

    cmd = captured["cmd"]
    assert "--set" in cmd
    assert "runtime.disable_cuda_graph=true" in cmd
