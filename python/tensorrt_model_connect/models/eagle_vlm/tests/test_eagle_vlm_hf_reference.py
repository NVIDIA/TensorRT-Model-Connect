# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eagle-VLM Hugging Face reference tests."""

from __future__ import annotations

import sys

from tensorrt_model_connect.models.eagle_vlm.tests.e2e_plugins.references import hf_transformers
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec, StageOutput


def test_reranking_reference_uses_requested_device(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(**kwargs):
        captured.update(kwargs)
        return StageOutput(stage_name="full_inference", data={"scores": [1.0]})

    monkeypatch.setattr(hf_transformers, "run_reference_subprocess", fake_subprocess)
    monkeypatch.setattr(
        hf_transformers,
        "_resolve_cached_model_ref",
        lambda model: model,
    )
    case = E2ECase(
        name="rerank-reference",
        hf_id="org/reranker",
        family="eagle_vlm",
        runtime_strategy="eagle_vlm_reranking",
        task_strategy="reranking",
        bundle="rerank.bundle",
        inputs={"prompt": "query", "documents": ["document"]},
        metadata={"reference_device": "cpu"},
    )
    context = RunContext(case=case, artifacts_dir=str(tmp_path))

    hf_transformers.HfTransformersReference().run_stage(
        case,
        StageSpec("full_inference"),
        context,
    )

    command = captured["command"]
    assert command[:2] == [context.reference_python_path() or sys.executable, "-c"]
    script = command[2]
    assert "device = 'cpu'" in script
    assert "torch.cuda.is_available()" not in script
