# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Labs Diffusion-owned Hugging Face reference tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.nemotron_labs_diffusion.e2e_plugins.references import (
    hf_transformers as nemotron_hf_transformers,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_mode_aliases() -> None:
    normalize = nemotron_hf_transformers._normalize_nemotron_labs_diffusion_mode
    assert normalize("auto") == "diffusion"
    assert normalize("dlm") == "diffusion"
    assert normalize("autoregressive") == "ar"
    assert normalize("linear-speculation-lora") == "linear_spec_lora"


def test_cached_snapshot_resolution_uses_config_anchor(
    monkeypatch, tmp_path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[dict] = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return str(snapshot)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        fake_snapshot_download,
    )

    resolved = nemotron_hf_transformers._resolve_cached_model_ref(
        "nvidia/Nemotron-Labs-Diffusion-8B"
    )

    assert resolved == str(snapshot)
    assert calls == [{
        "args": ("nvidia/Nemotron-Labs-Diffusion-8B",),
        "kwargs": {
            "allow_patterns": ["config.json"],
            "local_files_only": True,
        },
    }]


def test_reference_uses_custom_auto_model_path(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd, 0, stdout="OK mode=linear_spec_lora tokens=1\n", stderr=""
        )

    monkeypatch.setattr(
        nemotron_hf_transformers,
        "_resolve_cached_model_ref",
        lambda hf_id: "/cached/model",
    )
    monkeypatch.setattr(nemotron_hf_transformers.subprocess, "run", _fake_run)

    case = E2ECase(
        name="nemotron-labs-diffusion",
        hf_id="nvidia/Nemotron-Labs-Diffusion-8B",
        family="nemotron_labs_diffusion",
        runtime_strategy="nemotron_labs_diffusion",
        task_strategy="text_generation_causal",
        inputs={
            "prompt": "hello",
            "max_new_tokens": 64,
            "generation_mode": "linear_spec_adapter",
            "block_length": 16,
            "threshold": 0.75,
            "temperature": 0.0,
        },
        metadata={
            "precision": "bf16",
            "contract_config": {
                "use_chat_template": True,
                "enable_thinking": False,
            },
        },
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/ref/python",
    )

    out = nemotron_hf_transformers.HfTransformersReference().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    cmd = captured["cmd"]
    assert cmd[:2] == ["/ref/python", "-c"]
    script = cmd[2]
    assert "from transformers import AutoModel, AutoTokenizer" in script
    assert "AutoModelForCausalLM" not in script
    assert 'template_path = Path(model_ref) / "chat_template.jinja"' in script
    assert "fallback_chat_template" in script
    assert "<|im_start|>system\\\\n<|im_end|>" in script
    assert 'adapter_path = Path(model_ref) / "linear_spec_lora"' in script
    assert "str(adapter_path)" in script
    assert '_tf_generic.check_model_inputs = lambda fn: fn' in script
    assert "PeftModel.from_pretrained" in script
    assert "generation_model = model.model" in script
    assert "def _call_supported(fn, input_ids, **kwargs):" in script
    assert "return fn(input_ids, **filtered_kwargs)" in script
    assert "generation_model.linear_spec_generate" in script
    assert "ids_tensor," in script
    assert 'subfolder="linear_spec_lora"' in script
    assert "generation_mode = 'linear_spec_lora'" in script
    assert "block_length = 16" in script
    assert "threshold = 0.75" in script
    assert "enable_thinking = False" in script
    assert out.metadata["generation_mode"] == "linear_spec_lora"
    assert out.logits is None
