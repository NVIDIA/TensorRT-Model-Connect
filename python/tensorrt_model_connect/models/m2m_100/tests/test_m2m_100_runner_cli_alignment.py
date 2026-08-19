# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M2M-100 E2E runner to CLI alignment tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tensorrt_model_connect.models.m2m_100.tests.e2e_plugins.runners import text_generation
from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.manifest_loader import load_manifest


def test_nllb_manifest_aligns_precision_and_language_contract() -> None:
    case = load_manifest(Path(__file__).with_name("manifests") / "nllb-200.json")

    assert case.metadata["precision"] == "fp16"
    assert case.metadata["reference_precision"] == "fp16"
    assert case.metadata["contract_config"] == {
        "auto_class": "AutoModelForSeq2SeqLM",
        "source_language": "eng_Latn",
        "source_language_token_id": 256047,
        "target_language": "fra_Latn",
        "forced_bos_token_id": 256057,
    }


def test_nllb_runner_passes_language_token_contract(monkeypatch, tmp_path) -> None:
    case = E2ECase(
        name="nllb-case",
        hf_id="facebook/nllb-200-distilled-600M",
        family="m2m_100",
        runtime_strategy="m2m_100_seq2seq_encoder_decoder",
        task_strategy="text_generation_causal",
        bundle="nllb-case.bundle",
        inputs={},
        metadata={
            "contract_config": {
                "source_language_token_id": 256047,
                "forced_bos_token_id": 256057,
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
            output.write(json.dumps({"generated": "Bonjour", "token_ids": [256057, 1]}))
            output.write("\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(text_generation.subprocess, "run", _fake_run)

    text_generation.TextGenerationCausalRunner()._run_cpp_binary(
        ctx,
        str(tmp_path / case.bundle),
        "The house is wonderful.",
        20,
        case=case,
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--source-language-token-id") + 1] == "256047"
    assert cmd[cmd.index("--forced-bos-token-id") + 1] == "256057"
