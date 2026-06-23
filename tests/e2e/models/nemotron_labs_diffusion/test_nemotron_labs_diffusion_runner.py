from __future__ import annotations

import subprocess
from pathlib import Path

from tests.e2e.models.nemotron_labs_diffusion.e2e_plugins import (
    runner as nemotron_runner,
)
from tests.e2e.models.nemotron_labs_diffusion.e2e_plugins.runners import (
    text_generation as nemotron_text_generation,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _case() -> E2ECase:
    return E2ECase(
        name="case-a",
        hf_id="nvidia/Nemotron-Labs-Diffusion-8B",
        family="nemotron_labs_diffusion",
        runtime_strategy="nemotron_labs_diffusion",
        task_strategy="text_generation_causal",
        bundle="case-a.trtfb",
        ci_lane="acceptance",
        reference_family="nemotron_labs_diffusion_model_card",
        user_contract="model_card_generation_parity",
        inputs={
            "prompt": "hello",
            "max_new_tokens": 32,
            "generation_mode": "diffusion",
            "block_length": 32,
            "threshold": 0.9,
        },
        metadata={"contract_config": {}},
    )


def _ctx(case: E2ECase, tmp_path: Path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def test_runner_maps_diffusion_generation_cli_flags(monkeypatch, tmp_path) -> None:
    case = _case()
    ctx = _ctx(case, tmp_path)
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            '{"id":0,"generated":"ok","token_ids":[1,2,3]}\n',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(nemotron_text_generation.subprocess, "run", _fake_run)

    out = nemotron_runner.NemotronLabsDiffusionTextGenerationCausalRunner().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    cmd = captured["cmd"]
    assert "--generation-mode" in cmd
    assert "diffusion" in cmd
    assert "--block-length" in cmd
    assert "--threshold" in cmd
    assert "-o" in cmd
    assert out.data["token_ids"] == [1, 2, 3]
    assert out.metadata["cpp"]["command"] == cmd
