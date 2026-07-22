# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Magpie's offline NeMo reference dependency resolution."""

from __future__ import annotations

import subprocess

from tests.e2e.models.magpie_tts.e2e_plugins.references.nemo_reference import (
    MAGPIE_SPEAKER_ENCODER_FILENAME,
    MAGPIE_SPEAKER_ENCODER_REPO,
    MAGPIE_SPEAKER_ENCODER_URL,
    NemoReference,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_magpie_reference_maps_upstream_url_to_pre_warmed_file(monkeypatch, tmp_path) -> None:
    case = E2ECase(
        name="magpie-case",
        hf_id="nvidia/magpie_tts_multilingual_357m",
        hf_revision="34d7e40da85cabc97f92198889b65cea27bc7fd1",
        family="magpie_tts",
        runtime_strategy="text_to_audio_magpie",
        task_strategy="text_to_audio",
        inputs={"prompt": "Hello"},
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/profiles/magpie/bin/python",
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                '{"num_samples": 1, "rms": 0.1, "duration_s": 0.1, '
                '"sample_rate": 22050, "wav_path": "reference.wav"}'
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "tests.e2e.models.magpie_tts.e2e_plugins.references.nemo_reference.subprocess.run",
        fake_run,
    )

    output = NemoReference().run_stage(case, StageSpec("full_generation"), ctx)

    command = captured["cmd"]
    assert command[0] == "/profiles/magpie/bin/python"
    script = command[2]
    compile(script, "<magpie-reference>", "exec")
    cublas_config = script.index(
        'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")'
    )
    torch_import = script.index("import torch")
    deterministic_algorithms = script.index(
        "torch.use_deterministic_algorithms(True)"
    )
    assert f"repo_id={MAGPIE_SPEAKER_ENCODER_REPO!r}" in script
    assert f"filename={MAGPIE_SPEAKER_ENCODER_FILENAME!r}" in script
    assert "local_files_only=True" in script
    assert f"speaker_checkpoint_url = {MAGPIE_SPEAKER_ENCODER_URL!r}" in script
    assert "path = speaker_checkpoint" in script
    assert "revision='34d7e40da85cabc97f92198889b65cea27bc7fd1'" in script
    assert 'filename="magpie_tts_multilingual_357m.nemo"' in script
    model_load = script.index("model = MagpieTTSModel.restore_from")
    device_transfer = script.index("model = model.to(device)")
    generation_seed = script.rindex("torch.manual_seed(42)")
    generation = script.index("audio_tensor, audio_len = model.do_tts")
    script_lines = [line.strip() for line in script.splitlines()]
    assert cublas_config < torch_import < deterministic_algorithms < model_load
    assert model_load < device_transfer < generation_seed < generation
    assert script_lines.count("torch.manual_seed(42)") == 2
    assert script_lines.count("torch.cuda.manual_seed_all(42)") == 2
    assert script_lines.count("random.seed(42)") == 2
    assert script_lines.count("np.random.seed(42)") == 2
    assert output.data["returncode"] == 0
    assert output.data["rms"] == 0.1
