# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import hashlib
import json

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.video_profile import check_controls, prepare_inputs


def test_video_workload_preserves_full_geometry_and_public_default_sentinels(tmp_path):
    source, prepared = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    noise = base64.b64encode(b"\0" * 8).decode()
    source.write_text(json.dumps({"text": "a cat", "extra": {"initial_latents_b64": noise}}) + "\n")
    rows = prepare_inputs(source, prepared, {"width": 832, "height": 480, "num_frames": 17,
        "num_steps": 30, "seed": 42, "guidance_scale": -1, "cfg_scale": -1, "negative_prompt": ""})
    row, = rows
    assert "guidance_scale" not in row["extra"] and "cfg_scale" not in row["extra"]
    assert row["extra"]["num_frames"] == 17 and row["extra"]["num_inference_steps"] == 30
    record = {"generation_config": {"width": 832, "height": 480, "seed": 42, "num_steps": 30},
              "fps": 24, "output_frames": 17, "prompt_sha256": hashlib.sha256(b"a cat").hexdigest(),
              "negative_prompt_sha256": hashlib.sha256(b"").hexdigest(),
              "initial_latents_sha256": hashlib.sha256(b"\0" * 8).hexdigest()}
    payload = {"model": "native", "prompt": row["text"], **row["extra"]}
    check_controls(payload, record, rows)
    for changed in ({"num_frames": 5}, {"num_inference_steps": 2}, {"prompt": "different"},
                    {"initial_latents_b64": base64.b64encode(b"\1" * 8).decode()}):
        with pytest.raises(ValueError):
            check_controls({**payload, **changed}, record, rows)
    with pytest.raises(ValueError, match="native request"):
        check_controls(payload, {**record, "output_frames": 5}, rows)


def test_partial_clip_and_unsupported_duration_cannot_be_qualified(tmp_path):
    source = tmp_path / "source.jsonl"
    base = {"width": 1280, "height": 704, "num_frames": 121, "num_steps": 50, "seed": 42}
    for extra in ({"num_frames": 5}, {"size": "672x384"}, {"seconds": 1}):
        source.write_text(json.dumps({"text": "cats", "extra": extra}) + "\n")
        with pytest.raises(ConfigError):
            prepare_inputs(source, tmp_path / "prepared.jsonl", base)


def test_long_video_timeout_is_forwarded_once_to_published_client(monkeypatch, tmp_path):
    from trtmc_aiperf_qual import aiperf_runner
    from trtmc_aiperf_qual.config import Environment

    monkeypatch.setattr(aiperf_runner, "_run", lambda *args, **kwargs: 0)
    monkeypatch.setattr(aiperf_runner, "_wait_ready", lambda *args: None)
    run = aiperf_runner.run_aiperf(Environment({"hf_datasets_cache": str(tmp_path)}),
        tmp_path / "client", ["--endpoint-type", "video_generation"],
        command_prefix=["python", "-m", "aiperf"], request_timeout_s=14400)
    assert run.command.count("--request-timeout-seconds") == 1
    assert run.command[run.command.index("--request-timeout-seconds") + 1] == "14400"
