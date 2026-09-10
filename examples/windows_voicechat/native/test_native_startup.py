# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the built native executable without allocating a model on the GPU.

Set TRTMC_VOICECHAT_BRIDGE to the staged executable before running pytest.
"""

import json
import os
from pathlib import Path
import subprocess

import pytest

from tensorrt_model_connect.bundle_writer import BundleWriter


@pytest.fixture
def bridge() -> Path:
    configured = os.environ.get("TRTMC_VOICECHAT_BRIDGE")
    if not configured:
        pytest.skip("Set TRTMC_VOICECHAT_BRIDGE to run native startup checks")
    executable = Path(configured)
    assert executable.is_file(), f"Native bridge does not exist: {executable}"
    return executable


def test_help_needs_no_bundle_and_preserves_stdout(bridge: Path) -> None:
    result = subprocess.run([str(bridge), "--help"], capture_output=True, text=True,
                            encoding="utf-8", timeout=30, check=False)
    assert result.returncode == 0
    assert "--bundle" in result.stdout
    assert "TensorRT-RTX" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("family", "backend", "message"),
    [("nemotron_voicechat", "trt", "TensorRT-RTX"),
     ("other_family", "trt_rtx", "nemotron_voicechat")],
)
def test_wrong_backend_or_family_fails_before_engine_load(
    bridge: Path, tmp_path: Path, family: str, backend: str, message: str,
) -> None:
    # A Unicode path also exercises wmain and the native bundle reader's UTF-8
    # boundary. The payload is intentionally not an engine: metadata must reject it.
    bundle = tmp_path / "voice-é-语音.bundle"
    writer = BundleWriter(bundle)
    writer.set_header(family=family, task="speech_session", backend=backend)
    writer.add_bytes("engine.plan", b"not-an-engine")
    writer.finish()
    result = subprocess.run(
        [str(bridge), "--bundle", str(bundle), "--runtime-root", str(bridge.parent)],
        input="", capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert result.returncode != 0
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["fatal"] is True
    assert message in events[0]["message"]
    assert "loading" not in {event["type"] for event in events}


def test_malformed_bundle_returns_json_error(bridge: Path, tmp_path: Path) -> None:
    bundle = tmp_path / "truncated.bundle"
    bundle.write_bytes(b"BUNDLE\x01\x00")
    result = subprocess.run(
        [str(bridge), "--bundle", str(bundle), "--runtime-root", str(bridge.parent)],
        input="", capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert result.returncode != 0
    error = json.loads(result.stdout)
    assert error["type"] == "error" and error["fatal"] is True
