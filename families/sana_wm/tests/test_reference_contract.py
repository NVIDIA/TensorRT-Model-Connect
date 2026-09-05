# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from . import test_e2e as e2e


def test_official_source_dependencies_are_family_owned() -> None:
    requirements = {
        line.strip()
        for line in (e2e.TEST_ROOT.parent / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "flash-linear-attention>=0.4.2",
        "imageio[pyav]",
        "mmcv==1.7.2",
        "pyrallis",
        "pytz",
        "qwen-vl-utils",
        "termcolor",
    } <= requirements


def test_manifest_owns_the_exact_camera_control_workload() -> None:
    _, manifest, case = e2e.CASES["sana-wm-bidirectional"]
    assert manifest["video_num_frames"] == 321
    assert case["camera_intrinsics_file"] == "assets/demo_0_intrinsics.npy"
    assert case["action"] == "w-80,jw-40,w-40,lw-60,w-100"
    assert case["translation_speed"] == 0.055
    assert case["rotation_speed_deg"] == 1.2
    assert case["cfg_scale"] == 5.0
    assert case["fps"] == 16
    assert case["flow_shift"] == 9.8
    assert case["no_action_overlay"] is True
    assert case["seed"] == 42
    assert json.loads((e2e.TEST_ROOT / "reference-source.json").read_text()) == {
        "repository": "NVlabs/Sana",
        "revision": "59629fdf790850797cb657bad014fce432bd713d",
    }


def test_native_receives_camera_inputs_and_cfg(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["sana-wm-bidirectional"]
    captured = {}

    def run_json(*args):
        captured["arguments"] = args[6:]
        return {"output": "native-frames"}

    monkeypatch.setattr(e2e, "_run_json", run_json)
    e2e._native(
        Path("trtmc"),
        Path("runtime"),
        Path("bundle"),
        Path("model"),
        manifest,
        case,
        tmp_path,
    )
    arguments = captured["arguments"]
    assert arguments[arguments.index("--action") + 1] == case["action"]
    assert arguments[arguments.index("--cfg-scale") + 1] == "5.0"
    assert arguments[arguments.index("--seed") + 1] == "42"
    assert "--refiner-seed" not in arguments
    intrinsics = Path(arguments[arguments.index("--intrinsics") + 1])
    np.testing.assert_array_equal(
        np.fromfile(intrinsics, dtype=np.float32),
        np.asarray(case["camera_intrinsics"], dtype=np.float32),
    )


def test_raw_snapshot_calls_declared_official_entrypoint(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["sana-wm-bidirectional"]
    manifest = {**manifest, "video_num_frames": 3}
    source = tmp_path / "source"
    entrypoint = source / "inference_video_scripts/wm/inference_sana_wm.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# declared official source\n", encoding="utf-8")
    model_dir = tmp_path / "raw-checkpoint"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "vae").mkdir()
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    assert not (model_dir / "model_index.json").exists()
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(source))
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "reference_generated.mp4").write_bytes(b"video")
        return e2e.subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def decode(video_path: Path, frames_dir: Path):
        captured["video_path"] = video_path
        frames_dir.mkdir()
        paths = []
        for index in range(3):
            path = frames_dir / f"frame_{index:04d}.png"
            path.write_bytes(b"png")
            paths.append(path)
        return paths

    monkeypatch.setattr(e2e.subprocess, "run", run)
    monkeypatch.setattr(e2e, "_decode_reference_video", decode)
    result = e2e._official_reference(model_dir, manifest, case, tmp_path)

    command = captured["command"]
    assert command[1] == str(entrypoint)
    assert command[command.index("--action") + 1] == case["action"]
    assert command[command.index("--intrinsics") + 1] == str(
        e2e._asset(case["camera_intrinsics_file"])
    )
    assert command[command.index("--translation_speed") + 1] == "0.055"
    assert command[command.index("--rotation_speed_deg") + 1] == "1.2"
    assert command[command.index("--num_frames") + 1] == "3"
    assert command[command.index("--step") + 1] == "60"
    assert command[command.index("--cfg_scale") + 1] == "5.0"
    assert command[command.index("--fps") + 1] == "16"
    assert command[command.index("--flow_shift") + 1] == "9.8"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--refiner_seed") + 1] == "42"
    assert "--no_action_overlay" in command
    assert command[command.index("--config") + 1] == str(model_dir / "config.yaml")
    assert command[command.index("--model_path") + 1] == str(
        model_dir / "dit/sana_wm_1600m_720p.safetensors"
    )
    assert command[command.index("--refiner_root") + 1] == str(model_dir / "refiner")
    assert command[command.index("--refiner_gemma_root") + 1] == str(
        model_dir / "refiner/text_encoder"
    )
    assert command[command.index("--output_dir") + 1] == str(tmp_path / "reference-video")
    assert command[command.index("--name") + 1] == "reference"
    assert captured["kwargs"]["cwd"] == source
    assert captured["kwargs"]["check"] is True
    assert captured["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["kwargs"]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["kwargs"]["env"]["PYTHONPATH"] == str(source)
    assert captured["video_path"] == tmp_path / "reference-video/reference_generated.mp4"
    assert len(result["frame_paths"]) == 3


def test_official_reference_dependency_failure_is_not_hidden(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["sana-wm-bidirectional"]
    source = tmp_path / "source"
    entrypoint = source / "inference_video_scripts/wm/inference_sana_wm.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# declared official source\n", encoding="utf-8")
    model_dir = tmp_path / "raw-checkpoint"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(source))

    def fail(command, **kwargs):
        assert kwargs["check"] is True
        raise e2e.subprocess.CalledProcessError(
            1, command, stderr="ModuleNotFoundError: No module named 'pyrallis'"
        )

    monkeypatch.setattr(e2e.subprocess, "run", fail)
    with pytest.raises(e2e.subprocess.CalledProcessError):
        e2e._official_reference(model_dir, manifest, case, tmp_path)


def test_frame_stats_load_each_candidate_once(monkeypatch) -> None:
    actual_paths = [Path(f"actual-{index}") for index in range(3)]
    values = {
        path: np.full((2, 3, 3), 0.2 + index * 0.2, dtype=np.float32)
        for index, path in enumerate(actual_paths)
    }
    loaded = []

    def load(path: Path) -> np.ndarray:
        loaded.append(path)
        return values[path]

    monkeypatch.setattr(e2e, "_load_rgb", load)
    mean, std = e2e._frame_stats(actual_paths)

    assert loaded == actual_paths
    assert mean == pytest.approx(0.4)
    assert std == pytest.approx(np.std([0.2, 0.4, 0.6]))
