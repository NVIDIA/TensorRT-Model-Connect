# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml

from .. import native_plugin_builder
from . import official_reference
from . import test_e2e as e2e
from .benchmark import prepare_environment


def _write_stage1_text_encoder(model_dir: Path) -> Path:
    stage1_text_encoder = model_dir / "stage1_text_encoder"
    stage1_text_encoder.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (stage1_text_encoder / name).write_text("{}\n", encoding="utf-8")
    return stage1_text_encoder


def test_qualification_candidate_uses_the_prepared_family_model() -> None:
    profile = yaml.safe_load(
        (Path(__file__).parent / "benchmark/sana-wm-bidirectional.yaml").read_text(encoding="utf-8")
    )

    prepared_model = "trtmc-reference/SANA-model"
    assert profile["candidate"]["model_directory"] == prepared_model
    assert profile["reference_environment"]["paths"]["model_dir"] == prepared_model


def test_native_plugin_uses_installed_tensorrt_library(tmp_path: Path) -> None:
    tensorrt_library = tmp_path / "tensorrt/libnvinfer.so.11"
    command = native_plugin_builder._configure_command(
        tmp_path / "source",
        tmp_path / "build",
        "/torch/cmake",
        tensorrt_library,
    )

    assert f"-DSANA_WM_TRT_LIBRARY={tensorrt_library}" in command


def test_native_plugin_discovers_tensorrt_python_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "tensorrt_libs"
    package.mkdir()
    library = package / "libnvinfer.so.11"
    library.touch()
    spec = native_plugin_builder.importlib.util.spec_from_file_location(
        "tensorrt_libs",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    monkeypatch.setattr(native_plugin_builder.importlib.util, "find_spec", lambda _name: spec)

    assert native_plugin_builder._installed_tensorrt_library() == library


def test_official_source_dependencies_are_family_owned() -> None:
    requirements = {
        line.strip()
        for line in (e2e.TEST_ROOT.parent / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "flash-linear-attention>=0.4.2",
        "imageio[ffmpeg,pyav]",
        "mmcv==1.7.2",
        "pyrallis",
        "pytz",
        "qwen-vl-utils",
        "termcolor",
    } <= requirements


def test_qualification_snapshot_is_materialized_inside_family_environment(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    blob = cache / "blobs/weights"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"weights")
    snapshot = cache / "snapshots/revision"
    (snapshot / "dit").mkdir(parents=True)
    (snapshot / "dit/model.safetensors").symlink_to(blob)
    environment = tmp_path / "environment"
    environment.mkdir()
    destination = environment / "SANA-model"

    prepare_environment._materialize_snapshot(snapshot, destination)

    materialized = destination / "dit/model.safetensors"
    assert destination.is_dir() and not destination.is_symlink()
    assert destination.resolve().is_relative_to(environment)
    assert materialized.read_bytes() == b"weights"
    assert materialized.samefile(blob)
    assert prepare_environment.MODEL_REVISION == "e96271d77398def8ebb9fc595e7c0056dc625ab7"


def test_qualification_materializes_pinned_stage1_text_encoder(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        (snapshot / name).write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    captured = {}

    def download(model_id, **kwargs):
        captured.update(model_id=model_id, **kwargs)
        return str(snapshot)

    monkeypatch.setattr(prepare_environment, "snapshot_download", download)
    prepare_environment._materialize_stage1_text_encoder(model)

    destination = model / "stage1_text_encoder"
    assert destination.is_dir() and not destination.is_symlink()
    assert (destination / "tokenizer.json").is_file()
    assert captured == {
        "model_id": "Efficient-Large-Model/gemma-2-2b-it",
        "revision": "569d9809d0c8b6722d4d31b5a77a2ec7a400650a",
    }


def test_official_reference_loads_stage1_encoder_from_explicit_local_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage1_text_encoder = tmp_path / "stage1-text-encoder"
    stage1_text_encoder.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (stage1_text_encoder / name).write_text("{}\n", encoding="utf-8")
    calls: list[tuple[object, ...]] = []

    class Decoder:
        def to(self, device: str):
            calls.append(("to", device))
            return "local-decoder"

    class Model:
        def get_decoder(self):
            calls.append(("get_decoder",))
            return Decoder()

    class AutoTokenizer:
        padding_side = "left"

        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return AutoTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls.append(("model", path, kwargs))
            return Model()

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    transformers.AutoTokenizer = AutoTokenizer
    torch = ModuleType("torch")
    torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    tokenizer, decoder = official_reference.load_local_stage1_text_encoder(
        stage1_text_encoder, "cuda:0"
    )
    assert isinstance(tokenizer, AutoTokenizer)
    assert tokenizer.padding_side == "right"
    assert decoder == "local-decoder"
    assert calls == [
        ("tokenizer", str(stage1_text_encoder), {"local_files_only": True}),
        (
            "model",
            str(stage1_text_encoder),
            {"local_files_only": True, "torch_dtype": "bf16"},
        ),
        ("get_decoder",),
        ("to", "cuda:0"),
    ]


def test_qualification_environment_replaces_gui_opencv(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        prepare_environment.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    prepare_environment._install_headless_opencv()

    assert calls[0][0][-2:] == ["--yes", "opencv-python"]
    assert calls[0][1]["check"] is False
    assert calls[1][0][-1] == "opencv-python-headless==4.11.0.86"
    assert calls[1][1]["check"] is True


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
    stage1_text_encoder = _write_stage1_text_encoder(model_dir)
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    assert not (model_dir / "model_index.json").exists()
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(source))
    captured = {}
    observations = {}

    def record(name, value):
        observations[name] = value
        return value

    monkeypatch.setattr(e2e, "record_evidence", record)

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--output_dir") + 1])
        output.mkdir()
        (output / "reference_generated.mp4").write_bytes(b"video")
        return e2e.subprocess.CompletedProcess(
            command, 0, stdout="reference progress\n", stderr="reference diagnostic\n"
        )

    def decode(video_path: Path, frames_dir: Path):
        captured["video_path"] = video_path
        frames_dir.mkdir()
        paths = []
        for index in range(2):
            path = frames_dir / f"frame_{index:04d}.png"
            path.write_bytes(b"png")
            paths.append(path)
        return paths

    monkeypatch.setattr(e2e.subprocess, "run", run)
    monkeypatch.setattr(e2e, "_decode_reference_video", decode)
    result = e2e._official_reference(model_dir, manifest, case, tmp_path)

    command = captured["command"]
    assert command[1] == str(e2e.TEST_ROOT / "official_reference.py")
    assert command[command.index("--reference-repo") + 1] == str(source)
    assert command[command.index("--stage1-text-encoder") + 1] == str(stage1_text_encoder)
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
    assert len(result["frame_paths"]) == 2
    assert observations["reference_process"] == {
        "argv": command,
        "stdout": "reference progress\n",
        "stderr": "reference diagnostic\n",
    }


def test_official_reference_dependency_failure_is_not_hidden(monkeypatch, tmp_path: Path) -> None:
    _, manifest, case = e2e.CASES["sana-wm-bidirectional"]
    source = tmp_path / "source"
    entrypoint = source / "inference_video_scripts/wm/inference_sana_wm.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# declared official source\n", encoding="utf-8")
    model_dir = tmp_path / "raw-checkpoint"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    _write_stage1_text_encoder(model_dir)
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


@pytest.mark.parametrize(
    ("frame_indices", "passes"),
    [
        (list(range(319)), False),
        (list(range(320)), True),
        (list(range(321)), False),
        ([*range(319), 320], False),
    ],
    ids=["short", "complete", "extra", "gap"],
)
def test_official_refiner_requires_every_output_frame(
    frame_indices, passes, monkeypatch, tmp_path: Path
) -> None:
    _, manifest, case = e2e.CASES["sana-wm-bidirectional"]
    source = tmp_path / "source"
    entrypoint = source / "inference_video_scripts/wm/inference_sana_wm.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# declared official source\n", encoding="utf-8")
    model_dir = tmp_path / "checkpoint"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    _write_stage1_text_encoder(model_dir)
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("TRTMC_REFERENCE_SOURCE_DIR", str(source))
    monkeypatch.setattr(
        e2e.subprocess,
        "run",
        lambda command, **kwargs: e2e.subprocess.CompletedProcess(
            command, 0, stdout="reference completed", stderr=""
        ),
    )
    frames = [tmp_path / f"frame_{index:04d}.png" for index in frame_indices]
    monkeypatch.setattr(e2e, "_decode_reference_video", lambda *_: frames)

    if passes:
        assert e2e._official_reference(model_dir, manifest, case, tmp_path) == {
            "frame_paths": frames
        }
    else:
        with pytest.raises(AssertionError):
            e2e._official_reference(model_dir, manifest, case, tmp_path)
