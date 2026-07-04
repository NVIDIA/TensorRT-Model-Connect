# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SANA-WM model-card inference entrypoint."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parent / "reference" / "inference_sana_wm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("inference_sana_wm", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clear_sana_reference_env(monkeypatch) -> None:
    for name in (
        "SANA_WM_SCRIPT",
        "SANA_REPO",
        "TRTMC_STORAGE_ROOT",
        "TRTMC_SANA_WM_FORCE_FLA_STUB",
        "TRTMC_SANA_WM_FORCE_PYRALLIS_STUB",
    ):
        monkeypatch.delenv(name, raising=False)


def test_inference_sana_wm_maps_model_card_args_to_diffusers(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_id"] = model_id
            captured["load_kwargs"] = kwargs
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

        def __call__(
            self,
            *,
            image,
            prompt,
            action=None,
            translation_speed=None,
            rotation_speed_deg=None,
            num_frames=None,
        ):
            captured["call_kwargs"] = {
                "image": image,
                "prompt": prompt,
                "action": action,
                "translation_speed": translation_speed,
                "rotation_speed_deg": rotation_speed_deg,
                "num_frames": num_frames,
            }
            return types.SimpleNamespace(frames=[["frame0", "frame1"]])

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = object()
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
    diffusers_mod = types.ModuleType("diffusers")
    diffusers_mod.DiffusionPipeline = _Pipeline
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.load_image = lambda path: f"image:{path}"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)

    prompt_file = tmp_path / "demo_0.txt"
    prompt_file.write_text("drive forward", encoding="utf-8")
    out_dir = tmp_path / "results" / "demo"
    module = _load_module()

    def fake_save_frames(frames, output_dir):
        captured["saved_frames"] = {"frames": frames, "output_dir": output_dir}

    monkeypatch.setattr(module, "_save_frames", fake_save_frames)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_sana_wm.py",
            "--image",
            "asset/sana_wm/demo_0.png",
            "--prompt",
            str(prompt_file),
            "--action",
            "w-80,jw-40,w-40,lw-60,w-100",
            "--translation_speed",
            "0.055",
            "--rotation_speed_deg",
            "1.2",
            "--num_frames",
            "321",
            "--output_dir",
            str(out_dir),
        ],
    )

    assert module.main() == 0

    assert captured["model_id"] == "Efficient-Large-Model/SANA-WM_bidirectional"
    assert captured["load_kwargs"]["trust_remote_code"] is True
    assert captured["load_kwargs"]["device_map"] == "cuda"
    assert captured["device"] == "cuda"
    assert captured["call_kwargs"] == {
        "image": "image:asset/sana_wm/demo_0.png",
        "prompt": "drive forward",
        "action": "w-80,jw-40,w-40,lw-60,w-100",
        "translation_speed": 0.055,
        "rotation_speed_deg": 1.2,
        "num_frames": 321,
    }
    assert captured["saved_frames"] == {
        "frames": ["frame0", "frame1"],
        "output_dir": out_dir,
    }


def test_inference_sana_wm_loads_warmed_snapshot_offline(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_id"] = model_id
            captured["load_kwargs"] = kwargs
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

    def fake_hf_hub_download(*, repo_id, filename, local_files_only):
        captured["hf_hub_download"] = {
            "repo_id": repo_id,
            "filename": filename,
            "local_files_only": local_files_only,
        }
        return str(snapshot / filename)

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = object()
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    diffusers_mod = types.ModuleType("diffusers")
    diffusers_mod.DiffusionPipeline = _Pipeline
    huggingface_hub_mod = types.ModuleType("huggingface_hub")
    huggingface_hub_mod.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub_mod)

    module = _load_module()
    module._load_pipeline()

    assert captured["hf_hub_download"] == {
        "repo_id": "Efficient-Large-Model/SANA-WM_bidirectional",
        "filename": "model_index.json",
        "local_files_only": True,
    }
    assert captured["model_id"] == str(snapshot)
    assert captured["device"] == "cpu"


def test_inference_sana_wm_delegates_to_external_official_script(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    sana_repo = tmp_path / "official_sana"
    external = sana_repo / "inference_video_scripts" / "wm" / "inference_sana_wm.py"
    external.parent.mkdir(parents=True)
    (sana_repo / "official_probe.py").write_text(
        "VALUE = 'official-pythonpath-ok'\n",
        encoding="utf-8",
    )
    (external.parent / "config.yaml").write_text(
        "nested:\n  value: 17\n",
        encoding="utf-8",
    )
    marker = tmp_path / "official_ran.txt"
    external.write_text(
        """
import argparse
from dataclasses import dataclass
from fla.modules import ShortConvolution
import imageio.v3 as iio
import logging
import numpy as np
from pathlib import Path
import official_probe
import pyrallis
import torch

@dataclass
class NestedConfig:
    value: int

@dataclass
class Config:
    nested: NestedConfig

def write_video(output_dir, name, video_hwc, fps, logger):
    raise AssertionError("shim should replace official write_video")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--translation_speed", required=True)
    parser.add_argument("--rotation_speed_deg", required=True)
    parser.add_argument("--num_frames", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    assert Path(args.image).read_bytes() == b"test image"
    assert Path(args.prompt).read_text(encoding="utf-8") == "drive forward"
    assert Path(args.output_dir).is_absolute()
    assert args.action == "w-80,jw-40,w-40,lw-60,w-100"
    assert args.translation_speed == "0.055"
    assert args.rotation_speed_deg == "1.2"
    assert args.num_frames == "321"
    assert official_probe.VALUE == "official-pythonpath-ok"
    assert iio._trtmc_stub is True
    assert iio.__spec__ is not None
    assert pyrallis._trtmc_stub is True
    assert ShortConvolution._trtmc_stub is True
    conv = ShortConvolution(hidden_size=1, kernel_size=2, activation=None)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[2.0, 3.0]]]))
    convolved, state = conv(torch.tensor([[[1.0], [2.0], [3.0]]]))
    assert torch.equal(convolved, torch.tensor([[[3.0], [8.0], [13.0]]]))
    assert state is None
    config = pyrallis.parse(
        config_class=Config,
        config_path=Path(__file__).with_name("config.yaml"),
        args=[],
    )
    assert config.nested.value == 17
    video = np.zeros((0, 1, 1, 3), dtype=np.uint8)
    write_video(Path(args.output_dir), "demo", video, 16, logging.getLogger("test"))
    Path({marker!r}).write_text(Path.cwd().name + ":" + args.output_dir, encoding="utf-8")
""".format(marker=str(marker)),
        encoding="utf-8",
    )

    image_file = tmp_path / "asset" / "sana_wm" / "demo_0.png"
    image_file.parent.mkdir(parents=True)
    image_file.write_bytes(b"test image")
    prompt_file = tmp_path / "demo_0.txt"
    prompt_file.write_text("drive forward", encoding="utf-8")
    output_dir = tmp_path / "results" / "demo"
    monkeypatch.setenv("SANA_REPO", str(sana_repo))
    monkeypatch.setenv("TRTMC_SANA_WM_FORCE_FLA_STUB", "1")
    monkeypatch.setenv("TRTMC_SANA_WM_FORCE_PYRALLIS_STUB", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_sana_wm.py",
            "--image",
            "asset/sana_wm/demo_0.png",
            "--prompt",
            str(prompt_file),
            "--action",
            "w-80,jw-40,w-40,lw-60,w-100",
            "--translation_speed",
            "0.055",
            "--rotation_speed_deg",
            "1.2",
            "--num_frames",
            "321",
            "--output_dir",
            "results/demo",
        ],
    )

    assert module.main() == 0
    assert marker.read_text(encoding="utf-8") == f"official_sana:{output_dir}"


def test_inference_sana_wm_discovers_storage_root_official_script(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    external = (
        tmp_path
        / "sana_wm"
        / "reference"
        / f"Sana-{module._SANA_REFERENCE_COMMIT[:12]}"
        / "inference_video_scripts"
        / "wm"
        / "inference_sana_wm.py"
    )
    external.parent.mkdir(parents=True)
    external.write_text("# official SANA-WM entrypoint\n", encoding="utf-8")
    monkeypatch.delenv("SANA_WM_SCRIPT", raising=False)
    monkeypatch.delenv("SANA_REPO", raising=False)
    monkeypatch.setenv("TRTMC_STORAGE_ROOT", str(tmp_path))

    assert module._resolve_external_script() == external


def test_inference_sana_wm_bootstraps_pinned_storage_reference(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    commands: list[list[str]] = []

    def fake_run(command, *, check, **_kwargs):
        assert check is True
        commands.append(command)
        if command[1] == "clone":
            Path(command[-1]).mkdir(parents=True)
        else:
            repo_root = Path(command[2])
            script = (
                repo_root
                / "inference_video_scripts"
                / "wm"
                / "inference_sana_wm.py"
            )
            script.parent.mkdir(parents=True)
            script.write_text("# official SANA-WM entrypoint\n", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("TRTMC_STORAGE_ROOT", str(tmp_path))

    script = module._resolve_external_script()

    repo_root = (
        tmp_path
        / "sana_wm"
        / "reference"
        / f"Sana-{module._SANA_REFERENCE_COMMIT[:12]}"
    )
    assert script == (
        repo_root / "inference_video_scripts" / "wm" / "inference_sana_wm.py"
    )
    assert commands == [
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            module._SANA_REPOSITORY,
            str(repo_root),
        ],
        [
            "git",
            "-C",
            str(repo_root),
            "checkout",
            "--detach",
            module._SANA_REFERENCE_COMMIT,
        ],
    ]


def test_inference_sana_wm_rejects_pipeline_without_action_controls(
    monkeypatch, tmp_path
) -> None:
    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

        def to(self, device):
            return self

        def __call__(self, *, image, prompt):
            return types.SimpleNamespace(frames=[["frame0"]])

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = object()
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    diffusers_mod = types.ModuleType("diffusers")
    diffusers_mod.DiffusionPipeline = _Pipeline
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.load_image = lambda path: f"image:{path}"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)

    prompt_file = tmp_path / "demo_0.txt"
    prompt_file.write_text("drive forward", encoding="utf-8")
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference_sana_wm.py",
            "--image",
            "asset/sana_wm/demo_0.png",
            "--prompt",
            str(prompt_file),
            "--action",
            "w-80",
            "--translation_speed",
            "0.055",
            "--rotation_speed_deg",
            "1.2",
            "--num_frames",
            "321",
            "--output_dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(RuntimeError, match="camera-control arguments"):
        module.main()
