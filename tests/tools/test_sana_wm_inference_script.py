"""Tests for the SANA-WM model-card inference entrypoint."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "inference_video_scripts"
    / "inference_sana_wm.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("inference_sana_wm", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_inference_sana_wm_delegates_to_external_official_script(
    monkeypatch, tmp_path
) -> None:
    module = _load_module()
    sana_repo = tmp_path / "official_sana"
    external = sana_repo / "inference_video_scripts" / "inference_sana_wm.py"
    external.parent.mkdir(parents=True)
    marker = tmp_path / "official_ran.txt"
    external.write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--action", required=True)
parser.add_argument("--translation_speed", required=True)
parser.add_argument("--rotation_speed_deg", required=True)
parser.add_argument("--num_frames", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

assert args.image == "asset/sana_wm/demo_0.png"
assert args.action == "w-80,jw-40,w-40,lw-60,w-100"
assert args.translation_speed == "0.055"
assert args.rotation_speed_deg == "1.2"
assert args.num_frames == "321"
Path({marker!r}).write_text(args.output_dir, encoding="utf-8")
""".format(marker=str(marker)),
        encoding="utf-8",
    )

    prompt_file = tmp_path / "demo_0.txt"
    prompt_file.write_text("drive forward", encoding="utf-8")
    monkeypatch.setenv("SANA_REPO", str(sana_repo))
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
    assert marker.read_text(encoding="utf-8") == "results/demo"


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
