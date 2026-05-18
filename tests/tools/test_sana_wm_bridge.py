"""Tests for the SANA-WM Python bridge."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[2] / "tensorrt_model_connect"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from tensorrt_model_connect import sana_wm_bridge  # noqa: E402


def test_bridge_runs_official_script_and_materializes_frames(tmp_path, monkeypatch) -> None:
    script = tmp_path / "inference_sana_wm.py"
    script.write_text(
        """
import argparse
import shutil
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

out = Path(args.output_dir)
if out.exists():
    shutil.rmtree(out)
assert not Path(args.prompt).is_relative_to(out)
assert Path(args.prompt).read_text(encoding="utf-8") == "drive forward"
assert args.image.endswith("demo_0.png")
assert args.action == "w-80"
assert args.translation_speed == "0.055"
assert args.rotation_speed_deg == "1.2"
assert args.num_frames == "3"
out.mkdir(parents=True, exist_ok=True)
(out / "official_marker.txt").write_text("ok", encoding="utf-8")
""",
        encoding="utf-8",
    )
    image = tmp_path / "demo_0.png"
    image.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "official_out"
    frames_dir = tmp_path / "frames"
    meta_json = tmp_path / "meta.json"

    def fake_materialize(actual_output_dir, actual_frames_dir):
        assert actual_output_dir == output_dir
        assert actual_frames_dir == frames_dir
        assert (output_dir / "official_marker.txt").read_text(encoding="utf-8") == "ok"
        return {
            "frames_dir": str(frames_dir),
            "num_frames": 1,
            "height": 2,
            "width": 4,
            "channels": 3,
        }

    monkeypatch.setattr(sana_wm_bridge, "_materialize_frames", fake_materialize)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sana_wm_bridge",
            "--sana-script",
            str(script),
            "--image",
            str(image),
            "--prompt-text",
            "drive forward",
            "--action",
            "w-80",
            "--translation-speed",
            "0.055",
            "--rotation-speed-deg",
            "1.2",
            "--num-frames",
            "3",
            "--output-dir",
            str(output_dir),
            "--frames-dir",
            str(frames_dir),
            "--meta-json",
            str(meta_json),
        ],
    )

    assert sana_wm_bridge.main() == 0

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    assert meta["num_frames"] == 1
    assert meta["height"] == 2
    assert meta["width"] == 4


def test_bridge_diffusers_fallback_rejects_missing_action_controls(
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

    args = types.SimpleNamespace(
        hf_id="Efficient-Large-Model/SANA-WM_bidirectional",
        image=str(tmp_path / "demo_0.png"),
        prompt_text="drive forward",
        action="w-80",
        translation_speed=0.055,
        rotation_speed_deg=1.2,
        num_frames=321,
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(RuntimeError, match="camera-control arguments"):
        sana_wm_bridge._run_diffusers_fallback(args)


def test_bridge_no_diffusers_fallback_rejects_repo_local_shim(
    monkeypatch, tmp_path
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("SANA_WM_SCRIPT", raising=False)
    monkeypatch.delenv("SANA_REPO", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sana_wm_bridge",
            "--image",
            "asset/sana_wm/demo_0.png",
            "--prompt-text",
            "drive forward",
            "--action",
            "w-80",
            "--translation-speed",
            "0.055",
            "--rotation-speed-deg",
            "1.2",
            "--num-frames",
            "3",
            "--output-dir",
            str(tmp_path / "official_out"),
            "--frames-dir",
            str(tmp_path / "frames"),
            "--no-diffusers-fallback",
        ],
    )

    with pytest.raises(RuntimeError, match="official script not found"):
        sana_wm_bridge.main()
