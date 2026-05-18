#!/usr/bin/env python3
"""SANA-WM model-card-compatible inference entrypoint.

This script preserves the command surface documented by
Efficient-Large-Model/SANA-WM_bidirectional. It is a local compatibility shim:
it delegates to DiffusionPipeline only when the installed pipeline explicitly
exposes the SANA-WM camera-control arguments needed for parity.
"""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path


_HF_ID = "Efficient-Large-Model/SANA-WM_bidirectional"


def _from_pretrained_compat(pipeline_cls, model_id: str, kwargs: dict):
    try:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        compat_kwargs = dict(kwargs)
        compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
        return pipeline_cls.from_pretrained(model_id, **compat_kwargs)


def _pipeline_accepts(callable_obj, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _require_pipeline_controls(callable_obj, names: list[str]) -> None:
    missing = [name for name in names if not _pipeline_accepts(callable_obj, name)]
    if missing:
        raise RuntimeError(
            "Loaded Diffusers pipeline does not expose SANA-WM camera-control "
            "arguments: "
            + ", ".join(missing)
            + ". Use the official inference_video_scripts/inference_sana_wm.py "
            "runtime when validating SANA-WM action parity."
        )


def _resolve_external_script() -> Path | None:
    candidates: list[Path] = []
    env_script = os.environ.get("SANA_WM_SCRIPT", "")
    if env_script:
        candidates.append(Path(env_script))
    sana_repo = os.environ.get("SANA_REPO", "")
    if sana_repo:
        candidates.append(
            Path(sana_repo) / "inference_video_scripts" / "inference_sana_wm.py"
        )

    local_path = Path(__file__).resolve()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            if candidate.resolve() == local_path:
                continue
        except OSError:
            pass
        return candidate
    return None


def _run_external_script(script_path: Path) -> None:
    subprocess.run([sys.executable, str(script_path), *sys.argv[1:]], check=True)


def _load_pipeline():
    import torch
    from diffusers import DiffusionPipeline

    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "cuda"

    try:
        pipe = _from_pretrained_compat(DiffusionPipeline, _HF_ID, kwargs)
    except (NotImplementedError, ValueError):
        if kwargs.get("device_map") != "cuda":
            raise
        kwargs = dict(kwargs)
        kwargs.pop("device_map", None)
        pipe = _from_pretrained_compat(DiffusionPipeline, _HF_ID, kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe


def _frame_to_image(frame):
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if hasattr(frame, "detach"):
        tensor = frame.detach().cpu()
        if getattr(tensor, "ndim", 0) == 3 and int(tensor.shape[0]) in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 0)
        if getattr(tensor, "is_floating_point", lambda: False)():
            tensor = tensor.clamp(0, 1).mul(255).byte()
        frame = tensor.numpy()

    import numpy as np

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0, 1) * 255
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _save_frames(frames, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        _frame_to_image(frame).save(output_dir / f"frame_{idx:04d}.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    control = parser.add_mutually_exclusive_group()
    control.add_argument("--camera", default="")
    control.add_argument("--action", default="")
    parser.add_argument("--intrinsics", default="")
    parser.add_argument("--translation_speed", type=float, default=0.055)
    parser.add_argument("--rotation_speed_deg", type=float, default=1.2)
    parser.add_argument("--num_frames", type=int, default=321)
    parser.add_argument("--output_dir", required=True)

    # Accepted for compatibility with the model-card script. Diffusers resolves
    # these assets from the HF repo; local override support belongs to the
    # upstream Sana runtime when it is available.
    parser.add_argument("--config", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--refiner_checkpoint", default="")
    parser.add_argument("--refiner_gemma_root", default="")
    parser.add_argument("--no_refiner", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    external_script = _resolve_external_script()
    if external_script is not None:
        _run_external_script(external_script)
        return 0

    from diffusers.utils import load_image

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    pipe = _load_pipeline()

    kwargs = {
        "image": load_image(args.image),
        "prompt": prompt,
    }
    optional = {
        "action": args.action,
        "camera": args.camera,
        "intrinsics": args.intrinsics,
        "translation_speed": args.translation_speed,
        "rotation_speed_deg": args.rotation_speed_deg,
        "num_frames": args.num_frames,
        "no_refiner": args.no_refiner,
    }
    required_controls = ["translation_speed", "rotation_speed_deg", "num_frames"]
    if args.action:
        required_controls.append("action")
    if args.camera:
        required_controls.append("camera")
    if args.intrinsics:
        required_controls.append("intrinsics")
    _require_pipeline_controls(pipe.__call__, required_controls)

    for key, value in optional.items():
        if value in ("", None) or value is False:
            continue
        if _pipeline_accepts(pipe.__call__, key):
            kwargs[key] = value

    output = pipe(**kwargs)
    frames = getattr(output, "frames", None)
    if frames is None:
        frames = getattr(output, "images", None)
    if frames is None:
        raise RuntimeError("SANA-WM pipeline produced no frames/images")
    if frames and isinstance(frames[0], list):
        frames = frames[0]

    _save_frames(frames, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
