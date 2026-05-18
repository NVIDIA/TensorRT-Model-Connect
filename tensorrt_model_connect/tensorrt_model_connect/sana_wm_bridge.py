"""Runtime bridge for SANA-WM image-to-video generation.

The preferred path delegates to NVlabs/Sana's
``inference_video_scripts/inference_sana_wm.py`` so TRTMC and the official
Python command use the same camera/action contract. If that script is not
available, the bridge attempts a DiffusionPipeline fallback only when the
loaded pipeline explicitly exposes the SANA-WM camera/action arguments needed
for parity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _write_prompt_file(prompt: str, output_dir: Path) -> Path:
    prompt_dir = output_dir.parent if output_dir.parent != Path("") else Path(".")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{output_dir.name or 'sana_wm'}_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def _candidate_scripts(explicit: str) -> Iterable[Path]:
    if explicit:
        yield Path(explicit)

    env_script = os.environ.get("SANA_WM_SCRIPT", "")
    if env_script:
        yield Path(env_script)

    sana_repo = os.environ.get("SANA_REPO", "")
    if sana_repo:
        yield Path(sana_repo) / "inference_video_scripts" / "inference_sana_wm.py"

    cwd = Path.cwd()
    yield cwd / "inference_video_scripts" / "inference_sana_wm.py"
    yield cwd / "Sana" / "inference_video_scripts" / "inference_sana_wm.py"


def _repo_local_shim_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "inference_video_scripts"
        / "inference_sana_wm.py"
    )


def _is_repo_local_shim(path: Path) -> bool:
    try:
        return path.resolve() == _repo_local_shim_path().resolve()
    except OSError:
        return False


def _resolve_official_script(explicit: str, *, require_external: bool = False) -> Path | None:
    for path in _candidate_scripts(explicit):
        if path.is_file():
            if require_external and _is_repo_local_shim(path):
                continue
            return path
    return None


def _optional_env_arg(cmd: list[str], env_name: str, flag: str) -> None:
    value = os.environ.get(env_name, "")
    if value:
        cmd.extend([flag, value])


def _run_official_script(args: argparse.Namespace, script_path: Path) -> None:
    prompt_file = _write_prompt_file(args.prompt_text, Path(args.output_dir))

    cmd = [
        sys.executable,
        str(script_path),
        "--image",
        args.image,
        "--prompt",
        str(prompt_file),
        "--action",
        args.action,
        "--translation_speed",
        str(args.translation_speed),
        "--rotation_speed_deg",
        str(args.rotation_speed_deg),
        "--num_frames",
        str(args.num_frames),
        "--output_dir",
        args.output_dir,
    ]
    _optional_env_arg(cmd, "SANA_WM_CONFIG", "--config")
    _optional_env_arg(cmd, "SANA_WM_MODEL_PATH", "--model_path")
    _optional_env_arg(cmd, "SANA_WM_REFINER_CHECKPOINT", "--refiner_checkpoint")
    _optional_env_arg(cmd, "SANA_WM_REFINER_GEMMA_ROOT", "--refiner_gemma_root")
    if os.environ.get("SANA_WM_NO_REFINER", "").lower() in {"1", "true", "yes"}:
        cmd.append("--no_refiner")

    subprocess.run(cmd, check=True)


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
            + ". Set SANA_WM_SCRIPT/SANA_REPO to an official runtime checkout "
            "when validating SANA-WM action parity."
        )


def _from_pretrained_compat(pipeline_cls, model_id: str, kwargs: dict):
    try:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        compat_kwargs = dict(kwargs)
        compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
        return pipeline_cls.from_pretrained(model_id, **compat_kwargs)


def _run_diffusers_fallback(args: argparse.Namespace) -> None:
    import torch
    from diffusers import DiffusionPipeline
    from diffusers.utils import load_image

    load_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "cuda"

    try:
        pipe = _from_pretrained_compat(DiffusionPipeline, args.hf_id, load_kwargs)
    except (NotImplementedError, ValueError):
        if load_kwargs.get("device_map") != "cuda":
            raise
        load_kwargs = dict(load_kwargs)
        load_kwargs.pop("device_map", None)
        pipe = _from_pretrained_compat(DiffusionPipeline, args.hf_id, load_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    kwargs = {
        "image": load_image(args.image),
        "prompt": args.prompt_text,
    }
    optional = {
        "action": args.action,
        "translation_speed": args.translation_speed,
        "rotation_speed_deg": args.rotation_speed_deg,
        "num_frames": args.num_frames,
    }
    _require_pipeline_controls(pipe.__call__, list(optional))
    for key, value in optional.items():
        if _pipeline_accepts(pipe.__call__, key):
            kwargs[key] = value

    output = pipe(**kwargs)
    frames = getattr(output, "frames", None)
    if frames is None:
        frames = getattr(output, "images", None)
    if frames is None:
        raise RuntimeError("SANA-WM DiffusionPipeline produced no frames/images")
    if frames and isinstance(frames[0], list):
        frames = frames[0]

    _save_frames_as_images(frames, Path(args.output_dir) / "sana_wm_fallback_frames")


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


def _save_frames_as_images(frames, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        _frame_to_image(frame).save(output_dir / f"frame_{idx:04d}.png")


def _iter_media_files(root: Path) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            images.append(path)
        elif suffix in VIDEO_SUFFIXES:
            videos.append(path)
    return images, videos


def _materialize_frames(output_dir: Path, frames_dir: Path) -> dict:
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    images, videos = _iter_media_files(output_dir)
    frame_count = 0

    if videos:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError(
                "SANA-WM output contained video, and imageio is required to "
                "extract frames from it"
            ) from exc
        for video in videos[:1]:
            for frame in iio.imiter(video):
                dst = frames_dir / f"frame_{frame_count:04d}.png"
                Image.fromarray(frame).convert("RGB").save(dst)
                frame_count += 1
    elif images:
        for path in images:
            dst = frames_dir / f"frame_{frame_count:04d}.png"
            Image.open(path).convert("RGB").save(dst)
            frame_count += 1
    else:
        raise RuntimeError(f"No image or video output found under {output_dir}")

    first = Image.open(frames_dir / "frame_0000.png").convert("RGB")
    return {
        "frames_dir": str(frames_dir),
        "num_frames": frame_count,
        "height": first.height,
        "width": first.width,
        "channels": 3,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", default="Efficient-Large-Model/SANA-WM_bidirectional")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--translation-speed", type=float, required=True)
    parser.add_argument("--rotation-speed-deg", type=float, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--meta-json", default="")
    parser.add_argument("--sana-script", default="")
    parser.add_argument("--no-diffusers-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    frames_dir = Path(args.frames_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    script_path = _resolve_official_script(
        args.sana_script, require_external=args.no_diffusers_fallback
    )
    if script_path is not None:
        _run_official_script(args, script_path)
    elif args.no_diffusers_fallback:
        raise RuntimeError(
            "SANA-WM official script not found. Set SANA_WM_SCRIPT or "
            "SANA_REPO to a checkout containing "
            "inference_video_scripts/inference_sana_wm.py"
        )
    else:
        print(
            "WARNING: SANA-WM official script not found; using DiffusionPipeline "
            "fallback when available",
            file=sys.stderr,
        )
        _run_diffusers_fallback(args)

    meta = _materialize_frames(output_dir, frames_dir)
    if args.meta_json:
        Path(args.meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
