#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
_SANA_REPOSITORY = "https://github.com/NVlabs/Sana.git"
_SANA_REFERENCE_COMMIT = "59629fdf790850797cb657bad014fce432bd713d"
_EXTERNAL_PATH_FLAGS = {
    "--camera",
    "--config",
    "--image",
    "--intrinsics",
    "--model_path",
    "--output_dir",
    "--prompt",
    "--refiner_checkpoint",
    "--refiner_gemma_root",
    "--refiner_root",
}


def _from_pretrained_compat(pipeline_cls, model_id: str, kwargs: dict):
    try:
        return pipeline_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        compat_kwargs = dict(kwargs)
        compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
        return pipeline_cls.from_pretrained(model_id, **compat_kwargs)


def _resolve_cached_model_ref(model_id: str) -> str:
    local_path = Path(model_id)
    if local_path.is_dir():
        return str(local_path)

    try:
        from huggingface_hub import hf_hub_download

        model_index = Path(
            hf_hub_download(
                repo_id=model_id,
                filename="model_index.json",
                local_files_only=True,
            )
        )
    except Exception:
        return model_id

    if model_index.is_file():
        return str(model_index.parent)
    return model_id


def _pipeline_accepts(callable_obj, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _storage_sana_script(storage_root: Path) -> Path:
    repo_root = (
        storage_root
        / "sana_wm"
        / "reference"
        / f"Sana-{_SANA_REFERENCE_COMMIT[:12]}"
    )
    script = repo_root / "inference_video_scripts" / "wm" / "inference_sana_wm.py"
    if script.is_file():
        return script
    if repo_root.exists():
        raise RuntimeError(f"Incomplete cached Sana reference checkout: {repo_root}")

    repo_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            _SANA_REPOSITORY,
            str(repo_root),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "checkout",
            "--detach",
            _SANA_REFERENCE_COMMIT,
        ],
        check=True,
    )
    if not script.is_file():
        raise RuntimeError(f"Pinned Sana checkout is missing its reference script: {script}")
    return script


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
            Path(sana_repo) / "inference_video_scripts" / "wm" / "inference_sana_wm.py"
        )
        candidates.append(
            Path(sana_repo) / "inference_video_scripts" / "inference_sana_wm.py"
        )
    storage_root = os.environ.get("TRTMC_STORAGE_ROOT", "")
    if storage_root:
        candidates.append(_storage_sana_script(Path(storage_root)))

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


def _external_repo_root(script_path: Path) -> Path:
    """Return the upstream Sana repo root for direct or wm/ script layouts."""
    for parent in script_path.parents:
        if (parent / "inference_video_scripts").is_dir() and (parent / "diffusion").is_dir():
            return parent
    if (
        script_path.parent.name == "wm"
        and script_path.parent.parent.name == "inference_video_scripts"
    ):
        return script_path.parent.parent.parent
    return script_path.parent.parent


def _resolve_external_path_args(args: list[str], launch_dir: Path) -> list[str]:
    resolved = list(args)
    for index, arg in enumerate(resolved[:-1]):
        if arg not in _EXTERNAL_PATH_FLAGS:
            continue
        value = Path(resolved[index + 1]).expanduser()
        if not value.is_absolute():
            resolved[index + 1] = str((launch_dir / value).resolve())
    return resolved


def _run_external_script(script_path: Path) -> None:
    repo_root = _external_repo_root(script_path)
    external_args = _resolve_external_path_args(sys.argv[1:], Path.cwd())
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not current_pythonpath
        else str(repo_root) + os.pathsep + current_pythonpath
    )
    runner = """
import importlib.util
import dataclasses
import enum
import os
import pickle
import sys
import types
import typing
from pathlib import Path

script_path = Path(sys.argv[1])
sys.argv = [str(script_path), *sys.argv[2:]]

def _install_pyrallis_stub():
    force_stub = os.environ.get("TRTMC_SANA_WM_FORCE_PYRALLIS_STUB") == "1"
    if not force_stub and importlib.util.find_spec("pyrallis") is not None:
        return

    import yaml

    def _coerce(annotation, value):
        if annotation is typing.Any or value is None:
            return value
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if origin in (typing.Union, types.UnionType):
            for candidate in args:
                if candidate is type(None):
                    continue
                try:
                    return _coerce(candidate, value)
                except (TypeError, ValueError):
                    continue
            return value
        if origin is list:
            item_type = args[0] if args else typing.Any
            return [_coerce(item_type, item) for item in value]
        if origin is tuple:
            item_types = args[:-1] if args and args[-1] is Ellipsis else args
            if len(item_types) == 1:
                return tuple(_coerce(item_types[0], item) for item in value)
            return tuple(
                _coerce(item_types[index], item) if index < len(item_types) else item
                for index, item in enumerate(value)
            )
        if origin is dict:
            key_type, value_type = args or (typing.Any, typing.Any)
            return {
                _coerce(key_type, key): _coerce(value_type, item)
                for key, item in value.items()
            }
        if dataclasses.is_dataclass(annotation):
            return _dataclass_from_dict(annotation, value)
        if annotation is Path:
            return Path(value)
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return annotation(value)
        return value

    def _dataclass_from_dict(config_class, values):
        if not isinstance(values, dict):
            raise TypeError(f"Expected mapping for {config_class.__name__}")
        hints = typing.get_type_hints(config_class)
        kwargs = {}
        for field in dataclasses.fields(config_class):
            if field.name in values:
                kwargs[field.name] = _coerce(
                    hints.get(field.name, field.type), values[field.name]
                )
        return config_class(**kwargs)

    def load(config_class, stream):
        return _dataclass_from_dict(config_class, yaml.safe_load(stream) or {})

    def parse(*, config_class, config_path, args=None):
        del args
        with Path(config_path).open("r", encoding="utf-8") as stream:
            return load(config_class, stream)

    pyrallis = types.ModuleType("pyrallis")
    pyrallis.__spec__ = importlib.util.spec_from_loader("pyrallis", loader=None)
    pyrallis.load = load
    pyrallis.parse = parse
    pyrallis._trtmc_stub = True
    sys.modules["pyrallis"] = pyrallis

_install_pyrallis_stub()

def _install_mmcv_registry_stub():
    if importlib.util.find_spec("mmcv") is not None:
        return

    class Config(dict):
        pass

    class Registry:
        def __init__(self, name, *args, **kwargs):
            self.name = name
            self.module_dict = {}

        def get(self, key):
            return self.module_dict.get(key)

        def register_module(self, module=None, name=None, force=False):
            def _register(obj):
                key = name or obj.__name__
                if not force and key in self.module_dict:
                    raise KeyError(f"{key} is already registered in {self.name}")
                self.module_dict[key] = obj
                return obj

            if module is not None:
                return _register(module)
            return _register

        def build(self, cfg, default_args=None):
            return build_from_cfg(cfg, self, default_args=default_args)

    def build_from_cfg(cfg, registry, default_args=None):
        if cfg is None:
            raise TypeError("cfg must be a dict")
        args = dict(default_args or {})
        args.update(dict(cfg))
        obj_type = args.pop("type")
        obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not registered in {registry.name}")
        return obj_cls(**args)

    mmcv = types.ModuleType("mmcv")
    mmcv.Config = Config
    mmcv.Registry = Registry
    mmcv.build_from_cfg = build_from_cfg
    mmcv.mkdir_or_exist = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    mmcv.dump = lambda obj, path: Path(path).write_bytes(pickle.dumps(obj))
    mmcv.load = lambda path: pickle.loads(Path(path).read_bytes())

    runner = types.ModuleType("mmcv.runner")
    runner.OPTIMIZERS = Registry("optimizers")
    runner.OPTIMIZER_BUILDERS = Registry("optimizer_builders")
    runner.DefaultOptimizerConstructor = object
    runner.build_optimizer = lambda *args, **kwargs: None
    runner.get_dist_info = lambda: (0, 1)

    utils = types.ModuleType("mmcv.utils")
    try:
        import torch.nn as nn

        utils._BatchNorm = nn.modules.batchnorm._BatchNorm
        utils._InstanceNorm = nn.modules.instancenorm._InstanceNorm
    except ImportError:
        class _BatchNorm:
            pass

        class _InstanceNorm:
            pass

        utils._BatchNorm = _BatchNorm
        utils._InstanceNorm = _InstanceNorm
    logging_mod = types.ModuleType("mmcv.utils.logging")
    logging_mod.logger_initialized = {}
    utils.logging = logging_mod

    mmcv.runner = runner
    mmcv.utils = utils
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.runner"] = runner
    sys.modules["mmcv.utils"] = utils
    sys.modules["mmcv.utils.logging"] = logging_mod

_install_mmcv_registry_stub()

def _install_imageio_writer_stub():
    imageio = types.ModuleType("imageio")
    imageio.__spec__ = importlib.util.spec_from_loader(
        "imageio", loader=None, is_package=True
    )
    imageio.__path__ = []
    imageio_v3 = types.ModuleType("imageio.v3")
    imageio_v3.__spec__ = importlib.util.spec_from_loader(
        "imageio.v3", loader=None
    )

    def _unexpected_imwrite(*args, **kwargs):
        raise RuntimeError("SANA-WM reference must use the PNG frame writer")

    imageio_v3.imwrite = _unexpected_imwrite
    imageio_v3._trtmc_stub = True
    imageio.v3 = imageio_v3
    sys.modules["imageio"] = imageio
    sys.modules["imageio.v3"] = imageio_v3

_install_imageio_writer_stub()

spec = importlib.util.spec_from_file_location("sana_wm_official_reference", script_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {script_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def _write_png_frames(output_dir, name, video_hwc, fps, logger):
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = np.asarray(video_hwc)
    if video.dtype != np.uint8:
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(video, 0.0, 1.0) * 255.0
        video = video.astype(np.uint8)
    if len(video) == 0:
        logger.info(f"Saved 0 PNG frames to {output_dir}")
        return output_dir
    from PIL import Image

    for idx, frame in enumerate(video):
        Image.fromarray(frame).convert("RGB").save(output_dir / f"frame_{idx:04d}.png")
    logger.info(f"Saved {len(video)} PNG frames to {output_dir}")
    return output_dir

module.write_video = _write_png_frames
module.main()
    """
    subprocess.run(
        [sys.executable, "-c", runner, str(script_path), *external_args],
        check=True,
        cwd=repo_root,
        env=env,
    )


def _load_pipeline():
    import torch
    from diffusers import DiffusionPipeline

    kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "cuda"

    model_ref = _resolve_cached_model_ref(_HF_ID)
    try:
        pipe = _from_pretrained_compat(DiffusionPipeline, model_ref, kwargs)
    except (NotImplementedError, ValueError):
        if kwargs.get("device_map") != "cuda":
            raise
        kwargs = dict(kwargs)
        kwargs.pop("device_map", None)
        pipe = _from_pretrained_compat(DiffusionPipeline, model_ref, kwargs)

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
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=60)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default="output")

    # Accepted for compatibility with the model-card script. Diffusers resolves
    # these assets from the HF repo; local override support belongs to the
    # upstream Sana runtime when it is available.
    parser.add_argument("--config", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--refiner_checkpoint", default="")
    parser.add_argument("--refiner_root", default="")
    parser.add_argument("--refiner_gemma_root", default="")
    parser.add_argument("--refiner_seed", type=int, default=42)
    parser.add_argument("--sink_size", type=int, default=1)
    parser.add_argument("--sampling_algo", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_action_overlay", action="store_true")
    parser.add_argument("--offload_vae", action="store_true")
    parser.add_argument("--offload_refiner", action="store_true")
    parser.add_argument("--no_refiner", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    external_script = _resolve_external_script()
    if external_script is not None:
        _run_external_script(external_script)
        return 0

    from diffusers.utils import load_image

    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
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
        "fps": args.fps,
        "step": args.step,
        "cfg_scale": args.cfg_scale,
        "flow_shift": args.flow_shift,
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
