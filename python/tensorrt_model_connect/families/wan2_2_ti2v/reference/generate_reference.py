#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official Wan2.2 TI2V-5B 720p text-to-video contract.

The official repository imports optional S2V/Animate dependencies eagerly.
Apply ``official-ti2v-only-import.patch`` to that checkout before invoking this
script.  The patch changes imports only; the TI2V model and sampling math remain
the official implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage"
)
MODEL_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--save-latent",
        action="store_true",
        help="Save the normalized latent passed to the official VAE decoder",
    )
    parser.add_argument(
        "--save-latent-trajectory",
        action="store_true",
        help="Save all 50 official scheduler outputs for closed-loop drift analysis",
    )
    args = parser.parse_args()

    args.official_source = args.official_source.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.official_source))
    from wan.configs import ti2v_5B  # pylint: disable=import-outside-toplevel
    import wan.textimage2video as textimage2video  # pylint: disable=import-outside-toplevel
    from wan.textimage2video import WanTI2V  # pylint: disable=import-outside-toplevel
    from wan.utils.utils import save_video  # pylint: disable=import-outside-toplevel

    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    pipeline = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=str(args.checkpoint),
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    )
    captured_latents: list[torch.Tensor] = []
    if args.save_latent:
        official_decode = pipeline.vae.decode

        def capture_decode(zs: list[torch.Tensor]):
            captured_latents.extend(latent.detach().cpu() for latent in zs)
            return official_decode(zs)

        pipeline.vae.decode = capture_decode
    captured_trajectory: list[torch.Tensor] = []
    official_scheduler_step = textimage2video.FlowUniPCMultistepScheduler.step
    if args.save_latent_trajectory:

        def capture_scheduler_step(scheduler, *step_args, **step_kwargs):
            result = official_scheduler_step(scheduler, *step_args, **step_kwargs)
            captured_trajectory.append(result[0].detach().squeeze(0).float().cpu())
            return result

        textimage2video.FlowUniPCMultistepScheduler.step = capture_scheduler_step
    loaded = time.perf_counter()
    try:
        video = pipeline.generate(
            args.prompt,
            img=None,
            size=(1280, 704),
            max_area=1280 * 704,
            frame_num=121,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=50,
            guide_scale=5.0,
            seed=args.seed,
            offload_model=False,
        )
    finally:
        textimage2video.FlowUniPCMultistepScheduler.step = official_scheduler_step
    torch.cuda.synchronize(args.device)
    generated = time.perf_counter()

    torch.save(video.cpu(), args.output / "video_fp32.pt")
    latent_metadata = None
    if args.save_latent:
        if len(captured_latents) != 1:
            raise RuntimeError(
                "Expected one normalized latent from the official VAE decode, "
                f"captured {len(captured_latents)}"
            )
        latent_path = args.output / "latent_fp32.pt"
        latent = captured_latents[0].float()
        torch.save(latent, latent_path)
        latent_metadata = {
            "path": latent_path.name,
            "shape": list(latent.shape),
            "dtype": str(latent.dtype),
            "sha256": _sha256(latent_path),
        }
    trajectory_metadata = None
    if args.save_latent_trajectory:
        if len(captured_trajectory) != 50:
            raise RuntimeError(
                f"Expected 50 official scheduler outputs, captured {len(captured_trajectory)}"
            )
        trajectory_path = args.output / "latent_trajectory_fp32.pt"
        trajectory = torch.stack(captured_trajectory)
        torch.save(trajectory, trajectory_path)
        trajectory_metadata = {
            "path": trajectory_path.name,
            "shape": list(trajectory.shape),
            "dtype": str(trajectory.dtype),
            "sha256": _sha256(trajectory_path),
        }
    uint8_frames = (
        ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    )
    frames_dir = args.output / "frames"
    frames_dir.mkdir()
    frame_hashes = []
    for index, frame in enumerate(uint8_frames):
        path = frames_dir / f"frame_{index:04d}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        frame_hashes.append(_sha256(path))
    save_video(
        video[None],
        save_file=str(args.output / "official.mp4"),
        fps=24,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    saved = time.perf_counter()

    metadata = {
        "schema_version": 1,
        "kind": "wan2_2_ti2v_5b_official_reference",
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "model_revision": MODEL_REVISION,
        "official_source_commit": _official_commit(args.official_source),
        "prompt": args.prompt,
        "seed": args.seed,
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "sample_solver": "unipc",
        "offload_model": False,
        "convert_model_dtype": False,
        "t5_cpu": False,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(args.device),
        "load_seconds": loaded - started,
        "generation_seconds": generated - loaded,
        "save_seconds": saved - generated,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(args.device),
        "raw_tensor_shape": list(video.shape),
        "raw_tensor_dtype": str(video.dtype),
        "frame_sha256": frame_hashes,
    }
    if latent_metadata is not None:
        metadata["vae_input_latent"] = latent_metadata
    if trajectory_metadata is not None:
        metadata["scheduler_latent_trajectory"] = trajectory_metadata
    (args.output / "reference.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
