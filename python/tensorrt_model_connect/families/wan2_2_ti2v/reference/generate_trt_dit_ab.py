#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official 50-step pipeline with only the DiT replaced by TensorRT."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import tensorrt as trt
import torch
from PIL import Image


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage"
)


class FirstCallCaptured(RuntimeError):
    """Stop the official loop after saving its first real DiT inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrtDenoiser:
    def __init__(
        self,
        plan: Path,
        device: torch.device,
        time_embedding,
        *,
        native_model=None,
        compare_native_calls: int = 0,
        capture_first_call: Path | None = None,
        capture_call_index: int = 0,
        capture_only: bool = False,
    ):
        self.device = device
        self.time_embedding = time_embedding
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load {plan}")
        self.engine_path = str(plan.resolve())
        self.engine_bytes = plan.stat().st_size
        self.engine_sha256 = _sha256(plan)
        self.engine_device_memory_bytes = int(self.engine.device_memory_size_v2)
        self.engine_io = [
            {
                "name": self.engine.get_tensor_name(index),
                "mode": str(self.engine.get_tensor_mode(self.engine.get_tensor_name(index))),
                "dtype": str(self.engine.get_tensor_dtype(self.engine.get_tensor_name(index))),
                "shape": list(self.engine.get_tensor_shape(self.engine.get_tensor_name(index))),
            }
            for index in range(self.engine.num_io_tensors)
        ]
        self.context = self.engine.create_execution_context()
        self.calls = 0
        self.elapsed_seconds = 0.0
        self.native_model = native_model
        self.compare_native_calls = compare_native_calls
        self.native_comparisons = []
        self.capture_first_call = capture_first_call
        self.capture_call_index = capture_call_index
        self.capture_only = capture_only

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def __call__(self, x, t, context, seq_len, y=None):
        if y is not None:
            raise ValueError("The qualification path is text-to-video only")
        if self.calls == self.capture_call_index and self.capture_first_call is not None:
            self.capture_first_call.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "call": self.calls,
                    "latent": x[0].detach().cpu(),
                    "timestep": t.detach().cpu(),
                    "context": context[0].detach().cpu(),
                    "seq_len": int(seq_len),
                    "latent_dtype": str(x[0].dtype),
                    "timestep_dtype": str(t.dtype),
                    "context_dtype": str(context[0].dtype),
                },
                self.capture_first_call,
            )
            if self.capture_only:
                raise FirstCallCaptured(str(self.capture_first_call))
        native_output = None
        if self.native_model is not None and self.calls < self.compare_native_calls:
            native_output = self.native_model(x, t=t, context=context, seq_len=seq_len, y=y)[
                0
            ].float()
        latent = x[0].unsqueeze(0).float().contiguous()
        if latent.shape[2:] != (31, 44, 80):
            raise ValueError(f"Unexpected latent shape {tuple(latent.shape)}")
        if seq_len != 27280:
            raise ValueError(f"Unexpected sequence length {seq_len}")
        timestep = t.reshape(-1)[0:1]
        # WanModel.forward computes the sinusoidal embedding inside its nested
        # FP32 autocast-disabled region.  This wrapper itself is called from
        # the pipeline's outer BF16 autocast context, so explicitly disable it
        # here; converting BF16-rounded features to float afterwards is too
        # late and measurably changes all 50 denoising steps.
        with torch.amp.autocast("cuda", enabled=False):
            time_features = self.time_embedding(256, timestep).float().contiguous()
        text = torch.zeros(1, 512, 4096, device=self.device, dtype=torch.float32)
        text[0, : context[0].shape[0]] = context[0].float()
        output = torch.empty_like(latent)
        for name, tensor in (
            ("latents", latent),
            ("time_features", time_features),
            ("encoder_hidden_states", text),
            ("noise_prediction", output),
        ):
            self.context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream(self.device).cuda_stream
        started = time.perf_counter()
        # The family-owned CUDA plugins reproduce source dtype boundaries
        # explicitly.  Disable the surrounding qualification pipeline's
        # autocast while TensorRT dispatches them.
        with torch.amp.autocast("cuda", enabled=False):
            if not self.context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("TensorRT denoiser execution failed")
        torch.cuda.synchronize(self.device)
        self.elapsed_seconds += time.perf_counter() - started
        if native_output is not None:
            delta = output[0] - native_output
            self.native_comparisons.append(
                {
                    "call": self.calls,
                    "timestep_min": float(t.min()),
                    "timestep_max": float(t.max()),
                    "max_abs_error": float(delta.abs().max()),
                    "mean_abs_error": float(delta.abs().mean()),
                    "rmse": float(delta.square().mean().sqrt()),
                    "cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(
                            output[0].flatten().double(),
                            native_output.flatten().double(),
                            dim=0,
                        )
                    ),
                }
            )
        self.calls += 1
        return [output[0]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--native-plugin", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-latent",
        type=Path,
        help="Official normalized latent captured immediately before VAE decode",
    )
    parser.add_argument(
        "--reference-latent-trajectory",
        type=Path,
        help="Official 50-step scheduler-output trajectory for drift localization",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compare-native-calls", type=int, default=0)
    parser.add_argument("--capture-first-call", type=Path)
    parser.add_argument("--capture-call-index", type=int, default=0)
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    if args.capture_call_index < 0:
        parser.error("--capture-call-index must be non-negative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for native_plugin in args.native_plugin:
        ctypes.CDLL(str(native_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.configs import ti2v_5B  # pylint: disable=import-outside-toplevel
    from wan.modules.model import sinusoidal_embedding_1d  # pylint: disable=import-outside-toplevel
    import wan.textimage2video as textimage2video  # pylint: disable=import-outside-toplevel
    from wan.textimage2video import WanTI2V  # pylint: disable=import-outside-toplevel
    from wan.utils.utils import save_video  # pylint: disable=import-outside-toplevel

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
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
    native_model = pipeline.model
    if args.compare_native_calls:
        native_model.to(device)
    pipeline.model = TrtDenoiser(
        args.engine,
        device,
        sinusoidal_embedding_1d,
        native_model=native_model if args.compare_native_calls else None,
        compare_native_calls=args.compare_native_calls,
        capture_first_call=args.capture_first_call,
        capture_call_index=args.capture_call_index,
        capture_only=args.capture_only,
    )
    captured_latents: list[torch.Tensor] = []
    official_decode = pipeline.vae.decode

    def capture_decode(zs: list[torch.Tensor]):
        captured_latents.extend(latent.detach().cpu() for latent in zs)
        return official_decode(zs)

    pipeline.vae.decode = capture_decode
    captured_trajectory: list[torch.Tensor] = []
    official_scheduler_step = textimage2video.FlowUniPCMultistepScheduler.step

    def capture_scheduler_step(scheduler, *step_args, **step_kwargs):
        result = official_scheduler_step(scheduler, *step_args, **step_kwargs)
        captured_trajectory.append(result[0].detach().squeeze(0).float().cpu())
        return result

    textimage2video.FlowUniPCMultistepScheduler.step = capture_scheduler_step
    if not args.compare_native_calls:
        del native_model
        gc.collect()
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
    except FirstCallCaptured as captured:
        print(json.dumps({"captured_first_call": str(captured)}, indent=2))
        return 0
    finally:
        textimage2video.FlowUniPCMultistepScheduler.step = official_scheduler_step
    torch.cuda.synchronize(device)
    generated = time.perf_counter()

    video_cpu = video.float().cpu()
    torch.save(video_cpu, args.output / "video_fp32.pt")
    if len(captured_latents) != 1:
        raise RuntimeError(
            f"Expected one normalized latent passed to VAE decode, captured {len(captured_latents)}"
        )
    final_latent = captured_latents[0].float()
    latent_path = args.output / "latent_fp32.pt"
    torch.save(final_latent, latent_path)
    if len(captured_trajectory) != 50:
        raise RuntimeError(
            f"Expected 50 TRT scheduler outputs, captured {len(captured_trajectory)}"
        )
    latent_trajectory = torch.stack(captured_trajectory)
    trajectory_path = args.output / "latent_trajectory_fp32.pt"
    torch.save(latent_trajectory, trajectory_path)
    uint8_frames = (
        ((video_cpu.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).numpy()
    )
    frames_dir = args.output / "frames"
    frames_dir.mkdir()
    hashes = []
    for index, frame in enumerate(uint8_frames):
        path = frames_dir / f"frame_{index:04d}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        hashes.append(_sha256(path))
    save_video(
        video_cpu[None],
        save_file=str(args.output / "trt_dit.mp4"),
        fps=24,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    reference = torch.load(args.reference, map_location="cpu", weights_only=True).float()
    delta = video_cpu - reference
    ref_flat = reference.flatten().double()
    got_flat = video_cpu.flatten().double()
    per_frame_cosine = [
        float(
            torch.nn.functional.cosine_similarity(
                reference[:, index].flatten().double(),
                video_cpu[:, index].flatten().double(),
                dim=0,
            )
        )
        for index in range(reference.shape[1])
    ]
    latent_metrics = None
    if args.reference_latent is not None:
        reference_latent = torch.load(
            args.reference_latent, map_location="cpu", weights_only=True
        ).float()
        if tuple(reference_latent.shape) != tuple(final_latent.shape):
            raise ValueError(
                f"Reference latent shape {tuple(reference_latent.shape)} != "
                f"TRT latent shape {tuple(final_latent.shape)}"
            )
        latent_delta = final_latent - reference_latent
        latent_metrics = {
            "reference": str(args.reference_latent.resolve()),
            "output": str(latent_path.resolve()),
            "shape": list(final_latent.shape),
            "max_abs_error": float(latent_delta.abs().max()),
            "mean_abs_error": float(latent_delta.abs().mean()),
            "rmse": float(latent_delta.square().mean().sqrt()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    reference_latent.flatten().double(),
                    final_latent.flatten().double(),
                    dim=0,
                )
            ),
        }
    trajectory_metrics = None
    if args.reference_latent_trajectory is not None:
        reference_trajectory = torch.load(
            args.reference_latent_trajectory, map_location="cpu", weights_only=True
        ).float()
        if tuple(reference_trajectory.shape) != tuple(latent_trajectory.shape):
            raise ValueError(
                f"Reference trajectory shape {tuple(reference_trajectory.shape)} != "
                f"TRT trajectory shape {tuple(latent_trajectory.shape)}"
            )
        per_step = []
        for index in range(reference_trajectory.shape[0]):
            reference_step = reference_trajectory[index]
            actual_step = latent_trajectory[index]
            step_delta = actual_step - reference_step
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    reference_step.flatten().double(),
                    actual_step.flatten().double(),
                    dim=0,
                )
            )
            per_step.append(
                {
                    "step": index,
                    "max_abs_error": float(step_delta.abs().max()),
                    "mean_abs_error": float(step_delta.abs().mean()),
                    "rmse": float(step_delta.square().mean().sqrt()),
                    "cosine_similarity": cosine,
                }
            )
        trajectory_metrics = {
            "reference": str(args.reference_latent_trajectory.resolve()),
            "output": str(trajectory_path.resolve()),
            "shape": list(latent_trajectory.shape),
            "min_step_cosine": min(item["cosine_similarity"] for item in per_step),
            "mean_step_cosine": sum(item["cosine_similarity"] for item in per_step) / len(per_step),
            "first_step_below_0_998": next(
                (item["step"] for item in per_step if item["cosine_similarity"] < 0.998),
                None,
            ),
            "per_step": per_step,
        }
    metadata = {
        "kind": "wan2_2_ti2v_5b_trt_dit_official_pipeline_ab",
        "prompt": args.prompt,
        "seed": args.seed,
        "width": 1280,
        "height": 704,
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "flow_shift": 5.0,
        "engine": str(args.engine.resolve()),
        "engine_bytes": pipeline.model.engine_bytes,
        "engine_sha256": pipeline.model.engine_sha256,
        "engine_device_memory_bytes": pipeline.model.engine_device_memory_bytes,
        "engine_io": pipeline.model.engine_io,
        "load_seconds": loaded - started,
        "generation_seconds": generated - loaded,
        "denoiser_calls": pipeline.model.calls,
        "denoiser_seconds": pipeline.model.elapsed_seconds,
        "native_call_comparisons": pipeline.model.native_comparisons,
        "final_latent_metrics": latent_metrics,
        "latent_trajectory_metrics": trajectory_metrics,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "metrics": {
            "max_abs_error": float(delta.abs().max()),
            "mean_abs_error": float(delta.abs().mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(ref_flat, got_flat, dim=0)
            ),
            "min_frame_cosine": min(per_frame_cosine),
            "mean_frame_cosine": sum(per_frame_cosine) / len(per_frame_cosine),
            "per_frame_cosine": per_frame_cosine,
        },
        "frame_sha256": hashes,
    }
    (args.output / "comparison.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
