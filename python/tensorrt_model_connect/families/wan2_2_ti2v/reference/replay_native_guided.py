# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded native CFG tensors through the official UniPC scheduler."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


LATENT_SHAPE = (1, 48, 31, 44, 80)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--autocast-bf16", action="store_true")
    return parser.parse_args()


def _load_scheduler_class():
    source_root = Path(os.environ.get("WAN22_OFFICIAL_SOURCE", "/workspace/Wan2.2-official"))
    module_path = source_root / "wan" / "utils" / "fm_solvers_unipc.py"
    spec = importlib.util.spec_from_file_location("wan22_official_unipc_replay", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official UniPC source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


def _raw_tensor(path: Path, device: torch.device) -> torch.Tensor:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != int(np.prod(LATENT_SHAPE)):
        raise RuntimeError(f"Unexpected latent size: {path}")
    return torch.from_numpy(values.copy()).reshape(LATENT_SHAPE).to(device)


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    mismatch = left.view(torch.int32) != right.view(torch.int32)
    delta = left.double() - right.double()
    return {
        "bitwise_mismatch_count": int(mismatch.sum()),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
    }


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.steps > 50:
        raise ValueError("--steps must be in [1, 50]")
    device = torch.device("cuda")
    scheduler_class = _load_scheduler_class()
    scheduler = scheduler_class(
        num_train_timesteps=1000,
        shift=1,
        solver_order=2,
        predict_x0=True,
        solver_type="bh2",
    )
    scheduler.set_timesteps(50, device=device, shift=5.0)
    trace_dir = args.trace_dir.resolve()
    sample = _raw_tensor(trace_dir / "initial_latents.f32", device)
    trajectory = torch.load(
        args.reference.resolve(), map_location="cpu", weights_only=True, mmap=True
    )
    records = []
    autocast_context = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.autocast_bf16 else nullcontext()
    )
    with autocast_context:
        for step in range(args.steps):
            guided = _raw_tensor(trace_dir / f"step_{step}_guided.f32", device)
            sample = scheduler.step(
                guided,
                scheduler.timesteps[step],
                sample,
                return_dict=False,
            )[0]
            official_reference = trajectory[step].to(device)
            native = _raw_tensor(trace_dir / f"step_{step}_output_latents.f32", device)
            records.append(
                {
                    "step": step + 1,
                    "replay_vs_reference": _compare(sample, official_reference),
                    "replay_vs_native": _compare(sample, native),
                }
            )
    print(json.dumps({"steps": records}, indent=2))


if __name__ == "__main__":
    main()
