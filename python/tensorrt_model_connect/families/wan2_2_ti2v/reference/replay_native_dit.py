# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay native Wan2.2 DiT trace inputs through TensorRT's Python API."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


LATENT_SHAPE = (1, 48, 31, 44, 80)
CONTEXT_SHAPE = (1, 512, 4096)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--native-plugin", type=Path, action="append", required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def _raw(path: Path, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(f"Unexpected tensor size: {path}")
    return torch.from_numpy(values.copy()).reshape(shape).to(device)


def _compare(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    mismatch = actual.view(torch.int32) != reference.view(torch.int32)
    delta = actual.double() - reference.double()
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
    for plugin in args.native_plugin:
        ctypes.CDLL(str(plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.model import sinusoidal_embedding_1d  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda")
    trace_dir = args.trace_dir.resolve()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.resolve().read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create TensorRT execution context")

    tensors = {
        "latents": torch.empty(LATENT_SHAPE, device=device, dtype=torch.float32),
        "time_features": torch.empty((1, 256), device=device, dtype=torch.float32),
        "encoder_hidden_states": torch.empty(CONTEXT_SHAPE, device=device, dtype=torch.float32),
        "noise_prediction": torch.empty(LATENT_SHAPE, device=device, dtype=torch.float32),
    }
    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())
    prompt_context = _raw(trace_dir / "prompt_context.f32", CONTEXT_SHAPE, device)
    negative_context = _raw(trace_dir / "negative_context.f32", CONTEXT_SHAPE, device)
    timesteps = np.fromfile(trace_dir / "timesteps.i64", dtype=np.int64)
    records = []
    stream = torch.cuda.current_stream(device)

    for step in range(args.steps):
        latents = _raw(trace_dir / f"step_{step}_input_latents.f32", LATENT_SHAPE, device)
        tensors["latents"].copy_(latents)
        with torch.amp.autocast("cuda", enabled=False):
            time_features = sinusoidal_embedding_1d(
                256,
                torch.tensor([float(timesteps[step])], device=device),
            ).float()
        tensors["time_features"].copy_(time_features)
        step_record = {"step": step + 1, "timestep": int(timesteps[step])}
        replay_outputs = {}
        for label, source_context in (
            ("conditional", prompt_context),
            ("unconditional", negative_context),
        ):
            tensors["encoder_hidden_states"].copy_(source_context)
            if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT DiT replay failed")
            stream.synchronize()
            actual = tensors["noise_prediction"].clone()
            replay_outputs[label] = actual
            reference = _raw(trace_dir / f"step_{step}_{label}.f32", LATENT_SHAPE, device)
            step_record[f"{label}_vs_native"] = _compare(actual, reference)
        guided = replay_outputs["unconditional"] + 5.0 * (
            replay_outputs["conditional"] - replay_outputs["unconditional"]
        )
        native_guided = _raw(trace_dir / f"step_{step}_guided.f32", LATENT_SHAPE, device)
        step_record["guided_vs_native"] = _compare(guided, native_guided)
        records.append(step_record)
    print(json.dumps({"steps": records}, indent=2))


if __name__ == "__main__":
    main()
