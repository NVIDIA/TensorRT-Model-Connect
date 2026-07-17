# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and compare the full-shape official Wan2.2 CFG/UniPC contract.

The conditional and unconditional tensors are reconstructed from integer bit
patterns at every step.  The fixture therefore records only hashes, metrics,
and final tensors rather than 100 full DiT-output tensors.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import zlib

import numpy as np
import torch


LATENT_SHAPE = (1, 48, 31, 44, 80)
LATENT_COUNT = int(np.prod(LATENT_SHAPE))
NUM_STEPS = 50
GUIDANCE_SCALE = 5.0
FLOW_SHIFT = 5.0
PROBE_INDICES = (0, 1, 2, 3, 17, 1024, 65537, 1048575, LATENT_COUNT - 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cpp-replay",
        type=Path,
        help="C++ test executable supporting --stream-full; enables lockstep A/B",
    )
    parser.add_argument(
        "--stream-stages",
        action="store_true",
        help="Compare the corrector boundary as well as CFG and the final scheduler output",
    )
    parser.add_argument(
        "--autocast-bf16",
        action="store_true",
        help="Run the official scheduler under Wan2.2's outer BF16 autocast context",
    )
    return parser.parse_args()


def _load_scheduler_class():
    source_root = Path(os.environ.get("WAN22_OFFICIAL_SOURCE", "/workspace/Wan2.2-official"))
    module_path = source_root / "wan" / "utils" / "fm_solvers_unipc.py"
    spec = importlib.util.spec_from_file_location("wan22_official_unipc", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official UniPC module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


def _fixture_tensor(
    indices: torch.Tensor,
    step: int,
    multiplier: int,
    step_multiplier: int,
    salt: int,
) -> torch.Tensor:
    mixed = (indices * multiplier + (step + 1) * step_multiplier + salt) & 0xFFFFFFFF
    bits = ((mixed >> 31) << 31) | ((126 + ((mixed >> 30) & 1)) << 23) | (mixed & 0x007FFFFF)
    return bits.to(torch.int32).view(torch.float32).reshape(LATENT_SHAPE)


def _initial_sample(indices: torch.Tensor) -> torch.Tensor:
    return _fixture_tensor(indices, -1, 747796405, 0, 2891336453)


def _model_outputs(indices: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    conditional = _fixture_tensor(indices, step, 277803737, 1013904223, 0x12345678)
    unconditional = _fixture_tensor(indices, step, 1664525, 22695477, 0x9E3779B9)
    return conditional, unconditional


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError(f"C++ replay ended with {remaining} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _host_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _digest(raw: bytes) -> dict[str, str | int]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "crc32": f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.double()
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
    }


def _compare(official: torch.Tensor, cpp_raw: bytes, device: torch.device) -> dict:
    official_raw = _host_bytes(official)
    cpp_host = np.frombuffer(cpp_raw, dtype=np.float32)
    cpp = torch.from_numpy(cpp_host.copy()).reshape(LATENT_SHAPE).to(device)
    official_bits = official.view(torch.int32)
    cpp_bits = cpp.view(torch.int32)
    mismatch_mask = official_bits != cpp_bits
    mismatch_count = int(mismatch_mask.sum())
    first_mismatch = None
    if mismatch_count:
        first_index = int(mismatch_mask.flatten().nonzero()[0])
        first_mismatch = {
            "index": first_index,
            "official_bits": f"{int(official_bits.flatten()[first_index]) & 0xFFFFFFFF:08x}",
            "cpp_bits": f"{int(cpp_bits.flatten()[first_index]) & 0xFFFFFFFF:08x}",
        }
    delta = cpp.double() - official.double()
    official64 = official.double().flatten()
    cpp64 = cpp.double().flatten()
    cosine = torch.dot(official64, cpp64) / (
        torch.linalg.vector_norm(official64) * torch.linalg.vector_norm(cpp64)
    )
    official_flat = official.flatten()
    cpp_flat = cpp.flatten()
    return {
        "official": {**_digest(official_raw), **_stats(official)},
        "cpp": {**_digest(cpp_raw), **_stats(cpp)},
        "bitwise_mismatch_count": mismatch_count,
        "bitwise_match_fraction": (LATENT_COUNT - mismatch_count) / LATENT_COUNT,
        "first_mismatch": first_mismatch,
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine_similarity": float(cosine),
        "probes": [
            {
                "index": index,
                "official": float(official_flat[index]),
                "cpp": float(cpp_flat[index]),
                "abs_error": abs(float(cpp_flat[index]) - float(official_flat[index])),
            }
            for index in PROBE_INDICES
        ],
    }


def main() -> None:
    args = _parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The official fixture must run on CUDA")

    official_source = Path(
        os.environ.get("WAN22_OFFICIAL_SOURCE", "/workspace/Wan2.2-official")
    ).resolve()
    FlowUniPCMultistepScheduler = _load_scheduler_class()
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        solver_order=2,
        predict_x0=True,
        solver_type="bh2",
    )
    scheduler.set_timesteps(NUM_STEPS, device=device, shift=FLOW_SHIFT)
    indices = torch.arange(LATENT_COUNT, device=device, dtype=torch.int64)
    sample = _initial_sample(indices)
    initial_raw = _host_bytes(sample)

    replay = None
    if args.cpp_replay is not None:
        replay_mode = "--stream-stages" if args.stream_stages else "--stream-full"
        replay = subprocess.Popen(
            [str(args.cpp_replay.resolve()), replay_mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if replay.stdout is None or replay.stderr is None:
            raise RuntimeError("Could not capture the C++ replay streams")

    started = time.perf_counter()
    step_records = []
    final_cpp_raw = None
    for step, timestep in enumerate(scheduler.timesteps):
        conditional, unconditional = _model_outputs(indices, step)
        # Keep the source operation boundaries explicit: three eager CUDA
        # kernels for sub, mul, and add, matching Wan's expression.
        cfg_delta = conditional - unconditional
        cfg_scaled = cfg_delta * GUIDANCE_SCALE
        guided = unconditional + cfg_scaled
        step_context = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if args.autocast_bf16
            else nullcontext()
        )
        with step_context:
            if step == 0:
                corrected = sample
            else:
                corrected = scheduler.multistep_uni_c_bh_update(
                    this_model_output=scheduler.convert_model_output(guided, sample=sample),
                    last_sample=scheduler.last_sample,
                    this_sample=sample,
                    order=scheduler.this_order,
                )
            sample = scheduler.step(guided, timestep, sample, return_dict=False)[0]

        record = {
            "step": step + 1,
            "timestep": int(timestep),
        }
        if replay is None:
            record["cfg"] = {**_digest(_host_bytes(guided)), **_stats(guided)}
            record["latent"] = {**_digest(_host_bytes(sample)), **_stats(sample)}
        else:
            tensor_bytes = LATENT_COUNT * np.dtype(np.float32).itemsize
            cpp_cfg_raw = _read_exact(replay.stdout, tensor_bytes)
            cpp_corrected_raw = (
                _read_exact(replay.stdout, tensor_bytes) if args.stream_stages else None
            )
            cpp_latent_raw = _read_exact(replay.stdout, tensor_bytes)
            record["cfg"] = _compare(guided, cpp_cfg_raw, device)
            if cpp_corrected_raw is not None:
                record["corrected"] = _compare(corrected, cpp_corrected_raw, device)
            record["latent"] = _compare(sample, cpp_latent_raw, device)
            final_cpp_raw = cpp_latent_raw
        step_records.append(record)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if replay is not None:
        replay.stdout.close()
        stderr = replay.stderr.read().decode("utf-8", errors="replace")
        return_code = replay.wait()
        if return_code != 0:
            raise RuntimeError(f"C++ replay failed with code {return_code}: {stderr}")

    final_official_raw = _host_bytes(sample)
    (args.output / "official_final_latent_fp32.raw").write_bytes(final_official_raw)
    if final_cpp_raw is not None:
        (args.output / "cpp_final_latent_fp32.raw").write_bytes(final_cpp_raw)

    first_cfg_divergence = next(
        (
            record["step"]
            for record in step_records
            if record["cfg"].get("bitwise_mismatch_count", 0)
        ),
        None,
    )
    first_latent_divergence = next(
        (
            record["step"]
            for record in step_records
            if record["latent"].get("bitwise_mismatch_count", 0)
        ),
        None,
    )
    first_corrected_divergence = next(
        (
            record["step"]
            for record in step_records
            if record.get("corrected", {}).get("bitwise_mismatch_count", 0)
        ),
        None,
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "kind": "wan2_2_ti2v_full_shape_cfg_unipc_ab",
        "official_source": str(official_source),
        "official_source_revision": _git_revision(official_source),
        "fixture_script_sha256": _file_sha256(script_path),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "shape": list(LATENT_SHAPE),
        "count": LATENT_COUNT,
        "num_inference_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "flow_shift": FLOW_SHIFT,
        "autocast_bf16": args.autocast_bf16,
        "input_contract": {
            "encoding": "IEEE-754 bits from uint32 LCG",
            "initial": [747796405, 0, 2891336453],
            "conditional": [277803737, 1013904223, 0x12345678],
            "unconditional": [1664525, 22695477, 0x9E3779B9],
        },
        "initial": _digest(initial_raw),
        "timesteps": scheduler.timesteps.cpu().tolist(),
        "sigmas": scheduler.sigmas.tolist(),
        "cpp_replay": str(args.cpp_replay.resolve()) if args.cpp_replay else None,
        "cpp_replay_sha256": _file_sha256(args.cpp_replay.resolve()) if args.cpp_replay else None,
        "first_cfg_divergence_step": first_cfg_divergence,
        "first_corrected_divergence_step": first_corrected_divergence,
        "first_latent_divergence_step": first_latent_divergence,
        "elapsed_seconds": elapsed,
        "steps": step_records,
        "final_official": _digest(final_official_raw),
        "final_cpp": _digest(final_cpp_raw) if final_cpp_raw is not None else None,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "shape",
                    "first_cfg_divergence_step",
                    "first_corrected_divergence_step",
                    "first_latent_divergence_step",
                    "elapsed_seconds",
                    "final_official",
                    "final_cpp",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
