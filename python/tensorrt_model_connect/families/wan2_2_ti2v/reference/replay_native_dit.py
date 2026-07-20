# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay native Wan2.2 DiT trace inputs through TensorRT's Python API."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect.trt_compat import trt
import torch


LATENT_SHAPE = (1, 48, 31, 44, 80)
CONTEXT_SHAPE = (1, 512, 4096)
DEFAULT_MAX_RELATIVE_L2_ERROR = 0.01


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, str | int]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _trace_tree_identity(root: Path) -> dict[str, str | int]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    paths = sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    if not paths:
        raise ValueError(f"Wan2.2 DiT trace directory is empty: {resolved}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        relative_name = path.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(relative_name)
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(4 << 20):
                digest.update(chunk)
                total_bytes += len(chunk)
        digest.update(b"\0")
    return {
        "root": str(resolved),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "sha256_relative_names_and_contents": digest.hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--native-plugin", type=Path, action="append", required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.998)
    parser.add_argument(
        "--max-relative-l2-error",
        type=float,
        default=DEFAULT_MAX_RELATIVE_L2_ERROR,
        help="Maximum relative L2 error for every replayed output (default: %(default)s)",
    )
    return parser.parse_args()


def _raw(path: Path, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise RuntimeError(f"Unexpected tensor size: {path}")
    return torch.from_numpy(values.copy()).reshape(shape).to(device)


def _compare(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    mismatch = actual.view(torch.int32) != reference.view(torch.int32)
    delta = actual.double() - reference.double()
    actual_flat = actual.double().flatten()
    reference_flat = reference.double().flatten()
    reference_l2_norm = float(torch.linalg.vector_norm(reference_flat))
    actual_l2_norm = float(torch.linalg.vector_norm(actual_flat))
    delta_l2_norm = float(torch.linalg.vector_norm(delta.flatten()))
    relative_l2_error = (
        delta_l2_norm / reference_l2_norm
        if reference_l2_norm > 0.0
        else (0.0 if delta_l2_norm == 0.0 else math.inf)
    )
    if torch.count_nonzero(actual_flat) == 0 and torch.count_nonzero(reference_flat) == 0:
        cosine_similarity = 1.0
    else:
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(actual_flat, reference_flat, dim=0)
        )
    return {
        "bitwise_mismatch_count": int(mismatch.sum()),
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "reference_l2_norm": reference_l2_norm,
        "actual_l2_norm": actual_l2_norm,
        "delta_l2_norm": delta_l2_norm,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
    }


def _qualification(
    records: list[dict],
    min_cosine: float,
    max_relative_l2_error: float = DEFAULT_MAX_RELATIVE_L2_ERROR,
) -> dict:
    comparisons = [
        record[key]
        for record in records
        for key in ("conditional_vs_native", "unconditional_vs_native", "guided_vs_native")
    ]
    cosine_values = [item["cosine_similarity"] for item in comparisons]
    relative_l2_values = [item["relative_l2_error"] for item in comparisons]
    non_finite_cosine_count = sum(not math.isfinite(value) for value in cosine_values)
    non_finite_relative_l2_count = sum(not math.isfinite(value) for value in relative_l2_values)
    worst_cosine = min(cosine_values) if non_finite_cosine_count == 0 else None
    worst_relative_l2_error = max(relative_l2_values) if non_finite_relative_l2_count == 0 else None
    return {
        "min_cosine": min_cosine,
        "max_relative_l2_error": max_relative_l2_error,
        "comparisons_checked": len(comparisons),
        "non_finite_cosine_count": non_finite_cosine_count,
        "non_finite_relative_l2_count": non_finite_relative_l2_count,
        "worst_cosine_similarity": worst_cosine,
        "worst_relative_l2_error": worst_relative_l2_error,
        "passed": (
            worst_cosine is not None
            and worst_cosine >= min_cosine
            and worst_relative_l2_error is not None
            and worst_relative_l2_error <= max_relative_l2_error
        ),
    }


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.steps > 50:
        raise ValueError("--steps must be in [1, 50]")
    if not math.isfinite(args.min_cosine) or args.min_cosine < 0.0 or args.min_cosine > 1.0:
        raise ValueError("--min-cosine must be in [0, 1]")
    if (
        not math.isfinite(args.max_relative_l2_error)
        or args.max_relative_l2_error < 0.0
        or args.max_relative_l2_error > 1.0
    ):
        raise ValueError("--max-relative-l2-error must be in [0, 1]")
    native_plugins = [_file_identity(plugin) for plugin in args.native_plugin]
    plugin_handles = [
        ctypes.CDLL(plugin["path"], mode=ctypes.RTLD_GLOBAL) for plugin in native_plugins
    ]
    sys.path.insert(0, str(args.official_source.resolve()))
    from wan.modules.model import sinusoidal_embedding_1d  # pylint: disable=import-outside-toplevel

    device = torch.device("cuda")
    trace_dir = args.trace_dir.resolve()
    trace_tree = _trace_tree_identity(trace_dir)
    engine_path = args.engine.resolve()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    plan = engine_path.read_bytes()
    engine_sha256 = hashlib.sha256(plan).hexdigest()
    engine = runtime.deserialize_cuda_engine(plan)
    del plan
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
    report = {
        "kind": "wan2_2_ti2v_dit_trace_replay",
        "device": torch.cuda.get_device_name(device),
        "tensorrt_version": trt.__version__,
        "engine": str(engine_path),
        "engine_size_bytes": engine_path.stat().st_size,
        "engine_sha256": engine_sha256,
        "native_plugins": native_plugins,
        "trace_dir": str(trace_dir),
        "trace_tree": trace_tree,
        "steps": records,
        "qualification": _qualification(
            records,
            args.min_cosine,
            max_relative_l2_error=args.max_relative_l2_error,
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    assert plugin_handles
    if not report["qualification"]["passed"]:
        qualification = report["qualification"]
        if (
            qualification["non_finite_cosine_count"]
            or qualification["non_finite_relative_l2_count"]
        ):
            raise SystemExit(
                "Wan2.2 DiT trace replay failed accuracy gate: "
                f"non_finite_cosine={qualification['non_finite_cosine_count']}, "
                f"non_finite_relative_l2={qualification['non_finite_relative_l2_count']}"
            )
        raise SystemExit(
            "Wan2.2 DiT trace replay failed accuracy gate: "
            + json.dumps(qualification, sort_keys=True)
        )


if __name__ == "__main__":
    main()
