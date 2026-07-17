#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture every official Wan2.2 call-0 time-path boundary at full shape.

This is an evidence generator, not a production implementation.  It preserves
the official outer BF16 autocast and nested FP32 autocast scopes, then evaluates
the two Sequential modules one child at a time so the first divergent operator
can be identified without changing source semantics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-call", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_official_model(official_source: Path):
    root = official_source.resolve()
    sys.path.insert(0, str(root))
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(root / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(root / "wan" / "modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package
    from wan.modules.model import (  # pylint: disable=import-outside-toplevel
        WanModel,
        sinusoidal_embedding_1d,
    )

    return WanModel, sinusoidal_embedding_1d


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "numel": tensor.numel(),
    }


def save_tensor(output_dir: Path, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    path = output_dir / f"{name}.npy"
    array = tensor.detach().contiguous().cpu().numpy()
    np.save(path, array, allow_pickle=False)
    metadata = tensor_metadata(tensor)
    metadata.update(
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    )
    return metadata


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    WanModel, sinusoidal_embedding_1d = import_official_model(args.official_source)
    first_call = torch.load(args.first_call, map_location="cpu", weights_only=True)
    timestep_cpu = first_call["timestep"].to(dtype=torch.float32).contiguous()
    seq_len = int(first_call["seq_len"])
    if tuple(timestep_cpu.shape) != (1, seq_len):
        raise ValueError(
            f"Expected expanded call-0 timestep [1,{seq_len}], got {tuple(timestep_cpu.shape)}"
        )

    # Load on CPU, retain only the two source modules, then release the rest of
    # the 5B model before touching GPU3.
    model = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
    time_embedding = model.time_embedding
    time_projection = model.time_projection
    freq_dim = int(model.freq_dim)
    dim = int(model.dim)
    del model
    gc.collect()
    time_embedding.to(device)
    time_projection.to(device)

    linear1 = time_embedding[0]
    embedding_silu = time_embedding[1]
    linear2 = time_embedding[2]
    projection_silu = time_projection[0]
    projection_linear = time_projection[1]
    for name, module in {
        "linear1": linear1,
        "linear2": linear2,
        "projection_linear": projection_linear,
    }.items():
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"Official {name} is not nn.Linear")

    timestep = timestep_cpu.to(device)
    scope: dict[str, Any] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        scope["outer_enabled"] = torch.is_autocast_enabled("cuda")
        scope["outer_dtype"] = str(torch.get_autocast_dtype("cuda"))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            scope["inner_enabled"] = torch.is_autocast_enabled("cuda")
            scope["inner_dtype"] = str(torch.get_autocast_dtype("cuda"))
            sinusoidal_fp64 = sinusoidal_embedding_1d(freq_dim, timestep.flatten()).unflatten(
                0, (1, seq_len)
            )
            time_features = sinusoidal_fp64.float()
            time_linear1 = linear1(time_features)
            time_silu = embedding_silu(time_linear1)
            time_embed = linear2(time_silu)
            projection_silu_output = projection_silu(time_embed)
            time_projection_flat = projection_linear(projection_silu_output)
            time_projection_unflatten = time_projection_flat.unflatten(2, (6, dim))
    torch.cuda.synchronize(device)

    expected = {
        "sinusoidal_fp64": (1, seq_len, freq_dim),
        "time_features": (1, seq_len, freq_dim),
        "time_linear1": (1, seq_len, dim),
        "time_silu": (1, seq_len, dim),
        "time_embed": (1, seq_len, dim),
        "projection_silu": (1, seq_len, dim),
        "time_projection_flat": (1, seq_len, 6 * dim),
        "time_projection_unflatten": (1, seq_len, 6, dim),
    }
    values = {
        "sinusoidal_fp64": sinusoidal_fp64,
        "time_features": time_features,
        "time_linear1": time_linear1,
        "time_silu": time_silu,
        "time_embed": time_embed,
        "projection_silu": projection_silu_output,
        "time_projection_flat": time_projection_flat,
        "time_projection_unflatten": time_projection_unflatten,
    }
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise ValueError(f"{name} shape {tuple(values[name].shape)} != {shape}")
    for name in values:
        expected_dtype = torch.float64 if name == "sinusoidal_fp64" else torch.float32
        if values[name].dtype != expected_dtype:
            raise TypeError(f"{name} dtype {values[name].dtype} != {expected_dtype}")
    if time_projection_flat.data_ptr() != time_projection_unflatten.data_ptr():
        raise RuntimeError("Official time_projection.unflatten unexpectedly materialized a copy")

    manifest: dict[str, Any] = {
        "kind": "wan2_2_ti2v_official_call0_time_path",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "official_source": str(args.official_source.resolve()),
        "official_model_source_sha256": sha256_file(
            args.official_source.resolve() / "wan" / "modules" / "model.py"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "first_call": str(args.first_call.resolve()),
        "first_call_sha256": sha256_file(args.first_call.resolve()),
        "seq_len": seq_len,
        "freq_dim": freq_dim,
        "dim": dim,
        "autocast": scope,
        "parameters": {},
        "tensors": {},
        "unflatten_is_view": True,
    }
    for prefix, module in {
        "time_linear1": linear1,
        "time_linear2": linear2,
        "projection_linear": projection_linear,
    }.items():
        manifest["parameters"][f"{prefix}_weight"] = save_tensor(
            args.output_dir, f"{prefix}_weight", module.weight
        )
        manifest["parameters"][f"{prefix}_bias"] = save_tensor(
            args.output_dir, f"{prefix}_bias", module.bias
        )

    # The unflatten output aliases the flat output byte-for-byte, so store one
    # payload and record the view metadata instead of duplicating ~2 GiB.
    for name, tensor in values.items():
        if name == "time_projection_unflatten":
            manifest["tensors"][name] = {
                **tensor_metadata(tensor),
                "alias_of": "time_projection_flat",
                "sha256": manifest["tensors"]["time_projection_flat"]["sha256"],
            }
        else:
            manifest["tensors"][name] = save_tensor(args.output_dir, name, tensor)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
