#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the serialized I/O and activation contract of a VAE step plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorrt_model_connect.trt_compat import trt

from tensorrt_model_connect.families.wan2_2_ti2v.vae_builder import (
    load_vae_cuda_plugin,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plugin = load_vae_cuda_plugin()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {args.engine}")
    tensors = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        tensors.append(
            {
                "index": index,
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    device_memory = (
        int(engine.device_memory_size_v2)
        if hasattr(engine, "device_memory_size_v2")
        else int(engine.device_memory_size)
    )
    report = {
        "kind": "wan2_2_ti2v_vae_step_engine_contract",
        "engine": str(args.engine.resolve()),
        "engine_bytes": args.engine.stat().st_size,
        "engine_sha256": _sha256(args.engine),
        "plugin": str(plugin.resolve()),
        "plugin_sha256": _sha256(plugin),
        "device_memory_bytes": device_memory,
        "num_aux_streams": int(engine.num_aux_streams),
        "io_tensor_count": engine.num_io_tensors,
        "inputs": [item for item in tensors if item["mode"].endswith("INPUT")],
        "outputs": [item for item in tensors if item["mode"].endswith("OUTPUT")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
