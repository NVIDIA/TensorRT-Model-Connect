# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated builder for one Qwen native-KV engine role."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading

from .config import ModelConfig


def write_weight_snapshot(directory: Path, weights: dict) -> None:
    """Stage the parent's converted arrays before launching either build."""
    import numpy as np

    directory.mkdir()
    for index, value in enumerate(weights.values()):
        np.save(directory / f"{index}.npy", value, allow_pickle=False)
    (directory / "names.json").write_text(json.dumps(list(weights)), encoding="utf-8")


def read_weight_snapshot(directory: Path) -> dict:
    """Map tensors read-only and restore the mapper's scalar dimensions."""
    import numpy as np

    weights = {}
    names = json.loads((directory / "names.json").read_text(encoding="utf-8"))
    for index, name in enumerate(names):
        value = np.load(directory / f"{index}.npy", mmap_mode="r", allow_pickle=False)
        weights[name] = value.item() if value.ndim == 0 else value
    return weights


@contextmanager
def _watch_parent(enabled: bool):
    """Stop an active isolated worker when its parent's private pipe closes."""
    if not enabled:
        yield
        return
    finished = threading.Event()

    def watch():
        try:
            # Raw reads avoid a buffered stdin lock during interpreter shutdown.
            while os.read(0, 4096):
                pass
        except OSError:
            pass
        if not finished.is_set():
            # Abrupt-parent cleanup runs at the next GIL release. Ordinary
            # failures/cancellation also terminate and join workers in model.py.
            os._exit(75)

    threading.Thread(target=watch, name="qwen-build-parent", daemon=True).start()
    try:
        yield
    finally:
        finished.set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--cuda-device", type=int, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--watch-parent", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    with _watch_parent(args.watch_parent):
        from cuda.bindings import runtime as cudart

        status, = cudart.cudaSetDevice(args.cuda_device)
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot select CUDA device {args.cuda_device} for Qwen build: {status}")
        from .model import _QwenModel
        from .weightless import weightless_build

        config = ModelConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        precision = config.raw["_resolved_build_precision"]
        if precision != "bf16":
            raise ValueError("Qwen isolated weightless workers require BF16")
        config.raw["_decoder_engine_role"] = args.role
        owner = _QwenModel()
        weights = read_weight_snapshot(args.weights)
        with weightless_build():
            plan = owner.build_engine(
                config, weights, args.max_sequence_length, precision=precision,
                verbose=args.verbose,
            )
        args.plan.write_bytes(plan)
        args.metadata.write_text(json.dumps({
            "role": args.role,
            "plan_bytes": len(plan),
            "native_kv_cache_metadata": config.raw.get("_native_kv_cache_metadata"),
        }), encoding="utf-8")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
