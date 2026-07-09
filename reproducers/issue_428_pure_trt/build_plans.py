#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build issue #428 Gemma prefill/decode TensorRT plans without a .trtfb bundle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import tensorrt as trt


def build_role(plugin, config, weights, cache_length: int, role: str, verbose: bool) -> bytes:
    previous_role = config.raw.get("_decoder_engine_role")
    config.raw["_decoder_engine_role"] = role
    try:
        return plugin.build_engine(
            config,
            weights,
            cache_length,
            precision="fp16",
            verbose=verbose,
        )
    finally:
        if previous_role is None:
            config.raw.pop("_decoder_engine_role", None)
        else:
            config.raw["_decoder_engine_role"] = previous_role


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build pure TensorRT plans for the issue #428 reproducer"
    )
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-length", type=int, default=1741)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"gemma-2-2b-c{args.cache_length}"
    plan_paths = {
        "prefill": output_dir / f"{prefix}-prefill.plan",
        "decode": output_dir / f"{prefix}-decode.plan",
    }
    metadata_path = output_dir / f"{prefix}-metadata.json"
    if not args.force and metadata_path.exists() and all(path.exists() for path in plan_paths.values()):
        print(f"[pure-trt-build] reuse existing plans in {output_dir}", flush=True)
        print(f"[pure-trt-build] metadata={metadata_path}", flush=True)
        return 0

    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.engine_builder import _resolve_model
    from tensorrt_model_connect.families import find_plugin

    model_dir = _resolve_model(args.model)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None or plugin.name != "gemma":
        raise RuntimeError(
            f"expected Gemma plugin for {args.model}, got "
            f"{getattr(plugin, 'name', None)!r}"
        )

    print(
        f"[pure-trt-build] TensorRT={trt.__version__} model={args.model} "
        f"cache_length={args.cache_length} precision=fp16",
        flush=True,
    )
    print("[pure-trt-build] loading Gemma weights", flush=True)
    weights = plugin.load_weights(model_dir, config, precision="fp16")

    metadata = {
        "model": args.model,
        "resolved_model_dir": str(model_dir),
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "cache_length": args.cache_length,
        "precision": "fp16",
        "tensorrt_version": trt.__version__,
        "plans": {},
    }

    for role in ("prefill", "decode"):
        print(f"[pure-trt-build] building {role} plan", flush=True)
        started = time.monotonic()
        plan = build_role(
            plugin, config, weights, args.cache_length, role, args.verbose
        )
        elapsed = time.monotonic() - started
        path = plan_paths[role]
        path.write_bytes(plan)
        metadata["plans"][role] = {
            "path": str(path),
            "bytes": len(plan),
            "sha256": sha256_bytes(plan),
            "build_seconds": elapsed,
        }
        print(
            f"[pure-trt-build] {role} plan={path} bytes={len(plan)} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        del plan
        gc.collect()

    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[pure-trt-build] metadata={metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
