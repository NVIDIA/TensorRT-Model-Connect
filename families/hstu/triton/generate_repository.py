#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a Triton model entry from native bundle IO metadata."""

import argparse
import json
import math
from pathlib import Path
import shutil
import struct


def runtime_metadata(path):
    # Read only runtime.json; engine sections can be several gigabytes.
    with path.open("rb") as stream:
        if stream.read(8) != b"BUNDLE\x01\x00":
            raise ValueError("expected a native .bundle file")
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        section = header["sections"]["runtime.json"]
        stream.seek(16 + header_size + section["offset"])
        return json.loads(stream.read(section["length"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--name", default="hstu")
    parser.add_argument("--runtime-root", default="/src/build-local")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--kv-cache-memory-ratio", type=float)
    cache.add_argument("--kv-cache-size-bytes", type=int)
    parser.add_argument(
        "--history-cache",
        action="store_true",
        help="Generate an explicit-batch snapshot restore/scoring endpoint",
    )
    parser.add_argument("--kv-cache-max-entries", type=int)
    parser.add_argument("--kv-cache-ttl-seconds", type=float)
    parser.add_argument("--kv-cache-lease-seconds", type=float)
    args = parser.parse_args()
    if args.kv_cache_memory_ratio is not None and (
        not math.isfinite(args.kv_cache_memory_ratio) or not 0 < args.kv_cache_memory_ratio <= 1
    ):
        parser.error("--kv-cache-memory-ratio must be finite and in (0,1]")
    if args.kv_cache_size_bytes is not None and args.kv_cache_size_bytes <= 0:
        parser.error("--kv-cache-size-bytes must be positive")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a single model directory name")
    runtime = runtime_metadata(args.bundle)
    if args.history_cache:
        args.kv_cache_max_entries = (
            32 if args.kv_cache_max_entries is None else args.kv_cache_max_entries
        )
        args.kv_cache_ttl_seconds = (
            300 if args.kv_cache_ttl_seconds is None else args.kv_cache_ttl_seconds
        )
        if runtime.get("history_cache_schema") != 1:
            parser.error("bundle has no history-cache plans; rebuild with --history-cache")
        if args.kv_cache_size_bytes is None and args.kv_cache_memory_ratio is None:
            parser.error("--history-cache requires an explicit KV byte limit or memory ratio")
        lease = (
            min(30, args.kv_cache_ttl_seconds)
            if args.kv_cache_lease_seconds is None
            else args.kv_cache_lease_seconds
        )
        if (
            args.kv_cache_max_entries <= 0
            or args.kv_cache_max_entries > 1000000
            or not math.isfinite(args.kv_cache_ttl_seconds)
            or not 0 < args.kv_cache_ttl_seconds <= 31536000
            or not math.isfinite(lease)
            or not 0 <= lease <= args.kv_cache_ttl_seconds
        ):
            parser.error("invalid history-cache entry, TTL or lease limits")
    elif any(
        value is not None
        for value in (
            args.kv_cache_max_entries,
            args.kv_cache_ttl_seconds,
            args.kv_cache_lease_seconds,
        )
    ):
        parser.error("history-cache entry, TTL and lease options require --history-cache")
    directory = args.repository / args.name
    (directory / "1").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("model.py"), directory / "1/model.py")
    dtype = dict(float32="TYPE_FP32", int32="TYPE_INT32", int64="TYPE_INT64")
    lines = [
        f"name: {json.dumps(args.name)}",
        'backend: "python"',
        f"max_batch_size: {0 if args.history_cache else runtime['max_batch_size']}",
        "instance_group [{ kind: KIND_GPU count: 1 gpus: [0] }]",
    ]
    for field in ("input", "output"):
        specs = list(runtime[field + "s"])
        if args.history_cache:
            specs += [
                dict(
                    name="CACHE_CONTROL" if field == "input" else "CACHE_STATUS",
                    dtype="string",
                    shape=[1],
                ),
                dict(name="CACHE_SNAPSHOT", dtype="uint8", shape=[-1]),
            ]
        lines.append(field + " [")
        for index, spec in enumerate(specs):
            lines.append(
                "  { name: "
                + json.dumps(spec["name"])
                + ", data_type: "
                + dict(dtype, string="TYPE_STRING", uint8="TYPE_UINT8")[spec["dtype"]]
                + ", dims: "
                + json.dumps(spec["shape"] if args.history_cache else spec["shape"][1:])
                + (
                    " , optional: true"
                    if args.history_cache and field == "input" and spec["name"] != "CACHE_CONTROL"
                    else ""
                )
                + " }"
                + ("," if index + 1 < len(specs) else "")
            )
        lines.append("]")
    parameters = dict(bundle=str(args.bundle.resolve()), runtime_root=args.runtime_root)
    if args.history_cache:
        parameters.update(
            kv_cache_hydration_enabled="true",
            kv_cache_max_entries=str(args.kv_cache_max_entries),
            kv_cache_ttl_seconds=str(args.kv_cache_ttl_seconds),
            kv_cache_lease_seconds=str(lease),
            history_input_names=json.dumps([s["name"] for s in runtime["stages"][0]["inputs"]]),
            request_input_names=json.dumps([s["name"] for s in runtime["stages"][1]["inputs"]]),
        )
    for key in ("kv_cache_memory_ratio", "kv_cache_size_bytes"):
        value = getattr(args, key)
        if value is not None:
            parameters[key] = str(value)
    for key, value in parameters.items():
        lines.append(f'parameters {{ key: "{key}" value {{ string_value: {json.dumps(value)} }} }}')
    (directory / "config.pbtxt").write_text("\n".join(lines) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
