# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the native Model Connect repository for the supplied Triton image."""

import argparse
import json
import math
from pathlib import Path
import shutil
from .generate_repository import runtime_metadata


def generate(bundle, repository, name, *, parameters=None, instances=1):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be one directory component")
    if instances < 1:
        raise ValueError("instances must be positive")
    parameters = dict(parameters or {})
    unknown = set(parameters) - {"kv_cache_size_bytes", "kv_cache_memory_ratio"}
    if unknown:
        raise ValueError("native image has no supported parameter: " + ", ".join(sorted(unknown)))
    if len(parameters) > 1:
        raise ValueError("KV byte cap and memory ratio are mutually exclusive")
    if "kv_cache_size_bytes" in parameters:
        value = parameters["kv_cache_size_bytes"]
        if type(value) is not int or not 0 < value < 2**64:
            raise ValueError("KV bytes must be a positive uint64 integer")
    if "kv_cache_memory_ratio" in parameters:
        value = parameters["kv_cache_memory_ratio"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError("KV ratio must be finite and in (0,1]")
    runtime = runtime_metadata(bundle)
    fixed = runtime.get("batching") == "fixed"
    batch = runtime["max_batch_size"]
    target = Path(repository) / name
    if target.exists():
        raise FileExistsError(target)
    (target / "1").mkdir(parents=True)
    shutil.copyfile(bundle, target / "1/model.bundle")
    lines = [
        f"name: {json.dumps(name)}",
        'backend: "tensorrt"',
        'platform: "model_connect"',
        'default_model_filename: "model.bundle"',
        f"max_batch_size: {0 if fixed else batch}",
        f"instance_group [{{ kind: KIND_GPU count: {instances} gpus: [0] }}]",
    ]
    dtype = {"float32": "TYPE_FP32", "int64": "TYPE_INT64", "int32": "TYPE_INT32"}
    for field in ("input", "output"):
        lines.append(field + " [")
        for index, s in enumerate(runtime[field + "s"]):
            lines.append(
                "{ name: "
                + json.dumps(s["name"])
                + ", data_type: "
                + dtype[s["dtype"]]
                + ", dims: "
                + json.dumps([batch, *s["shape"][1:]] if fixed else s["shape"][1:])
                + " }"
                + ("," if index + 1 < len(runtime[field + "s"]) else "")
            )
        lines.append("]")
    for key, value in parameters.items():
        lines.append(
            "parameters { key: "
            + json.dumps(key)
            + " value { string_value: "
            + json.dumps(str(value))
            + " } }"
        )
    (target / "config.pbtxt").write_text("\n".join(lines) + "\n")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--name", default="hstu_research")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--kv-cache-size-bytes", type=int)
    g.add_argument("--kv-cache-memory-ratio", type=float)
    a = p.parse_args()
    params = {}
    if a.kv_cache_size_bytes is not None:
        if a.kv_cache_size_bytes <= 0:
            p.error("KV bytes must be positive")
        params["kv_cache_size_bytes"] = a.kv_cache_size_bytes
    if a.kv_cache_memory_ratio is not None:
        import math

        if not math.isfinite(a.kv_cache_memory_ratio) or not 0 < a.kv_cache_memory_ratio <= 1:
            p.error("ratio must be finite in (0,1]")
        params["kv_cache_memory_ratio"] = a.kv_cache_memory_ratio
    print(generate(a.bundle, a.repository, a.name, parameters=params))


if __name__ == "__main__":
    main()
