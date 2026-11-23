# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep Gemma's 0.12 checkpoint-backed per-layer embedding in the MC bundle."""

import json
from pathlib import Path

from safetensors import safe_open


def package_source(config_path: Path, source: Path, checkpoint: Path, *, ple: bool) -> None:
    """Copy the one referenced table without loading its multi-GB storage into RAM."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("checkpoint_weight_bindings"):
        raise ValueError("Gemma Edge ONNX requires self-contained engine weights")
    if bool(config.get("ple_pool")) != ple:
        raise ValueError("Gemma Edge per-layer embedding metadata differs from the checkpoint")
    if ple:
        pool = config["ple_pool"]
        record = pool["source"]
        key = record["tensor_key"]
        index = source / "model.safetensors.index.json"
        name = (json.loads(index.read_text(encoding="utf-8"))["weight_map"][key]
                if index.is_file() else "model.safetensors")
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Invalid Gemma per-layer embedding shard")
        path = source / name
        with safe_open(path, framework="numpy") as tensors, path.open("rb") as src:
            view = tensors.get_slice(key)
            shape = [pool["row_count"], pool["row_width"]]
            size = record["bytes"]
            if view.get_shape() != shape or view.get_dtype() != record["dtype"]:
                raise ValueError("Gemma per-layer embedding geometry or dtype mismatch")
            length = int.from_bytes(src.read(8), "little")
            start, stop = json.loads(src.read(length))[key]["data_offsets"]
            if stop - start != size or size <= 0:
                raise ValueError("Invalid Gemma per-layer embedding byte range")
            header = json.dumps({key: {"dtype": record["dtype"], "shape": shape,
                                       "data_offsets": [0, size]}}, separators=(",", ":")).encode()
            header += b" " * (-len(header) % 8)
            src.seek(8 + length + start)
            with (checkpoint / "model.safetensors").open("xb") as dst:
                dst.write(len(header).to_bytes(8, "little"))
                dst.write(header)
                while size:
                    chunk = src.read(min(size, 8 * 1024 * 1024))
                    if not chunk:
                        raise ValueError("Incomplete Gemma per-layer embedding")
                    dst.write(chunk)
                    size -= len(chunk)
        # The released runtime now resolves the tensor by key in the packaged checkpoint.
        # Byte samples, scales and shape from the original export remain unchanged.
        record.pop("location", None)
    config.pop("checkpoint_dir", None)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
