# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build-time conversion of NVIDIA recsys-examples HSTU checkpoints.

The source topology must be supplied explicitly. PyTorch is needed only to read
the upstream checkpoint; the exported bundle uses safetensors and native TRT.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_REPOSITORY = "https://github.com/NVIDIA/recsys-examples"
REFERENCE_REVISION = "97062d97eef53115105063801e35184e36186df5"
_EMBEDDING_PREFIX = "_embedding_collection._data_parallel_embedding_collection.embeddings."


class _DiscardedMetadata:
    """Inert representation of metadata that is never used as model weights."""

    _discarded_checkpoint_state = True

    def __init__(self, *args):
        self.serialized_arguments = args

    def __setstate__(self, state):
        self.serialized_state = state


def _load_pytorch_checkpoint(path: Path) -> Mapping[str, Any]:
    """Read tensors safely without creating source distributed runtime objects.

    DynamicEmb emits dummy ShardedTensor objects even with world_size=1. Their
    actual values live in the separate binary table dump. TransformerEngine
    also emits byte-stream extra state that is irrelevant to inference weights.
    Exact known globals map to inert local holders; unknown globals still fail
    PyTorch's weights-only loader. No distributed process group is needed.
    """
    import torch

    class DiscardedShardedTensor(torch.Tensor):
        _discarded_checkpoint_state = True

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            raise ValueError("discarded distributed checkpoint metadata cannot be used as weights")

        def __setstate__(self, state):
            self.serialized_state = state

    def process_group_metadata_only(value, attribute):
        # Pickle represents the nested ShardedTensor.ProcessGroupState class
        # through getattr. Never permit general attribute access from a file.
        if value is DiscardedShardedTensor and attribute == "ProcessGroupState":
            return _DiscardedMetadata
        raise ValueError("unsupported attribute lookup in HSTU checkpoint metadata")

    aliases = [
        (DiscardedShardedTensor, "torch.distributed._shard.sharded_tensor.api.ShardedTensor"),
        (process_group_metadata_only, "builtins.getattr"),
    ]
    for name in (
        "torch.distributed._shard.sharded_tensor.shard.Shard",
        "torch.distributed._shard.metadata.ShardMetadata",
        "torch.distributed.remote_device._remote_device",
        "torch.distributed._shard.sharded_tensor.metadata.TensorProperties",
        "torch.distributed._shard.sharded_tensor.metadata.MEM_FORMAT_ENCODING",
        "torch.distributed._shard.sharded_tensor.metadata.ShardedTensorMetadata",
        "torch.distributed._shard.sharding_spec.api.EnumerableShardingSpec",
        "_io.BytesIO",
    ):
        aliases.append((_DiscardedMetadata, name))
    with torch.serialization.safe_globals(aliases):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError("HSTU checkpoint requires a model_state_dict mapping")
    return checkpoint


def _array(value: Any) -> np.ndarray:
    if getattr(value, "_discarded_checkpoint_state", False):
        raise ValueError("discarded distributed checkpoint metadata cannot be used as weights")
    if hasattr(value, "detach"):
        # NumPy cannot represent torch.bfloat16 directly.
        value = value.detach().cpu().float().numpy()
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("checkpoint contains non-finite weights")
    return np.ascontiguousarray(result)


def load_dynamic_table(directory: Path, name: str, hidden_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Read all upstream DynamicEmb key/value shards and sort by raw int64 ID."""
    directories = {path.parent for path in directory.rglob(f"{name}_emb_keys.rank_*.world_size_*")}
    if len(directories) != 1:
        raise ValueError(f"expected one DynamicEmb table directory for {name}, found {len(directories)}")
    table_dir = directories.pop()
    metadata_path = table_dir / f"{name}_opt_args.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if int(metadata.get("embedding_dim", hidden_size)) != hidden_size:
        raise ValueError(f"DynamicEmb dimension mismatch for {name}")
    dtype_name = str(metadata.get("embedding_dtype", "float32")).removeprefix("torch.")
    dtypes = {"float32": np.dtype("<f4"), "float16": np.dtype("<f2"), "bfloat16": np.dtype("<u2")}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported DynamicEmb dtype {dtype_name!r} for {name}")
    key_parts, value_parts = [], []
    shards: dict[int, int] = {}
    for key_path in sorted(table_dir.glob(f"{name}_emb_keys.rank_*.world_size_*")):
        match = re.fullmatch(re.escape(name) + r"_emb_keys.rank_(\d+).world_size_(\d+)", key_path.name)
        if match is None:
            raise ValueError(f"invalid DynamicEmb shard name: {key_path.name}")
        rank, world_size = map(int, match.groups())
        if rank in shards or rank >= world_size or world_size < 1:
            raise ValueError(f"invalid or duplicate DynamicEmb shard: {key_path.name}")
        shards[rank] = world_size
        value_path = key_path.with_name(key_path.name.replace("_emb_keys.rank_", "_emb_values.rank_"))
        if key_path.stat().st_size % 8:
            raise ValueError(f"truncated DynamicEmb keys for {name}")
        keys = np.fromfile(key_path, dtype="<i8")
        expected_bytes = keys.size * hidden_size * dtypes[dtype_name].itemsize
        if not value_path.exists() or value_path.stat().st_size != expected_bytes:
            raise ValueError(f"DynamicEmb value count mismatch for {name}")
        values = np.fromfile(value_path, dtype=dtypes[dtype_name])
        if dtype_name == "bfloat16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        key_parts.append(keys)
        value_parts.append(values.astype(np.float32).reshape(-1, hidden_size))
    world_sizes = set(shards.values())
    if len(world_sizes) != 1 or set(shards) != set(range(next(iter(world_sizes)))):
        raise ValueError(f"incomplete or mixed DynamicEmb shard set for {name}")
    keys = np.concatenate(key_parts)
    values = np.concatenate(value_parts)
    order = np.argsort(keys)
    keys, values = keys[order], values[order]
    if not keys.size or np.any(keys[1:] == keys[:-1]):
        raise ValueError(f"empty or duplicate DynamicEmb keys for {name}")
    return np.ascontiguousarray(keys), _array(values)


def convert_state_dict(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    source_layout: str,
    checkpoint_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Convert one unsharded dense state dict into canonical head-major weights.

    ``fused`` uses transposed type-major UVQK matrices, ``paged`` uses
    torch.nn.Linear type-major matrices, and ``native`` uses head-major matrices.
    Dense tensor-parallel shards must be consolidated before calling this API.
    """
    if source_layout not in {"fused", "native", "paged"}:
        raise ValueError("source_layout must be fused, native, or paged")
    normalized = {}
    for key, value in state.items():
        # InferenceRankingGR wraps these two modules, whereas training RankingGR
        # checkpoints contain their parameters directly on the model.
        name = key.removeprefix("dense_module.").removeprefix("sparse_module.")
        if name in normalized:
            raise ValueError(f"duplicate checkpoint parameter after removing module prefix: {name}")
        normalized[name] = value
    state = normalized
    d, h, a = (int(config[key]) for key in ("hidden_size", "num_heads", "head_dim"))
    used: set[str] = set()
    result: dict[str, np.ndarray] = {}

    def take(destination: str, names: list[str], shape: tuple[int, ...], *, transpose=False, uvqk=False):
        matches = [name for name in names if name in state]
        if len(matches) != 1:
            raise ValueError(f"expected one source for {destination}; found {matches}")
        key = matches[0]
        value = _array(state[key])
        used.add(key)
        if transpose:
            value = value.T
        if value.shape != shape:
            raise ValueError(f"{key} has shape {value.shape}; expected {shape} after transpose={transpose}")
        if uvqk and source_layout != "native":
            # [4, heads, head_dim, ...] -> [heads, 4, head_dim, ...].
            value = value.reshape(4, h, a, *shape[1:]).swapaxes(0, 1).reshape(shape)
        result[destination] = np.ascontiguousarray(value)

    for index in range(int(config["num_layers"])):
        src = f"_hstu_block._attention_layers.{index}."
        dst = f"blocks.{index}."
        for norm, source, width, learned in (
            ("input_norm", "_input_layernorm", d, "learnable_input_layernorm"),
            ("output_norm", "_output_layernorm", h * a, "learnable_output_layernorm"),
        ):
            if config.get(learned, True):
                for suffix in ("weight", "bias"):
                    names = [src + source + "_" + suffix]
                    if norm == "output_norm" and source_layout == "native":
                        names.append(src + "_output_ln_dropout_mul." + suffix)
                    take(dst + norm + "." + suffix, names, (width,))
        take(dst + "uvqk.weight", [src + ("_linear_uvqk_weight" if source_layout == "fused" else "_linear_uvqk.weight")], (4 * h * a, d), transpose=source_layout == "fused", uvqk=True)
        if config.get("add_uvqk_bias", True):
            take(dst + "uvqk.bias", [src + ("_linear_uvqk_bias" if source_layout == "fused" else "_linear_uvqk.bias")], (4 * h * a,), uvqk=True)
        take(dst + "proj.weight", [src + ("_linear_proj_weight" if source_layout == "fused" else "_linear_proj.weight")], (d, h * a), transpose=source_layout == "fused")

    pre = "_hstu_block._preprocessor._positional_encoder."
    if int(config.get("position_buckets", 0)):
        take("position.weight", [pre + "_position_embeddings_weight"], (int(config["position_buckets"]), d))
    if int(config.get("time_buckets", 0)):
        take("time.weight", [pre + "_timestamp_embeddings_weight"], (int(config["time_buckets"]) + 1, d))
    width = d
    for index, output_width in enumerate(config.get("prediction_head", [])):
        take(f"head.{index}.weight", [f"_mlp._mlp.{2 * index}.weight"], (int(output_width), width))
        if config.get("prediction_bias", True):
            take(f"head.{index}.bias", [f"_mlp._mlp.{2 * index}.bias"], (int(output_width),))
        width = int(output_width)

    tables = {table["name"]: table for table in config["embedding_tables"]}
    # TorchRec can pack multiple static tables into one flattened state tensor.
    for key in state:
        if not key.startswith(_EMBEDDING_PREFIX):
            continue
        name_field = key.removeprefix(_EMBEDDING_PREFIX)
        if not name_field.endswith("_weights"):
            continue
        names = name_field.removesuffix("_weights").split("/")
        if any(name not in tables for name in names):
            raise ValueError(f"static checkpoint table missing from config: {names}")
        values = _array(state[key])
        rows = sum(int(tables[name]["num_embeddings"]) for name in names)
        if values.size != rows * d:
            raise ValueError(f"packed embedding size mismatch for {key}")
        offset = 0
        for name in names:
            count = int(tables[name]["num_embeddings"])
            result[f"embeddings.{name}.weight"] = np.ascontiguousarray(values.reshape(rows, d)[offset : offset + count])
            offset += count
        used.add(key)
    for name, table in tables.items():
        target = f"embeddings.{name}.weight"
        if target in result:
            continue
        aliases = [f"_embedding_collection.embeddings.{name}.weight", f"_static_embedding_collection.embeddings.{name}.weight", _EMBEDDING_PREFIX + name + ".weight"]
        if any(key in state for key in aliases):
            take(target, aliases, (int(table["num_embeddings"]), d))
        elif checkpoint_dir is not None:
            keys, values = load_dynamic_table(checkpoint_dir / "dynamicemb_module", name, d)
            result[target] = values
            result[f"embeddings.{name}.keys"] = keys
        else:
            raise ValueError(f"missing embedding table {name}; provide checkpoint_dir for DynamicEmb tables")
    ignored = {key for key in state if "_model_parallel_embedding_collection" in key or key.endswith("_extra_state") or key == "_dynamic_embedding_collection._embedding_tables._empty_tensor"}
    unexpected = sorted(set(state) - used - ignored)
    if unexpected:
        raise ValueError(f"unconverted checkpoint state (check topology/layout): {unexpected}")
    return result


def export_checkpoint(checkpoint_dir: str | Path, config_path: str | Path, output_dir: str | Path, *, source_layout: str) -> Path:
    """Write a portable config + safetensors bundle from an upstream checkpoint."""
    from safetensors.numpy import save_file
    from .config import parse_config

    checkpoint_dir, output_dir = Path(checkpoint_dir), Path(output_dir)
    config = parse_config(json.loads(Path(config_path).read_text()))
    state_path = checkpoint_dir / "torch_module" / "model.0.pth"
    shards = list((checkpoint_dir / "torch_module").glob("model.*.pth"))
    if len(shards) != 1:
        raise ValueError("expected one consolidated dense checkpoint (torch_module/model.0.pth)")
    state = _load_pytorch_checkpoint(state_path)
    weights = convert_state_dict(state["model_state_dict"], config, source_layout=source_layout, checkpoint_dir=checkpoint_dir)
    for table in config["embedding_tables"]:
        table["num_embeddings"] = int(weights[f"embeddings.{table['name']}.weight"].shape[0])
    config["model_type"] = "hstu"
    config["schema_version"] = 1
    config = parse_config(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "conversion.json").write_text(json.dumps({"source_repository": REFERENCE_REPOSITORY, "reference_revision": REFERENCE_REVISION, "source_layout": source_layout, "checkpoint": str(state_path.resolve()), "weights": "user-supplied upstream checkpoint"}, indent=2) + "\n")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config", required=True, help="explicit TRT MC HSTU topology JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-layout", required=True, choices=("fused", "native", "paged"))
    args = parser.parse_args(argv)
    export_checkpoint(args.checkpoint_dir, args.config, args.output_dir, source_layout=args.source_layout)


if __name__ == "__main__":
    main()
