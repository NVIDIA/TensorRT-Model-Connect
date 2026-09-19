# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit inference configuration for converted NVIDIA HSTU checkpoints."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


_DEFAULTS = {
    "schema_version": 1,
    "model_type": "hstu",
    "mode": "ranking",
    "layer_norm_epsilon": 1e-5,
    "learnable_input_layernorm": True,
    "learnable_output_layernorm": True,
    "add_uvqk_bias": True,
    "residual": True,
    "is_causal": True,
    "scaling_seqlen": -1,
    "target_group_size": 1,
    "disable_contextual_mask": False,
    "position_buckets": 0,
    "time_buckets": 0,
    "prediction_head": [],
    "prediction_activation": "relu",
    "prediction_bias": True,
    "output_postprocessor": "l2",
    "output_norm_epsilon": 1e-6,
    "enable_history_cache": False,
    # Native selection follows this existing cache flag: paged when enabled,
    # genuinely uncached dense attention otherwise. Unsupported auto choices
    # still use the ordinary TensorRT graph; no separate user mode is needed.
    "attention_implementation": "auto",
    "native_kernel_source": None,
}
_REQUIRED = {
    "hidden_size",
    "num_heads",
    "head_dim",
    "num_layers",
    "max_sequence_length",
    "embedding_tables",
}


def _positive_integer(value, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"HSTU {name} must be a positive integer")


def parse_config(raw: dict) -> dict:
    """Validate every represented semantic option; reject unknown topology."""
    if not isinstance(raw, dict):
        raise ValueError("HSTU config must be a JSON object")
    missing = _REQUIRED - raw.keys()
    unknown = raw.keys() - (_REQUIRED | _DEFAULTS.keys() | {"reference_source"})
    if missing or unknown:
        raise ValueError(f"HSTU config missing={sorted(missing)}, unknown={sorted(unknown)}")
    config = {**_DEFAULTS, **raw}
    if config["model_type"] != "hstu" or config["schema_version"] != 1:
        raise ValueError("HSTU requires model_type=hstu and schema_version=1")
    if config["mode"] not in {"ranking", "retrieval"}:
        raise ValueError("HSTU mode must be ranking or retrieval")
    if config["attention_implementation"] not in {"auto", "tensorrt", "nvidia_hstu"}:
        raise ValueError("HSTU attention_implementation must be auto, tensorrt, or nvidia_hstu")
    source = config["native_kernel_source"]
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("HSTU native_kernel_source must be a nonempty path or null")
    for name in (
        "hidden_size",
        "num_heads",
        "head_dim",
        "num_layers",
        "max_sequence_length",
        "target_group_size",
    ):
        _positive_integer(config[name], name)
    for name in (
        "learnable_input_layernorm",
        "learnable_output_layernorm",
        "add_uvqk_bias",
        "residual",
        "is_causal",
        "disable_contextual_mask",
        "prediction_bias",
        "enable_history_cache",
    ):
        if type(config[name]) is not bool:
            raise ValueError(f"HSTU {name} must be boolean")
    for name in ("layer_norm_epsilon", "output_norm_epsilon"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"HSTU {name} must be finite and positive")
    for name in ("position_buckets", "time_buckets"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"HSTU {name} must be a nonnegative integer")
    if config["time_buckets"] not in (0, 2048):
        raise ValueError("NVIDIA HSTU timestamp encoding requires 2048 time buckets")
    if config["time_buckets"] and not config["position_buckets"]:
        raise ValueError("HSTU timestamp encoding requires position embeddings")
    if (
        type(config["scaling_seqlen"]) is not int
        or config["scaling_seqlen"] == 0
        or config["scaling_seqlen"] < -1
    ):
        raise ValueError("HSTU scaling_seqlen must be -1 or a positive integer")
    if config["prediction_activation"] not in ("relu", "gelu"):
        raise ValueError("HSTU prediction_activation must be relu or gelu")
    if config["output_postprocessor"] != "l2":
        raise ValueError("NVIDIA HSTU ranking and retrieval use L2 output normalization")
    head = config["prediction_head"]
    if not isinstance(head, list):
        raise ValueError("HSTU prediction_head must be a list")
    for value in head:
        _positive_integer(value, "prediction_head width")
    if bool(head) != (config["mode"] == "ranking"):
        raise ValueError("HSTU ranking requires a prediction head; retrieval has no head")
    tables = config["embedding_tables"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("HSTU requires embedding_tables")
    names, roles = set(), []
    for table in tables:
        if not isinstance(table, dict) or set(table) != {"name", "role", "num_embeddings"}:
            raise ValueError("HSTU embedding table requires name, role, num_embeddings")
        name = table["name"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or name in names
        ):
            raise ValueError("HSTU embedding table names must be unique identifiers")
        names.add(name)
        if table["role"] not in ("item", "action", "context"):
            raise ValueError("HSTU embedding table role must be item, action, or context")
        roles.append(table["role"])
        _positive_integer(table["num_embeddings"], "num_embeddings")
    if roles.count("item") != 1 or roles.count("action") > 1:
        raise ValueError("HSTU requires exactly one item table and at most one action table")
    if sum(t["num_embeddings"] for t in tables) >= 2**31:
        raise ValueError("HSTU merged embedding table must fit INT32 row indices")
    return config


def load_config(path: str | Path) -> dict:
    return parse_config(json.loads(Path(path).read_text(encoding="utf-8")))


def expected_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    """Canonical head-major UVQK checkpoint tensor contract."""
    d = config["hidden_size"]
    hd = config["num_heads"] * config["head_dim"]
    shapes = {
        f"embeddings.{t['name']}.weight": (t["num_embeddings"], d)
        for t in config["embedding_tables"]
    }
    for index in range(config["num_layers"]):
        prefix = f"blocks.{index}"
        shapes[f"{prefix}.uvqk.weight"] = (4 * hd, d)
        shapes[f"{prefix}.proj.weight"] = (d, hd)
        if config["add_uvqk_bias"]:
            shapes[f"{prefix}.uvqk.bias"] = (4 * hd,)
        for kind, width in (("input", d), ("output", hd)):
            if config[f"learnable_{kind}_layernorm"]:
                shapes[f"{prefix}.{kind}_norm.weight"] = (width,)
                shapes[f"{prefix}.{kind}_norm.bias"] = (width,)
    if config["position_buckets"]:
        shapes["position.weight"] = (config["position_buckets"], d)
    if config["time_buckets"]:
        shapes["time.weight"] = (config["time_buckets"] + 1, d)
    width = d
    for index, output in enumerate(config["prediction_head"]):
        shapes[f"head.{index}.weight"] = (output, width)
        if config["prediction_bias"]:
            shapes[f"head.{index}.bias"] = (output,)
        width = output
    return shapes
