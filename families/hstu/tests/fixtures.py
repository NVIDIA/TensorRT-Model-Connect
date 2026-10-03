# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded, non-pretrained checkpoints for HSTU semantic qualification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from families.hstu.config import expected_shapes, parse_config


SEED = 20260915
ITEM_IDS = tuple(2**40 + 17 * index for index in range(24))
ACTION_IDS = tuple(2**35 + 11 * index for index in range(5))
CONTEXT_IDS = tuple(2**36 + 13 * index for index in range(6))


def tiny_config(**overrides) -> dict:
    """Two genuine HSTU blocks, unequal features, and multi-task ranking outputs."""
    raw = {
        "hidden_size": 16,
        "num_heads": 2,
        "head_dim": 8,
        "num_layers": 2,
        "max_sequence_length": 32,
        "mode": "ranking",
        "embedding_tables": [
            {"name": "item", "role": "item", "num_embeddings": len(ITEM_IDS)},
            {"name": "action", "role": "action", "num_embeddings": len(ACTION_IDS)},
            {"name": "context", "role": "context", "num_embeddings": len(CONTEXT_IDS)},
        ],
        "prediction_head": [12, 2],
        "position_buckets": 32,
        # A fixed divisor makes batch-padding invariance a valid comparison.
        "scaling_seqlen": 32,
        **overrides,
    }
    if raw["mode"] == "retrieval":
        raw["prediction_head"] = []
        if "embedding_tables" not in overrides:
            raw["embedding_tables"] = raw["embedding_tables"][:1]
    return parse_config(raw)


def make_checkpoint(directory: Path, **overrides) -> dict:
    """Write canonical FP32 weights plus sparse INT64 feature keys deterministically."""
    directory.mkdir(parents=True, exist_ok=True)
    config = tiny_config(**overrides)
    rng = np.random.default_rng(SEED)
    tensors = {}
    for name, shape in sorted(expected_shapes(config).items()):
        if name.endswith("_norm.weight"):
            values = rng.uniform(0.7, 1.3, size=shape)
        elif name.endswith("_norm.bias"):
            values = rng.normal(0.0, 0.08, size=shape)
        elif name.endswith(".bias"):
            values = rng.normal(0.0, 0.06, size=shape)
        elif name.startswith("embeddings."):
            values = rng.normal(0.0, 0.3, size=shape)
        elif name in {"position.weight", "time.weight"}:
            values = rng.normal(0.0, 0.04, size=shape)
        else:
            values = rng.normal(0.0, 0.35 / np.sqrt(shape[-1]), size=shape)
        tensors[name] = np.asarray(values, dtype=np.float32)
    key_parameters = {"item": (2**40, 17), "action": (2**35, 11), "context": (2**36, 13)}
    for table in config["embedding_tables"]:
        base, step = key_parameters[table["role"]]
        tensors[f"embeddings.{table['name']}.keys"] = np.asarray(
            [base + step * index for index in range(table["num_embeddings"])], dtype=np.int64
        )
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def sample_request(
    config: dict, *, history_lengths: list[int] | None = None,
    candidate_lengths: list[int] | None = None, context_counts: list[int] | None = None,
) -> dict:
    """Two users with unequal history/context/candidate lengths and large raw IDs."""
    sequences = [
        {
            "history_item_ids": list(ITEM_IDS[1:4]),
            "candidate_item_ids": list(ITEM_IDS[7:11]),
        },
        {
            "history_item_ids": list(ITEM_IDS[4:6]),
            "candidate_item_ids": list(ITEM_IDS[12:14]),
        },
    ]
    tables = {table["role"]: table for table in config["embedding_tables"]}
    if "action" in tables:
        sequences[0]["history_action_ids"] = list(ACTION_IDS[1:4])
        sequences[1]["history_action_ids"] = list(ACTION_IDS[:2])
    if "context" in tables:
        sequences[0]["contextual_features"] = [{"name": "context", "ids": list(CONTEXT_IDS[1:3])}]
        sequences[1]["contextual_features"] = [{"name": "context", "ids": [CONTEXT_IDS[4]]}]
    if history_lengths is not None:
        assert candidate_lengths is not None and len(history_lengths) == len(candidate_lengths) == 2
        context_counts = context_counts or [2, 1]
        for batch, sequence in enumerate(sequences):
            sequence["history_item_ids"] = [
                ITEM_IDS[(index + batch + 1) % len(ITEM_IDS)]
                for index in range(history_lengths[batch])
            ]
            sequence["candidate_item_ids"] = [
                ITEM_IDS[(index + batch + 7) % len(ITEM_IDS)]
                for index in range(candidate_lengths[batch])
            ]
            if "action" in tables:
                count = tables["action"]["num_embeddings"]
                sequence["history_action_ids"] = [
                    2**35 + 11 * ((index + batch) % count)
                    for index in range(history_lengths[batch])
                ]
            if "context" in tables:
                sequence["contextual_features"] = [{
                    "name": "context", "ids": list(CONTEXT_IDS[:context_counts[batch]])
                }]
    if config["time_buckets"]:
        for sequence in sequences:
            count = token_count(sequence)
            if config["mode"] == "retrieval":
                count -= len(sequence["candidate_item_ids"])
            # Nonuniform intervals exercise more than the zero/one timestamp bucket.
            sequence["token_timestamps"] = [1_700_000_000 + 71 * i * i for i in range(count)]
    return {"sequences": sequences}


def context_length(sequence: dict) -> int:
    return sum(len(feature["ids"]) for feature in sequence.get("contextual_features", ()))


def token_count(sequence: dict) -> int:
    return (
        context_length(sequence)
        + len(sequence["history_item_ids"])
        + len(sequence.get("history_action_ids", ()))
        + len(sequence["candidate_item_ids"])
    )
