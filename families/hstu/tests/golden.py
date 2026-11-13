# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The validation oracle: a seeded ranking request record and the reference's
golden logits for it, with the reference's own BF16 spread.

``python -m families.hstu.tests.golden <checkpoint_dir> <record.json> <golden.json>``
writes the record (if absent) and the golden: FP32 logits per candidate and
task, and how far the reference's BF16 run lands from them (max/mean absolute
error, the smallest per-task cosine). The family's threshold file declares the
tolerance from that spread BEFORE any TensorRT output is compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import HstuConfig, load_config
from ..weights import load_checkpoint, weights_record
from .reference import Reference

RECORD_SEED = 1


def make_record(config: HstuConfig, *, seed: int = RECORD_SEED) -> dict:
    """One ranking request at the configured geometry: a user's contextual
    features, ``history_length`` (item, action) events and ``num_candidates``
    candidates, every id inside its table."""
    rng = np.random.Generator(np.random.PCG64(seed))
    user = {
        feature.name: int(rng.integers(feature.vocab_size))
        for feature in config.contextual_features
    }
    items = config.item_feature
    return {
        "user": user,
        "history": {
            items.name: rng.integers(items.vocab_size, size=config.history_length).tolist(),
            config.action_feature.name: rng.integers(
                config.action_feature.vocab_size, size=config.history_length
            ).tolist(),
        },
        "candidates": {
            items.name: rng.integers(items.vocab_size, size=config.num_candidates).tolist()
        },
    }


def checkpoint_name(config: HstuConfig) -> dict:
    """The synthetic checkpoint a golden was written from: its name and seed
    (the generator writes the same bytes for the same seed)."""
    return {"name": config.raw["name"], "seed": weights_record(config)["seed"]}


def torch_tensors(model_dir: str | Path, config: HstuConfig) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.array(bits)).view(torch.bfloat16)
        for name, bits in load_checkpoint(model_dir, config).items()
    }


def spread(test: torch.Tensor, golden: torch.Tensor) -> dict:
    """How far ``test`` lands from ``golden`` (both [candidates, tasks])."""
    error = (test - golden).abs()
    cosine = torch.nn.functional.cosine_similarity(test.T, golden.T, dim=1)
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "min_task_cosine": float(cosine.min()),
        "golden_max_abs": float(golden.abs().max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("golden", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.checkpoint)
    if not args.record.exists():
        args.record.write_text(
            json.dumps(make_record(config), separators=(",", ":")) + "\n", encoding="utf-8"
        )
    record = json.loads(args.record.read_text(encoding="utf-8"))
    tensors = torch_tensors(args.checkpoint, config)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    golden = Reference(config, tensors, precision="fp32").forward(record)
    bf16 = Reference(config, tensors, precision="bf16").forward(record)
    document = {
        "checkpoint": checkpoint_name(config),
        "record": args.record.name,
        "logits_fp32": golden["logits"].tolist(),
        "reference_bf16_spread": {
            "logits": spread(bf16["logits"], golden["logits"]),
            "candidate_embeddings": spread(
                bf16["candidate_embeddings"], golden["candidate_embeddings"]
            ),
        },
    }
    args.golden.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(document["reference_bf16_spread"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
