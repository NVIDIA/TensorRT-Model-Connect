# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream the lexical table into a native bundle without dequantizing its rows."""

import math
import re

import ml_dtypes
import numpy as np

from .checkpoint import FlashCheckpoint


_PREFIX = "model.language_model.layers.1.ple.ple_embedding."
_SHARD = re.compile(re.escape(_PREFIX) + r"ngram_embedding\.shard_(\d+)\.weight")


def write_ngram_table(model_dir, writer):
    """Write the FP8 row section and its checkpoint-matched lookup constants."""
    with FlashCheckpoint(model_dir) as checkpoint:
        write_checkpoint_ngram_table(checkpoint, writer)


def write_checkpoint_ngram_table(checkpoint, writer):
    shards = sorted(
        (int(m[1]), key) for key in checkpoint.keys_to_files if (m := _SHARD.fullmatch(key))
    )
    if not shards or [i for i, _ in shards] != list(range(len(shards))):
        raise ValueError("Flash-Next requires contiguous FP8 table shards numbered from zero")

    def constant(suffix, dtype, shape):
        key = _PREFIX + suffix
        return checkpoint.tensor(key, dtype, shape)

    scale = constant("ngram_embedding.weight_scale", ml_dtypes.bfloat16, [1]).item()
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid Flash-Next FP8 table scale")
    metadata = {"scale": scale}
    for name, count in (
        ("layer_multipliers", 3),
        ("ngram_heads_vocab_sizes", 16),
        ("ngram_heads_offsets", 16),
    ):
        metadata[name] = constant(name, np.int64, [count]).tolist()

    blocks = []
    total = 0
    for _, key in shards:
        tensor = checkpoint.array(key)
        shape = tensor.shape
        if (
            tensor.dtype != ml_dtypes.float8_e4m3fn
            or len(shape) != 2
            or shape[1] != 160
            or shape[0] < 1
        ):
            raise ValueError(f"Invalid Flash-Next FP8 embedding shard: {key}")
        blocks.append((tensor, shape[0]))
        total += shape[0]

    end = 0
    for offset, size in zip(metadata["ngram_heads_offsets"], metadata["ngram_heads_vocab_sizes"]):
        if size <= 0 or offset < end or offset + size > total:
            raise ValueError("Invalid Flash-Next head row range")
        end = offset + size

    with writer.open_section("ngram_e4m3.bin") as output:
        for tensor, rows in blocks:
            for first in range(0, rows, 32768):
                # At most 5 MiB per write; preserve the original E4M3 bytes.
                part = tensor[first : first + 32768].view(np.uint8)
                output.write(memoryview(part).cast("B"))
    writer.add_json("ngram_layout.json", {**metadata, "table_rows": total})
