# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-sequence NumPy oracle, independent of the staged last-query graph."""

import numpy as np


def norm(x, epsilon):
    centered = x - x.mean(-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + epsilon)


def silu(x):
    return x / (1 + np.exp(-x))


def unit(x, epsilon):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), epsilon)


class Reference:
    def __init__(self, config, weights):
        self.c, self.w = config, weights

    def encode(self, inputs):
        a, w = self.c.architecture, self.w
        ids, lengths = inputs["history_ids"], inputs["history_lengths"][:, 0]
        b, t = ids.shape
        x = (
            w["item_embeddings"][ids] * np.float32(np.sqrt(a.hidden_size))
            + w["position_embeddings"][:t][None]
        ) * (ids != 0)[..., None]
        ts = inputs["history_timestamps"]
        delta = ts[:, 1 : t + 1, None] - ts[:, None, :t]
        bucket = np.clip(
            np.log(np.maximum(np.abs(delta), 1).astype(np.float32)) / np.float32(0.301),
            0,
            a.time_buckets,
        ).astype(np.int64)
        positions = a.max_sequence_length - 1 + np.arange(t)[None, :] - np.arange(t)[:, None]
        mask = (
            np.tril(np.ones((t, t), np.float32))[None, None]
            * (np.arange(t)[None, :] < lengths[:, None])[:, None, None, :]
        )
        for i in range(a.num_layers):
            prefix = f"layers.{i}."
            projected = silu(norm(x, a.epsilon) @ w[prefix + "uvqk"])
            vwidth, qwidth = a.num_heads * a.value_dim, a.num_heads * a.attention_dim
            u, v, q, key = np.split(projected, [vwidth, 2 * vwidth, 2 * vwidth + qwidth], axis=-1)
            q = q.reshape(b, t, a.num_heads, a.attention_dim).transpose(0, 2, 1, 3)
            key = key.reshape(b, t, a.num_heads, a.attention_dim).transpose(0, 2, 1, 3)
            v = v.reshape(b, t, a.num_heads, a.value_dim).transpose(0, 2, 1, 3)
            bias = w[prefix + "position_bias"][positions][None] + w[prefix + "time_bias"][bucket]
            weights = silu(q @ key.transpose(0, 1, 3, 2) + bias[:, None]) / np.float32(
                a.max_sequence_length
            )
            attended = ((weights * mask) @ v).transpose(0, 2, 1, 3).reshape(b, t, vwidth)
            x = (
                x
                + (u * norm(attended, a.epsilon)) @ w[prefix + "output_weight"]
                + w[prefix + "output_bias"]
            )
        return unit(x[np.arange(b), lengths - 1], a.l2_epsilon)

    def score(self, inputs):
        a, w = self.c.architecture, self.w
        user = self.encode(inputs)
        item = unit(w["item_embeddings"][inputs["candidate_ids"]], a.l2_epsilon)
        logits = np.sum(user[:, None, :] * item, axis=-1)
        valid = np.arange(item.shape[1])[None, :] < inputs["num_candidates"]
        picked = np.argmax(np.where(valid, logits, -1e30), axis=1).astype(np.int64)
        picked = np.where(valid.any(1), picked, -1)[:, None]
        return dict(scores=np.where(valid, logits, 0).astype(np.float32), picked_indices=picked)


def example_inputs(config, batch=1, seed=42):
    a, s = config.architecture, config.serving
    rng = np.random.default_rng(seed)
    lengths = (
        np.array([s.history_length], np.int64)
        if batch == 1
        else np.linspace(max(1, s.history_length // 2), s.history_length, batch, dtype=np.int64)
    )
    ids = rng.integers(1, a.vocab_size, (batch, s.history_length), dtype=np.int64)
    ids[np.arange(s.history_length)[None, :] >= lengths[:, None]] = 0
    return dict(
        history_ids=ids,
        history_lengths=lengths[:, None],
        history_timestamps=np.broadcast_to(
            1_600_000_000 + np.arange(s.history_length + 1) * 3600, (batch, s.history_length + 1)
        ).copy(),
        candidate_ids=rng.integers(1, a.vocab_size, (batch, s.candidates), dtype=np.int64),
        num_candidates=np.full((batch, 1), s.candidates, np.int64),
    )
