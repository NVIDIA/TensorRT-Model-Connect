# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense NumPy form of Meta's public HSTU block and a target-aware ranking adapter.

The block is LN -> SiLU(U,V,Q,K) -> SiLU(QK^T)/N -> mask -> AV ->
LN -> multiply U -> output projection + residual. There is no softmax.
The ranking head and greedy slate adapter are defined here, not claimed upstream.
"""

import numpy as np


def silu(x):
    return x / (1 + np.exp(-x))


def layer_norm(x, eps=1e-6):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)


class Reference:
    def __init__(self, config, weights):
        self.c, self.w = config, weights

    def project(self, raw, positions):
        x = raw * np.float32(np.sqrt(self.c.hidden_size)) + self.w["position_embeddings"][positions]
        uvqk = silu(layer_norm(x, self.c.epsilon) @ self.w["uvqk"])
        u, v, q, k = np.split(uvqk, 4, axis=-1)

        def heads(a):
            return a.reshape(*a.shape[:-1], self.c.num_heads, -1).transpose(0, 2, 1, 3)

        return x, u, heads(v), heads(q), heads(k)

    def score(self, inputs):
        c, w = self.c, self.w
        ids = inputs["candidate_ids"]
        b = ids.shape[0]
        lengths = inputs["history_lengths"].reshape(b)
        counts = inputs["num_candidates"].reshape(b)
        raw = w["item_embeddings"][ids]
        hist = w["item_embeddings"][inputs["history_ids"]]
        _, _, hv, _, hk = self.project(hist, np.arange(c.history_length)[None, :])
        scores = np.zeros((b, c.candidates), np.float32)
        placed = np.zeros_like(scores, dtype=bool)
        frozen = np.zeros((b, c.decode_steps, c.hidden_size), np.float32)
        picks = np.full((b, c.decode_steps), -1, np.int64)
        eye = np.eye(c.candidates, dtype=bool)[None, None, :, :]
        for step in range(c.decode_steps):
            x, u, cv, q, ck = self.project(raw, (lengths + step)[:, None])
            _, _, fv, _, fk = self.project(
                frozen, lengths[:, None] + np.arange(c.decode_steps)[None, :]
            )
            k = np.concatenate([hk, fk, ck], axis=2)
            v = np.concatenate([hv, fv, cv], axis=2)
            mask = np.concatenate(
                [
                    np.broadcast_to(
                        (np.arange(c.history_length)[None, :] < lengths[:, None])[:, None, None, :],
                        (b, 1, c.candidates, c.history_length),
                    ),
                    np.broadcast_to(
                        (np.arange(c.decode_steps) < step)[None, None, None, :],
                        (b, 1, c.candidates, c.decode_steps),
                    ),
                    np.broadcast_to(eye, (b, 1, c.candidates, c.candidates)),
                ],
                axis=-1,
            )
            a = (silu(q @ k.transpose(0, 1, 3, 2)) * mask / np.float32(c.normalization_length)) @ v
            a = a.transpose(0, 2, 1, 3).reshape(b, c.candidates, c.hidden_size)
            encoded = (u * layer_norm(a, c.epsilon)) @ w["output_weight"] + w["output_bias"] + x
            logits = (layer_norm(encoded, c.epsilon) @ w["head_weight"] + w["head_bias"]).reshape(
                b, c.candidates
            )
            # Canonicalize mathematically identical candidates before tie-breaking.
            first = np.argmax(ids[:, :, None] == ids[:, None, :], axis=2)
            logits = np.take_along_axis(logits, first, axis=1)
            eligible = (np.arange(c.candidates)[None, :] < counts[:, None]) & ~placed
            chosen = np.argmax(np.where(eligible, logits, np.float32(-1e30)), axis=1)
            exists = eligible.any(axis=1)
            row = np.arange(b)[exists]
            col = chosen[exists]
            scores[row, col] = logits[row, col]
            placed[row, col] = True
            frozen[row, step] = raw[row, col]
            picks[row, step] = col
        return {"scores": scores, "picked_indices": picks}


def example_inputs(config, batch=1, seed=42):
    rng = np.random.default_rng(seed)
    c = config
    return {
        "history_ids": rng.integers(0, c.vocab_size, (batch, c.history_length), dtype=np.int64),
        "history_lengths": (
            np.array([c.history_length], np.int64)
            if batch == 1
            else np.linspace(0, c.history_length, batch, dtype=np.int64)
        )[:, None],
        "candidate_ids": rng.integers(0, c.vocab_size, (batch, c.candidates), dtype=np.int64),
        "num_candidates": np.full((batch, 1), c.candidates, np.int64),
    }
