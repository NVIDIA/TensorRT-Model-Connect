# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-sequence FP32 NumPy oracle for the official DLRMv3 dense model.

Equations follow Meta's Apache-2.0 generative-recommenders implementation;
this computes history and targets together, independently of serving caches.
"""

import numpy as np
from .checkpoint import BLOCK, HEAD, POS, POST, PRE


def sigmoid(x):
    return np.exp(-np.logaddexp(np.float32(0), -x))


class Reference:
    def __init__(self, config, weights):
        self.c, self.w = config, weights

    def norm(self, x, weight, bias, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        centered = x - mean
        return (
            centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + eps) * weight + bias
        )

    def ln(self, x, prefix, eps=1e-5):
        return self.norm(x, self.w[prefix + "weight"], self.w[prefix + "bias"], eps)

    def linear(self, x, prefix):
        return x @ self.w[prefix + "weight"].T + self.w[prefix + "bias"]

    def mlp(self, x, prefix):
        x = self.linear(x, prefix + "0.")
        x = x * sigmoid(self.ln(x, prefix + "1."))
        return self.ln(self.linear(x, prefix + "2."), prefix + "3.")

    def score(self, inputs):
        w, s = self.w, self.c.serving
        b, t, c = inputs["history_lengths"].shape[0], s.history_length, s.candidates
        item = np.concatenate(
            [inputs["history_item_embeddings"], inputs["candidate_item_embeddings"]], 1
        )
        category = np.concatenate(
            [inputs["history_category_embeddings"], inputs["candidate_category_embeddings"]], 1
        )
        action = (
            inputs["history_actions"][..., None]
            & w[PRE + "_action_encoder._combined_action_weights"]
        ) > 0
        action = (action[..., None] * w[PRE + "_action_encoder._action_embedding_table"]).reshape(
            b, t, 40
        )
        target_actions = np.broadcast_to(
            w[PRE + "_action_encoder._target_action_embedding_table"], (b, c, 40)
        )
        x = self.mlp(item, PRE + "_content_embedding_mlp.") + self.mlp(
            category, PRE + "_additional_embedding_mlp."
        )
        x += self.mlp(np.concatenate([action, target_actions], 1), PRE + "_action_embedding_mlp.")
        lengths = inputs["history_lengths"]
        contextual = (
            inputs["user_embeddings"] * ((lengths + inputs["num_candidates"]) >= 20)[..., None]
        )
        contextual = (
            contextual @ w[PRE + "_batched_contextual_linear_weights"][0]
            + w[PRE + "_batched_contextual_linear_bias"]
        )
        x = np.concatenate([contextual, x], 1) * np.float32(np.sqrt(512))
        positions = np.concatenate(
            [
                np.zeros((b, 1), np.int32),
                np.clip(lengths + 1 - np.arange(t)[None], 1, 8191),
                np.ones((b, c), np.int32),
            ],
            1,
        )
        times = np.concatenate(
            [
                np.zeros((b, 1), np.int64),
                inputs["history_timestamps"],
                np.broadcast_to(inputs["query_timestamp"], (b, c)),
            ],
            1,
        )
        delta = np.maximum((inputs["query_timestamp"] - times).astype(np.float32), 1e-6) / 60
        bucket = np.clip(np.sqrt(delta), 0, 2048).astype(np.int32)
        x += (
            w[POS + "_position_embeddings_weight"][positions]
            + w[POS + "_timestamp_embeddings_weight"][bucket]
        )
        n = t + c + 1
        prefix = np.arange(t + 1)[None] <= lengths
        valid = np.concatenate([prefix, np.arange(c)[None] < inputs["num_candidates"]], 1)
        mask = np.zeros((b, n, n), np.float32)
        mask[:, : t + 1, : t + 1] = np.tril(np.ones((t + 1, t + 1)))
        mask[:, t + 1 :, : t + 1] = 1
        mask[:, t + 1 :, t + 1 :] = np.eye(c)
        mask *= valid[:, None, :]
        for i in range(5):
            p = BLOCK + str(i) + "."
            normalized = self.norm(x, w[p + "_input_norm_weight"], w[p + "_input_norm_bias"], 1e-6)
            uvqk = normalized @ w[p + "_uvqk_weight"] + w[p + "_uvqk_beta"]
            u, v, q, k = np.split(uvqk, 4, axis=-1)
            u *= sigmoid(u)
            v, q, k = [a.reshape(b, n, 4, 128).transpose(0, 2, 1, 3) for a in (v, q, k)]
            logits = (q @ k.transpose(0, 1, 3, 2)) * np.float32(128**-0.5)
            attn = (logits * sigmoid(logits) / np.float32(n) * mask[:, None]) @ v
            attn = attn.transpose(0, 2, 1, 3).reshape(b, n, 512)
            gated = u * self.norm(
                attn, w[p + "_output_norm_weight"], w[p + "_output_norm_bias"], 1e-6
            )
            x += np.concatenate([u, attn, gated], -1) @ w[p + "_output_weight"]
        user = x[:, t + 1 :]
        # Timestamp arithmetic is FP32 before floor/remainder, matching upstream.
        units = np.floor(
            inputs["query_timestamp"].astype(np.float32)[..., None] / w[POST + "_period_units"]
        )
        phase = (
            np.remainder(units, w[POST + "_units_per_period"])
            / w[POST + "_units_per_period"]
            * np.float32(2 * 3.14)
        )
        time_features = np.stack([np.cos(phase), np.sin(phase)], -1).reshape(b, 1, 4)
        user = self.ln(
            self.linear(
                np.concatenate([user, np.broadcast_to(time_features, (b, c, 4))], -1),
                POST + "_time_feature_combiner.",
            ),
            POST + "_layer_norm.",
        )
        item = self.mlp(
            np.concatenate(
                [inputs["candidate_item_embeddings"], inputs["candidate_category_embeddings"]], -1
            ),
            "_item_embedding_mlp.",
        )
        head = self.linear(user * item, HEAD + "0.")
        head *= sigmoid(self.ln(head, HEAD + "1."))
        scores = sigmoid(self.linear(head, HEAD + "2.")[..., 0])
        return dict(scores=np.where(valid[:, t + 1 :], scores, 0).astype(np.float32))


def example_inputs(config, batch=1, seed=42):
    rng = np.random.default_rng(seed)
    t, c = config.serving.history_length, config.serving.candidates

    def emb(n):
        return rng.normal(0, 0.05, (batch, n, 512)).astype(np.float32)

    lengths = np.linspace(max(0, t // 2), t, batch, dtype=np.int32)[:, None]
    if batch == 1:
        lengths[:] = t
    query = np.full((batch, 1), 1700000000, np.int64)
    return dict(
        history_item_embeddings=emb(t),
        history_category_embeddings=emb(t),
        history_lengths=lengths,
        history_timestamps=query - (t - np.arange(t))[None] * 3600,
        history_actions=rng.integers(0, 32, (batch, t), dtype=np.int32),
        user_embeddings=emb(1),
        query_timestamp=query,
        num_candidates=np.full((batch, 1), c, np.int32),
        candidate_item_embeddings=emb(c),
        candidate_category_embeddings=emb(c),
    )
