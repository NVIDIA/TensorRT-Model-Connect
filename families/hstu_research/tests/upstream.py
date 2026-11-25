# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the original Meta block unchanged; shim only dense/jagged transport."""

import abc
import ast
import hashlib
import math
import typing
from unittest.mock import patch
import numpy as np

SOURCE_SHA256 = "02c1e2b6226aded2809cd9ea8f270c9f1a35d29c1b044a4630cb9132fab5deb7"


def upstream_scores(config, weights, inputs, source):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(4)
    code = source.read_bytes()
    if hashlib.sha256(code).hexdigest() != SOURCE_SHA256:
        raise ValueError("upstream source differs from the pinned Meta revision")
    tree = ast.parse(code)
    names = {
        "RelativeAttentionBiasModule",
        "RelativeBucketedTimeAndPositionBasedBias",
        "SequentialTransductionUnitJagged",
        "_hstu_attention_maybe_from_cache",
    }
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    env = dict(
        vars(typing),
        torch=torch,
        F=F,
        abc=abc,
        math=math,
        HSTUCacheState=typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    )
    exec(compile(tree, str(source), "exec"), env)
    a = config.architecture
    n, d, b = a.max_sequence_length, a.hidden_size, inputs["history_ids"].shape[0]
    ids = np.pad(inputs["history_ids"], ((0, 0), (0, n - inputs["history_ids"].shape[1])))
    ts = np.pad(
        inputs["history_timestamps"], ((0, 0), (0, n - inputs["history_timestamps"].shape[1]))
    )
    raw = weights["item_embeddings"][ids] * np.float32(np.sqrt(d))
    raw = (raw + weights["position_embeddings"][None]) * (ids != 0)[..., None]
    x = torch.from_numpy(raw.reshape(b * n, d))
    offsets = torch.arange(b + 1, dtype=torch.int64) * n
    mask = torch.tril(torch.ones(n, n))

    def dense(values, offsets, max_lengths, padding_value=0.0):
        return values.reshape(b, max_lengths[0], -1)

    def jagged(values, offsets):
        return (values.reshape(b * n, -1),)

    # Causal valid rows cannot attend padded future keys; source padding outputs
    # therefore do not influence the final valid query.
    with (
        torch.no_grad(),
        patch.object(torch.ops.fbgemm, "jagged_to_padded_dense", dense, create=True),
        patch.object(torch.ops.fbgemm, "dense_to_jagged", jagged, create=True),
    ):
        for layer in range(a.num_layers):
            prefix = f"layers.{layer}."
            bias = env["RelativeBucketedTimeAndPositionBasedBias"](
                n, a.time_buckets, lambda x: (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long()
            )
            block = env["SequentialTransductionUnitJagged"](
                embedding_dim=d,
                linear_hidden_dim=a.value_dim,
                attention_dim=a.attention_dim,
                dropout_ratio=0,
                attn_dropout_ratio=0,
                num_heads=a.num_heads,
                linear_activation="silu",
                relative_attention_bias_module=bias,
                epsilon=a.epsilon,
            ).eval()
            block._uvqk.copy_(torch.from_numpy(weights[prefix + "uvqk"]))
            block._o.weight.copy_(torch.from_numpy(weights[prefix + "output_weight"].T.copy()))
            block._o.bias.copy_(torch.from_numpy(weights[prefix + "output_bias"]))
            bias._ts_w.copy_(torch.from_numpy(weights[prefix + "time_bias"]))
            bias._pos_w.copy_(torch.from_numpy(weights[prefix + "position_bias"]))
            x, _ = block(x, offsets, torch.from_numpy(ts), mask)
        user = x.reshape(b, n, d)[
            torch.arange(b), torch.from_numpy(inputs["history_lengths"][:, 0] - 1)
        ]
        user = user / torch.linalg.vector_norm(user, dim=-1, keepdim=True).clamp(min=a.l2_epsilon)
        item = torch.from_numpy(weights["item_embeddings"][inputs["candidate_ids"]])
        item = item / torch.linalg.vector_norm(item, dim=-1, keepdim=True).clamp(min=a.l2_epsilon)
        return (user[:, None] * item).sum(-1).numpy()
