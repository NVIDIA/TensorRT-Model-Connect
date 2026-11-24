# SPDX-License-Identifier: Apache-2.0
"""Execute the pinned Meta block itself; shim only dense/jagged transport on CPU."""

import ast
import hashlib
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from families.hstu.checkpoint import create_fixture, load_weights
from families.hstu.reference import Reference, example_inputs, layer_norm

SOURCE_SHA256 = "02c1e2b6226aded2809cd9ea8f270c9f1a35d29c1b044a4630cb9132fab5deb7"


def upstream_block():
    path = os.environ.get("HSTU_UPSTREAM_SOURCE")
    if not path:
        pytest.skip("set HSTU_UPSTREAM_SOURCE to the pinned public hstu.py")
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    import torch
    import torch.nn.functional as F
    import abc
    import math
    import typing

    namespace = dict(
        vars(typing),
        torch=torch,
        F=F,
        abc=abc,
        math=math,
        HSTUCacheState=typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    )
    keep = {
        "RelativeAttentionBiasModule",
        "SequentialTransductionUnitJagged",
        "_hstu_attention_maybe_from_cache",
    }
    tree = ast.parse(source)
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in keep
    ]
    # The mathematical functions/classes are executed unmodified, without unrelated imports.
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["SequentialTransductionUnitJagged"]


@pytest.mark.parametrize("candidates", [50, 64])
@pytest.mark.parametrize("history_length", [0, 128])
def test_reference_against_pinned_meta_block(tmp_path, candidates, history_length):
    cls = upstream_block()
    import torch

    torch.set_num_threads(4)
    c = create_fixture(tmp_path / "fixture", candidates)
    w = load_weights(tmp_path / "fixture", c)
    block = cls(
        embedding_dim=256,
        linear_hidden_dim=32,
        attention_dim=32,
        dropout_ratio=0,
        attn_dropout_ratio=0,
        num_heads=8,
        linear_activation="silu",
        relative_attention_bias_module=torch.nn.Identity(),
    ).eval()
    with torch.no_grad():
        block._uvqk.copy_(torch.from_numpy(w["uvqk"]))
        block._o.weight.copy_(torch.from_numpy(w["output_weight"].T.copy()))
        block._o.bias.copy_(torch.from_numpy(w["output_bias"]))
    x = example_inputs(c)
    x["history_lengths"].fill(history_length)
    expected = Reference(c, w).score(x)
    n = c.history_length + c.decode_steps + c.candidates
    raw = w["item_embeddings"][x["candidate_ids"]]
    history = w["item_embeddings"][x["history_ids"]]
    frozen = np.zeros((1, 50, 256), np.float32)
    scores = np.zeros((1, candidates), np.float32)
    placed = np.zeros(candidates, bool)
    picks = []

    def to_dense(values, offsets, max_lengths, padding_value=0.0):
        return values.reshape(len(offsets[0]) - 1, max_lengths[0], -1)

    def to_jagged(values, offsets):
        return (values.reshape(-1, values.shape[-1]),)

    with (
        patch.object(torch.ops.fbgemm, "jagged_to_padded_dense", to_dense, create=True),
        patch.object(torch.ops.fbgemm, "dense_to_jagged", to_jagged, create=True),
        torch.no_grad(),
    ):
        for step in range(50):
            positions = np.concatenate(
                [
                    np.arange(128),
                    history_length + np.arange(50),
                    np.full(candidates, history_length + step),
                ]
            )
            values = (
                np.concatenate([history, frozen, raw], axis=1) * np.float32(16)
                + w["position_embeddings"][positions][None, :]
            )
            mask = np.zeros((n, n), np.float32)
            mask[-candidates:, :history_length] = 1
            mask[-candidates:, 128 : 128 + step] = 1
            mask[-candidates:, -candidates:] = np.eye(candidates)
            result, _ = block(
                torch.from_numpy(values.reshape(n, 256)),
                torch.tensor([0, n], dtype=torch.int64),
                None,
                torch.from_numpy(mask),
            )
            target = result.numpy()[-candidates:][None, :]
            logits = (layer_norm(target) @ w["head_weight"] + w["head_bias"]).reshape(candidates)
            for col, item in enumerate(x["candidate_ids"][0]):
                logits[col] = logits[np.flatnonzero(x["candidate_ids"][0] == item)[0]]
            chosen = int(np.argmax(np.where(~placed, logits, -1e30)))
            placed[chosen] = True
            scores[0, chosen] = logits[chosen]
            picks.append(chosen)
            frozen[0, step] = raw[0, chosen]
    np.testing.assert_allclose(scores, expected["scores"], rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(np.array(picks), expected["picked_indices"][0])
