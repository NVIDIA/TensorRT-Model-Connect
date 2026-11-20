# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The HSTU ranking configuration: one exact, validated reading of a checkpoint's
``config.json``.

The architecture is NVIDIA recsys-examples' HSTU ranking model
(``examples/hstu``, KuaiRand-1K inference config) evaluated without its KV
cache (``InferenceRankingGR.forward_nokvcache``): one full forward per ranking
request. The request geometry is static: ``max_history_seqlen`` history items
interleaved with their actions, the contextual features in front and
``max_num_candidates`` candidates behind, so the HSTU stack runs
``num_contextual + 2 * history + candidates`` tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .support import MODEL_TYPE


@dataclass(frozen=True)
class Feature:
    name: str
    vocab_size: int


@dataclass(frozen=True)
class HstuConfig:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    head_dim: int
    layernorm_epsilon: float
    history_length: int
    num_candidates: int
    scaling_seqlen: int
    num_position_buckets: int
    contextual_features: tuple[Feature, ...]
    item_feature: Feature
    action_feature: Feature
    prediction_head_arch: tuple[int, ...]
    num_tasks: int
    raw: dict

    @property
    def num_contextual(self) -> int:
        return len(self.contextual_features)

    @property
    def tokens(self) -> int:
        """Tokens the HSTU stack runs: contextual, interleaved history, candidates."""
        return self.num_contextual + 2 * self.history_length + self.num_candidates

    @property
    def candidate_start(self) -> int:
        return self.tokens - self.num_candidates

    @property
    def uvqk_width(self) -> int:
        return 4 * self.num_attention_heads * self.head_dim

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    def attention_blocks(self, count: int, align: int) -> list[tuple[int, int]]:
        """``count`` query-row blocks ``[start, end)`` tiling the tokens, the
        inner bounds rounded to multiples of ``align`` (a bound that rounds onto
        another drops, so a short sequence is one block). The valid mask is
        causal, so it is zero right of a block's last row: each block needs
        only the keys ``[0, end)``."""
        n = self.tokens
        inner = {round(i * n / count / align) * align for i in range(1, count)}
        bounds = sorted({0, n} | {b for b in inner if 0 < b < n})
        return list(zip(bounds, bounds[1:]))

    def position_ids(self) -> list[int]:
        """recsys-examples' inference position law (``_add_position_embeddings``
        with ``high_inds = seq_len - num_targets``, clamped to the table): token
        ``n`` reads bucket ``min(n, min(N - C, K - 1))``. Every candidate
        shares one bucket: the first candidate's (``N - C``) when the table
        holds it, else the table's last (at 4096 history events, ``N - C`` =
        8198 > 8191, so the last 7 history tokens and every candidate read
        bucket 8191)."""
        high = min(self.candidate_start, self.num_position_buckets - 1)
        return [min(n, high) for n in range(self.tokens)]


def _positive(raw: dict, key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"HSTU config {key} must be a positive integer, got {value!r}")
    return value


def _feature(value: object, where: str) -> Feature:
    if not isinstance(value, dict):
        raise ValueError(f"HSTU config {where} must be an object with name and vocab_size")
    name, vocab = value.get("name"), value.get("vocab_size")
    if not isinstance(name, str) or not name or type(vocab) is not int or vocab < 1:
        raise ValueError(f"HSTU config {where} must name a feature and a positive vocab_size")
    return Feature(name, vocab)


def parse_config(raw: dict) -> HstuConfig:
    if raw.get("model_type") != MODEL_TYPE:
        raise ValueError(f"HSTU config model_type must be {MODEL_TYPE!r}")
    contextual = raw.get("contextual_features")
    if not isinstance(contextual, list) or not contextual:
        raise ValueError("HSTU config contextual_features must be a non-empty list")
    head = raw.get("prediction_head_arch")
    if (
        not isinstance(head, list)
        or len(head) < 1
        or any(type(width) is not int or width < 1 for width in head)
    ):
        raise ValueError("HSTU config prediction_head_arch must list positive widths")
    epsilon = raw.get("layernorm_epsilon")
    if not isinstance(epsilon, float) or epsilon <= 0:
        raise ValueError("HSTU config layernorm_epsilon must be a positive float")
    config = HstuConfig(
        hidden_size=_positive(raw, "hidden_size"),
        num_layers=_positive(raw, "num_layers"),
        num_attention_heads=_positive(raw, "num_attention_heads"),
        head_dim=_positive(raw, "head_dim"),
        layernorm_epsilon=epsilon,
        history_length=_positive(raw, "max_history_seqlen"),
        num_candidates=_positive(raw, "max_num_candidates"),
        scaling_seqlen=_positive(raw, "scaling_seqlen"),
        num_position_buckets=_positive(raw, "num_position_buckets"),
        contextual_features=tuple(
            _feature(value, f"contextual_features[{index}]")
            for index, value in enumerate(contextual)
        ),
        item_feature=_feature(raw.get("item_feature"), "item_feature"),
        action_feature=_feature(raw.get("action_feature"), "action_feature"),
        prediction_head_arch=tuple(head),
        num_tasks=_positive(raw, "num_tasks"),
        raw=raw,
    )
    if config.attention_width != config.hidden_size:
        raise ValueError("HSTU heads x head_dim must equal hidden_size (the residual width)")
    if config.prediction_head_arch[-1] != config.num_tasks:
        raise ValueError("HSTU prediction head must end in num_tasks logits")
    if raw.get("prediction_head_act_type") != "relu" or raw.get("prediction_head_bias") is not True:
        raise ValueError("HSTU prediction head must be the reference's biased ReLU MLP")
    names = [
        feature.name
        for feature in (*config.contextual_features, config.item_feature, config.action_feature)
    ]
    if len(set(names)) != len(names):
        raise ValueError("HSTU feature names must be unique")
    return config


def load_config(model_dir: str | Path) -> HstuConfig:
    path = Path(model_dir) / "config.json"
    return parse_config(json.loads(path.read_text(encoding="utf-8")))
