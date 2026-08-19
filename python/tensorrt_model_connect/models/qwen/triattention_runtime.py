# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned pure-Python TriAttention selection helpers.

This module intentionally avoids TensorRT/CUDA imports so it can be unit-tested
independently by Qwen debug tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _geometric_offsets(max_length: int = 65536) -> np.ndarray:
    values: list[float] = []
    current = 1
    while current <= max_length:
        values.append(float(current))
        current *= 2
    return np.asarray(values, dtype=np.float32)


def _split_complex_pairs(head_values: np.ndarray, rope_style: str) -> tuple[np.ndarray, np.ndarray]:
    if head_values.shape[1] % 2 != 0:
        raise ValueError("TriAttention requires an even head dimension")
    if rope_style == "interleaved":
        return head_values[:, ::2], head_values[:, 1::2]
    half = head_values.shape[1] // 2
    return head_values[:, :half], head_values[:, half:]


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (matrix - mean) / std


@dataclass(frozen=True)
class TriAttentionRuntimeConfig:
    kv_budget: int
    recent_window: int = 128
    score_aggregation: str = "mean"
    count_prompt_tokens: bool = True
    protect_prefill: bool = True
    disable_mlr: bool = False
    disable_trig: bool = False
    rope_style: str = "half"
    offset_max_length: int = 65536

    @classmethod
    def from_bundle_config(
        cls,
        tri_cfg: dict[str, Any],
        *,
        rope_style: str,
        max_cache_length: int,
    ) -> "TriAttentionRuntimeConfig":
        budget = int(tri_cfg.get("kv_budget", max_cache_length))
        return cls(
            kv_budget=budget,
            recent_window=int(tri_cfg.get("recent_window", 128)),
            score_aggregation=str(tri_cfg.get("score_aggregation", "mean")),
            count_prompt_tokens=bool(tri_cfg.get("count_prompt_tokens", True)),
            protect_prefill=bool(tri_cfg.get("protect_prefill", True)),
            disable_mlr=bool(tri_cfg.get("disable_mlr", False)),
            disable_trig=bool(tri_cfg.get("disable_trig", False)),
            rope_style=str(rope_style),
        )


class TriAttentionSelector:
    """Shared-token TriAttention selector for dense cache layouts.

    This is an MVP integration for TRT engines whose cache layout is
    ``[seq, attention_size]``. It keeps a single ordered token set shared by
    all heads/layers. Upstream TriAttention also supports stronger per-head
    selection, which would require head-wise compaction in our runtime path.
    """

    def __init__(self, stats_payload: dict[str, Any], config: TriAttentionRuntimeConfig):
        self.config = config
        self.head_dim = int(stats_payload["head_dim"])
        self.rope_style = str(stats_payload.get("rope_style", config.rope_style))
        self.num_attention_heads = int(stats_payload.get("num_attention_heads", 0))
        self.num_key_value_heads = int(
            stats_payload.get("num_key_value_heads", self.num_attention_heads or 0)
        )
        self.stats_head_count = int(
            stats_payload.get(
                "stats_head_count",
                self.num_key_value_heads or self.num_attention_heads or 0,
            )
        )
        self.sampled_heads = [tuple(item) for item in stats_payload.get("sampled_heads", [])]
        if not self.sampled_heads:
            raise ValueError("TriAttention stats payload has no sampled_heads")
        self.cache_group_size = 1
        if (
            self.num_attention_heads > 0
            and self.stats_head_count > 0
            and (self.num_attention_heads % self.stats_head_count) == 0
        ):
            self.cache_group_size = self.num_attention_heads // self.stats_head_count

        inv_freq = stats_payload.get("inv_freq")
        if inv_freq is None:
            rope_theta = float(stats_payload.get("rope_theta", 10000.0))
            freq_idx = np.arange(self.head_dim // 2, dtype=np.float32)
            inv_freq = 1.0 / np.power(rope_theta, (2.0 * freq_idx) / float(self.head_dim))
        self.inv_freq = np.asarray(inv_freq, dtype=np.float32)
        self.offsets = _geometric_offsets(config.offset_max_length)

        self.stats: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        raw_stats = stats_payload.get("stats", {})
        for layer, head in self.sampled_heads:
            key = f"layer{int(layer):02d}_head{int(head):02d}"
            entry = raw_stats.get(key)
            if entry is None:
                raise ValueError(f"TriAttention stats payload is missing {key!r}")
            self.stats[(int(layer), int(head))] = {
                "q_mean_real": np.asarray(entry["q_mean_real"], dtype=np.float32),
                "q_mean_imag": np.asarray(entry["q_mean_imag"], dtype=np.float32),
                "q_abs_mean": np.asarray(entry["q_abs_mean"], dtype=np.float32),
            }

    def _score_one_head(
        self,
        head_values: np.ndarray,
        *,
        next_position: int,
        stats: dict[str, np.ndarray],
    ) -> np.ndarray:
        k_real, k_imag = _split_complex_pairs(head_values, self.rope_style)

        q_real = stats["q_mean_real"][None, :]
        q_imag = stats["q_mean_imag"][None, :]
        q_abs = stats["q_abs_mean"][None, :]
        q_mean_abs = np.sqrt(np.maximum(q_real * q_real + q_imag * q_imag, 1e-8))

        prod_real = q_real * k_real + q_imag * k_imag
        prod_imag = q_imag * k_real - q_real * k_imag
        k_abs = np.sqrt(np.maximum(k_real * k_real + k_imag * k_imag, 1e-8))

        if self.config.disable_trig:
            trig_term = np.zeros((head_values.shape[0],), dtype=np.float32)
        else:
            phase = (
                (float(next_position) + self.offsets[:, None])[:, :, None]
                * self.inv_freq[None, None, :]
            )
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            score_offsets = (
                prod_real[None, :, :] * cos_phase
                - prod_imag[None, :, :] * sin_phase
            ).sum(axis=2)
            if self.config.score_aggregation == "max":
                trig_term = score_offsets.max(axis=0)
            else:
                trig_term = score_offsets.mean(axis=0)

        if self.config.disable_mlr:
            extra_coef = q_abs
        else:
            extra_coef = q_abs - q_mean_abs
        additive = (k_abs * extra_coef).sum(axis=1)
        return trig_term + additive

    def select_keep_indices(
        self,
        layer_caches: dict[int, np.ndarray],
        *,
        cache_positions: list[int],
        next_position: int,
        prefix_length: int = 0,
    ) -> np.ndarray:
        total_tokens = len(cache_positions)
        old_budget = max(0, self.config.kv_budget - 1)
        if total_tokens <= old_budget:
            return np.arange(total_tokens, dtype=np.int32)

        reserve_recent = min(max(self.config.recent_window - 1, 0), total_tokens, old_budget)

        position_array = np.asarray(cache_positions, dtype=np.int32)
        reserve_mask = np.zeros((total_tokens,), dtype=bool)
        if reserve_recent > 0:
            recent_order = np.argsort(position_array)[-reserve_recent:]
            reserve_mask[recent_order] = True

        if self.config.protect_prefill and prefix_length > 0:
            reserve_mask |= position_array < int(prefix_length)

        reserved = np.flatnonzero(reserve_mask)
        if reserved.size >= old_budget:
            return np.sort(reserved[:old_budget]).astype(np.int32)

        candidate_idx = np.flatnonzero(~reserve_mask)
        if candidate_idx.size == 0:
            return np.sort(reserved).astype(np.int32)

        head_scores: list[np.ndarray] = []
        for layer, head in self.sampled_heads:
            layer_cache = layer_caches.get(int(layer))
            if layer_cache is None:
                continue
            start = int(head) * self.cache_group_size * self.head_dim
            end = start + self.head_dim
            if end > layer_cache.shape[1]:
                continue
            sample_cache = layer_cache[:, start:end][candidate_idx]
            sample_score = self._score_one_head(
                sample_cache,
                next_position=next_position,
                stats=self.stats[(int(layer), int(head))],
            )
            head_scores.append(sample_score.astype(np.float32, copy=False))

        need = max(0, old_budget - reserved.size)
        if not head_scores:
            tail = candidate_idx[-need:] if need > 0 else np.empty((0,), dtype=np.int32)
            return np.sort(np.concatenate([reserved, tail])).astype(np.int32)

        head_matrix = np.stack(head_scores, axis=0)
        head_matrix = _zscore_rows(head_matrix)
        combined = head_matrix.max(axis=0)
        order = np.argsort(combined)
        chosen_candidates = candidate_idx[order[-need:]] if need > 0 else np.empty((0,), dtype=np.int32)
        keep = np.concatenate([reserved, chosen_candidates])
        keep.sort()
        return keep.astype(np.int32, copy=False)
