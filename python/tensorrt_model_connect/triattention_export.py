# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for embedding TriAttention calibration stats into bundles.

This keeps the runtime-side integration independent of upstream repo layout.
We convert the upstream ``.pt`` payload into a compact JSON section that the
debug/runtime layers can load without importing torch at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig


@dataclass(frozen=True)
class TriAttentionBundleConfig:
    """User-facing runtime knobs persisted into bundle config.json."""

    kv_budget: int
    divide_length: int = 128
    recent_window: int = 128
    score_aggregation: str = "mean"
    count_prompt_tokens: bool = True
    protect_prefill: bool = True
    disable_mlr: bool = False
    disable_trig: bool = False
    stats_section: str = "triattention_stats.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "kv_budget": int(self.kv_budget),
            "divide_length": int(self.divide_length),
            "recent_window": int(self.recent_window),
            "score_aggregation": str(self.score_aggregation),
            "count_prompt_tokens": bool(self.count_prompt_tokens),
            "protect_prefill": bool(self.protect_prefill),
            "disable_mlr": bool(self.disable_mlr),
            "disable_trig": bool(self.disable_trig),
            "stats_section": str(self.stats_section),
        }


def _default_rope_style(config: ModelConfig) -> str:
    del config
    return "half"


def _resolve_rope_theta(config: ModelConfig) -> float:
    raw = dict(config.raw or {})
    rope_theta = raw.get("rope_theta")
    if rope_theta is not None:
        return float(rope_theta)
    rope_parameters = raw.get("rope_parameters")
    if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
        return float(rope_parameters["rope_theta"])
    rope_scaling = raw.get("rope_scaling")
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_theta") is not None:
        return float(rope_scaling["rope_theta"])
    return float(config.rope_theta)


def _derive_inv_freq(
    *,
    config: ModelConfig,
    head_dim: int,
) -> list[float]:
    freq_count = int(head_dim) // 2
    if freq_count <= 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    rope_theta = _resolve_rope_theta(config)
    return [
        float(rope_theta ** (-(2.0 * idx) / float(head_dim)))
        for idx in range(freq_count)
    ]


def _derive_freq_scale_sq(
    *,
    config: ModelConfig,
    head_count: int,
    freq_count: int,
) -> list[list[float]]:
    raw = dict(config.raw or {})
    rope_scaling = raw.get("rope_scaling")
    attention_factor = 1.0
    if isinstance(rope_scaling, dict):
        raw_factor = rope_scaling.get("attention_factor", rope_scaling.get("attn_factor"))
        if raw_factor is not None:
            attention_factor = float(raw_factor)
    scale_sq = float(attention_factor * attention_factor)
    row = [scale_sq] * freq_count
    return [list(row) for _ in range(max(int(head_count), 1))]


def _to_float_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    raise TypeError(f"Cannot convert value of type {type(value).__name__} to float list")


def _to_float_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Cannot convert value of type {type(value).__name__} to float matrix")
    return [_to_float_list(row) for row in value]


def _parse_layer_head_key(key: str) -> tuple[int, int] | None:
    if not key.startswith("layer") or "_head" not in key:
        return None
    layer_text, head_text = key.split("_head", 1)
    try:
        return int(layer_text.replace("layer", "")), int(head_text)
    except ValueError:
        return None


def _normalize_rkv_payload(
    payload: dict[str, Any],
    *,
    config: ModelConfig,
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}) or {})
    stats_raw = payload.get("stats", {}) or {}
    sampled_heads = [tuple(item) for item in metadata.get("sampled_heads", [])]
    if not sampled_heads and not stats_raw:
        raise ValueError(
            "TriAttention stats file does not contain metadata.sampled_heads or stats entries. "
            "Use the upstream calibration output from the TriAttention repo."
        )

    num_attention_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads or config.num_attention_heads)
    num_layers = int(config.num_hidden_layers)
    if num_attention_heads <= 0 or num_kv_heads <= 0 or num_layers <= 0:
        raise ValueError(
            "TriAttention export requires positive num_attention_heads, "
            "num_key_value_heads, and num_hidden_layers in model config."
        )
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError(
            "TriAttention export requires num_attention_heads divisible by "
            f"num_key_value_heads, got {num_attention_heads} and {num_kv_heads}."
        )

    head_dim = int(metadata.get("head_dim", config.head_dim))
    freq_count = head_dim // 2
    stats_head_count = num_attention_heads
    layer_accum: dict[int, dict[str, list[list[float]] | list[int]]] = {}
    sampled_score_heads: set[tuple[int, int]] = set()

    for key, entry in stats_raw.items():
        parsed = _parse_layer_head_key(str(key))
        if parsed is None:
            continue
        layer, head = parsed
        if layer < 0 or layer >= num_layers:
            continue
        if head < 0 or head >= num_attention_heads:
            raise ValueError(
                f"TriAttention stats file contains out-of-range head {head} for layer {layer}"
            )
        q_mean_real = _to_float_list(entry["q_mean_real"])
        q_mean_imag = _to_float_list(entry["q_mean_imag"])
        q_abs_mean = _to_float_list(entry["q_abs_mean"])
        layer_bucket = layer_accum.setdefault(
            layer,
            {
                "q_mean_real": [[0.0] * freq_count for _ in range(stats_head_count)],
                "q_mean_imag": [[0.0] * freq_count for _ in range(stats_head_count)],
                "q_abs_mean": [[0.0] * freq_count for _ in range(stats_head_count)],
                "counts": [0] * stats_head_count,
            },
        )
        layer_real = layer_bucket["q_mean_real"]
        layer_imag = layer_bucket["q_mean_imag"]
        layer_abs = layer_bucket["q_abs_mean"]
        counts = layer_bucket["counts"]
        assert isinstance(layer_real, list)
        assert isinstance(layer_imag, list)
        assert isinstance(layer_abs, list)
        assert isinstance(counts, list)
        for idx in range(freq_count):
            layer_real[head][idx] += q_mean_real[idx]
            layer_imag[head][idx] += q_mean_imag[idx]
            layer_abs[head][idx] += q_abs_mean[idx]
        counts[head] += 1
        sampled_score_heads.add((layer, head))

    if not sampled_score_heads:
        raise ValueError("TriAttention stats file has no usable R-KV head stats")

    inv_freq = metadata.get("inv_freq")
    if inv_freq is None:
        inv_freq = _derive_inv_freq(config=config, head_dim=head_dim)

    default_freq_scale_sq = _derive_freq_scale_sq(
        config=config,
        head_count=stats_head_count,
        freq_count=freq_count,
    )
    layer_stats: dict[str, Any] = {}
    out_stats: dict[str, Any] = {}
    for layer_idx in range(num_layers):
        layer_bucket = layer_accum.get(layer_idx)
        if layer_bucket is None:
            q_mean_real = [[0.0] * freq_count for _ in range(stats_head_count)]
            q_mean_imag = [[0.0] * freq_count for _ in range(stats_head_count)]
            q_abs_mean = [[0.0] * freq_count for _ in range(stats_head_count)]
        else:
            layer_real = layer_bucket["q_mean_real"]
            layer_imag = layer_bucket["q_mean_imag"]
            layer_abs = layer_bucket["q_abs_mean"]
            counts = layer_bucket["counts"]
            assert isinstance(layer_real, list)
            assert isinstance(layer_imag, list)
            assert isinstance(layer_abs, list)
            assert isinstance(counts, list)
            q_mean_real = []
            q_mean_imag = []
            q_abs_mean = []
            for score_head in range(stats_head_count):
                denom = max(int(counts[score_head]), 1)
                q_mean_real.append(
                    [float(value / denom) for value in layer_real[score_head]]
                )
                q_mean_imag.append(
                    [float(value / denom) for value in layer_imag[score_head]]
                )
                q_abs_mean.append(
                    [float(value / denom) for value in layer_abs[score_head]]
                )
        layer_stats[str(layer_idx)] = {
            "q_mean_real": q_mean_real,
            "q_mean_imag": q_mean_imag,
            "q_abs_mean": q_abs_mean,
            "freq_scale_sq": [list(row) for row in default_freq_scale_sq],
        }
        for score_head in range(stats_head_count):
            if (layer_idx, score_head) not in sampled_score_heads:
                continue
            out_stats[f"layer{layer_idx:02d}_head{score_head:02d}"] = {
                "q_mean_real": list(q_mean_real[score_head]),
                "q_mean_imag": list(q_mean_imag[score_head]),
                "q_abs_mean": list(q_abs_mean[score_head]),
            }

    out: dict[str, Any] = {
        "version": 2,
        "format": "sampled_head_rkv",
        "head_dim": head_dim,
        "rope_style": str(metadata.get("rope_style", _default_rope_style(config))),
        "rope_theta": float(metadata.get("rope_theta", _resolve_rope_theta(config))),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "stats_head_count": stats_head_count,
        "num_layers": num_layers,
        "sampled_heads": [
            [int(layer), int(head)]
            for layer, head in sorted(sampled_score_heads)
        ],
        "stats": out_stats,
        "layer_stats": layer_stats,
    }
    if inv_freq is not None:
        out["inv_freq"] = _to_float_list(inv_freq)
    return out


def _normalize_layer_stats_payload(
    payload: dict[str, Any],
    *,
    config: ModelConfig,
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}) or {})
    layer_stats = payload.get("layer_stats", {}) or {}
    if not layer_stats:
        raise ValueError(
            "Unsupported TriAttention stats layout. Expected layer_stats entries."
        )

    num_attention_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads or config.num_attention_heads)
    num_layers = int(config.num_hidden_layers)
    serialized_layer_stats: dict[str, Any] = {}
    out_stats: dict[str, Any] = {}
    sampled_score_heads: list[tuple[int, int]] = []
    stats_head_count: int | None = None
    for layer_idx, layer_entry in layer_stats.items():
        q_mean_complex = layer_entry.get("q_mean_complex")
        q_abs_mean = layer_entry.get("q_abs_mean")
        freq_scale_sq = layer_entry.get("freq_scale_sq")
        if hasattr(q_mean_complex, "detach"):
            q_mean_complex = q_mean_complex.detach().cpu()
        if hasattr(q_abs_mean, "detach"):
            q_abs_mean = q_abs_mean.detach().cpu()
        if hasattr(freq_scale_sq, "detach"):
            freq_scale_sq = freq_scale_sq.detach().cpu()
        if q_mean_complex is None or q_abs_mean is None:
            continue
        q_mean_complex = q_mean_complex.tolist()
        q_abs_mean = q_abs_mean.tolist()
        q_mean_real = []
        q_mean_imag = []
        for head_row in q_mean_complex:
            q_mean_real.append([float(pair[0]) for pair in head_row])
            q_mean_imag.append([float(pair[1]) for pair in head_row])
        row_count = len(q_mean_real)
        if stats_head_count is None:
            stats_head_count = row_count
        elif stats_head_count != row_count:
            raise ValueError("TriAttention layer_stats head count is inconsistent across layers")
        try:
            layer_num = int(layer_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid layer_stats key {layer_idx!r}") from exc
        for head_idx in range(row_count):
            sampled_score_heads.append((layer_num, head_idx))
            out_stats[f"layer{layer_num:02d}_head{head_idx:02d}"] = {
                "q_mean_real": list(q_mean_real[head_idx]),
                "q_mean_imag": list(q_mean_imag[head_idx]),
                "q_abs_mean": [float(x) for x in q_abs_mean[head_idx]],
            }
        serialized_layer_stats[str(layer_idx)] = {
            "q_mean_real": q_mean_real,
            "q_mean_imag": q_mean_imag,
            "q_abs_mean": [[float(x) for x in row] for row in q_abs_mean],
            "freq_scale_sq": (
                _to_float_matrix(freq_scale_sq)
                if freq_scale_sq is not None
                else _derive_freq_scale_sq(
                    config=config,
                    head_count=(stats_head_count or num_attention_heads),
                    freq_count=len(q_mean_real[0]) if q_mean_real else config.head_dim // 2,
                )
            ),
        }

    out: dict[str, Any] = {
        "version": 2,
        "format": "sampled_head_layer_stats",
        "head_dim": int(metadata.get("head_dim", config.head_dim)),
        "rope_style": str(metadata.get("rope_style", _default_rope_style(config))),
        "rope_theta": float(metadata.get("rope_theta", _resolve_rope_theta(config))),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "stats_head_count": int(stats_head_count or num_attention_heads),
        "num_layers": num_layers,
        "sampled_heads": [
            [int(layer), int(head)]
            for layer, head in sorted(sampled_score_heads)
        ],
        "stats": out_stats,
        "layer_stats": serialized_layer_stats,
    }
    inv_freq = metadata.get("inv_freq")
    if inv_freq is not None:
        out["inv_freq"] = _to_float_list(inv_freq)
    else:
        out["inv_freq"] = _derive_inv_freq(config=config, head_dim=out["head_dim"])
    return out


def export_triattention_stats_section(
    stats_path: str | Path,
    *,
    config: ModelConfig,
) -> bytes:
    """Convert an upstream TriAttention stats ``.pt`` file to bundle JSON bytes."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "torch is required to read TriAttention stats files. "
            "Install torch or omit --triattention-stats."
        ) from exc

    stats_file = Path(stats_path)
    payload = torch.load(stats_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(
            f"Unsupported TriAttention stats payload type {type(payload).__name__}"
        )

    if "stats" in payload:
        normalized = _normalize_rkv_payload(payload, config=config)
    elif "layer_stats" in payload:
        normalized = _normalize_layer_stats_payload(payload, config=config)
    else:
        raise ValueError(
            "Unsupported TriAttention stats format. Expected 'stats' or 'layer_stats' top-level keys."
        )

    return json.dumps(normalized, indent=2).encode("utf-8")
