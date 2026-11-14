# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate the OneFormer fields this family builds.

The family reproduces ``OneFormerForUniversalSegmentation`` with a Swin
backbone at inference (``is_training`` false, so the text mapper is absent).
Every field the graph depends on is checked here; a checkpoint outside this
layout is rejected instead of being built with a silently different topology.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Fixed by the reference implementation, not by the checkpoint config.
NUM_FEATURE_LEVELS = 3
DEFORMABLE_POINTS = 4
NUM_GROUPS = 32


@dataclass(frozen=True)
class SwinConfig:
    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: int
    patch_size: int
    mlp_ratio: float
    layer_norm_eps: float

    def stage_dims(self) -> tuple[int, ...]:
        return tuple(self.embed_dim * 2**stage for stage in range(len(self.depths)))


@dataclass(frozen=True)
class PreprocessConfig:
    shortest_edge: int
    longest_edge: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float


@dataclass(frozen=True)
class OneFormerConfig:
    swin: SwinConfig
    preprocess: PreprocessConfig
    conv_dim: int
    mask_dim: int
    hidden_dim: int
    num_attention_heads: int
    encoder_layers: int
    encoder_feedforward_dim: int
    decoder_layers: int
    dim_feedforward: int
    query_dec_layers: int
    num_queries: int
    num_labels: int
    task_seq_len: int
    layer_norm_eps: float
    class_names: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"oneformer: {message}")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"oneformer requires {path.name} in the checkpoint: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must be a JSON object")
    return value


def _int_tuple(value, name: str) -> tuple[int, ...]:
    _require(
        isinstance(value, list) and all(isinstance(v, int) for v in value),
        f"{name} must be a list of ints",
    )
    return tuple(int(v) for v in value)


def _swin(raw: dict) -> SwinConfig:
    _require(raw.get("model_type") == "swin", "only Swin backbones are supported")
    _require(
        not raw.get("use_absolute_embeddings", False),
        "Swin absolute position embeddings are not supported",
    )
    _require(raw.get("hidden_act", "gelu") == "gelu", "the Swin MLP must use exact GELU")
    _require(bool(raw.get("qkv_bias", True)), "the Swin attention must have query/key/value biases")
    depths = _int_tuple(raw.get("depths"), "backbone depths")
    heads = _int_tuple(raw.get("num_heads"), "backbone num_heads")
    _require(len(depths) == 4 and len(heads) == 4, "the Swin backbone must have four stages")
    out_features = raw.get("out_features")
    _require(
        out_features == ["stage1", "stage2", "stage3", "stage4"],
        "the backbone must emit stage1-stage4",
    )
    embed_dim = int(raw["embed_dim"])
    for stage, count in enumerate(heads):
        _require((embed_dim * 2**stage) % count == 0, "Swin heads must divide the stage width")
    swin = SwinConfig(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=heads,
        window_size=int(raw["window_size"]),
        patch_size=int(raw["patch_size"]),
        mlp_ratio=float(raw.get("mlp_ratio", 4.0)),
        layer_norm_eps=float(raw.get("layer_norm_eps", 1e-5)),
    )
    _require(swin.window_size >= 2 and swin.patch_size >= 1, "invalid Swin window or patch size")
    return swin


def _preprocess(raw: dict) -> PreprocessConfig:
    _require(
        raw.get("do_resize", True)
        and raw.get("do_rescale", True)
        and raw.get("do_normalize", True),
        "the image processor must resize, rescale and normalize",
    )
    _require(int(raw.get("resample", 2)) == 2, "the image processor must resample bilinearly")
    size = raw.get("size")
    _require(
        isinstance(size, dict) and {"shortest_edge", "longest_edge"} <= set(size),
        "the image processor size must give shortest_edge and longest_edge",
    )
    shortest, longest = int(size["shortest_edge"]), int(size["longest_edge"])
    _require(0 < shortest < longest, "the image processor needs 0 < shortest_edge < longest_edge")
    mean = tuple(float(v) for v in raw.get("image_mean", ()))
    std = tuple(float(v) for v in raw.get("image_std", ()))
    _require(
        len(mean) == 3 and len(std) == 3 and all(v > 0 for v in std),
        "image mean/std must have 3 channels",
    )
    rescale = float(raw.get("rescale_factor", 1 / 255))
    _require(rescale > 0, "rescale_factor must be positive")
    return PreprocessConfig(shortest, longest, mean, std, rescale)


def load_config(model_dir: str | Path) -> OneFormerConfig:
    root = Path(model_dir)
    raw = _read_json(root / "config.json")
    _require(
        raw.get("model_type") == "oneformer",
        f"model_type must be 'oneformer', got {raw.get('model_type')!r}",
    )
    _require(
        not raw.get("is_training", False), "training checkpoints (text mapper) are not supported"
    )
    _require(not raw.get("pre_norm", False), "pre-norm transformer decoders are not supported")
    _require(raw.get("use_task_norm", True), "the task token must be normalized (use_task_norm)")
    _require(raw.get("norm", "GN") == "GN", "the pixel decoder must use GroupNorm")
    _require(
        not raw.get("enforce_input_proj", False) and int(raw["conv_dim"]) == int(raw["hidden_dim"]),
        "the transformer input projections must be identities (conv_dim == hidden_dim)",
    )
    _require(
        _int_tuple(raw.get("strides"), "strides") == (4, 8, 16, 32),
        "feature strides must be 4, 8, 16, 32",
    )
    _require(int(raw.get("common_stride", 4)) == 4, "common_stride must be 4 (one FPN level)")
    id2label = raw.get("id2label") or {}
    num_labels = len(id2label)
    _require(
        num_labels > 0 and sorted(int(k) for k in id2label) == list(range(num_labels)),
        "id2label must name every class id 0..N-1",
    )
    class_names = tuple(str(id2label[str(index)]) for index in range(num_labels))
    config = OneFormerConfig(
        swin=_swin(raw.get("backbone_config") or {}),
        preprocess=_preprocess(_read_json(root / "preprocessor_config.json")),
        conv_dim=int(raw["conv_dim"]),
        mask_dim=int(raw["mask_dim"]),
        hidden_dim=int(raw["hidden_dim"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        encoder_layers=int(raw["encoder_layers"]),
        encoder_feedforward_dim=int(raw["encoder_feedforward_dim"]),
        decoder_layers=int(raw["decoder_layers"]),
        dim_feedforward=int(raw["dim_feedforward"]),
        query_dec_layers=int(raw["query_dec_layers"]),
        num_queries=int(raw["num_queries"]),
        num_labels=num_labels,
        task_seq_len=int(raw["task_seq_len"]),
        layer_norm_eps=float(raw.get("layer_norm_eps", 1e-5)),
        class_names=class_names,
    )
    _require(config.conv_dim % NUM_GROUPS == 0, "conv_dim must divide into 32 GroupNorm groups")
    _require(
        config.hidden_dim % config.num_attention_heads == 0,
        "attention heads must divide hidden_dim",
    )
    _require(config.mask_dim == config.hidden_dim, "mask_dim must equal hidden_dim")
    _require(
        config.decoder_layers >= 2 and config.num_queries >= 2, "too few decoder layers or queries"
    )
    _require(
        int(raw.get("num_classes", num_labels)) == num_labels, "num_classes must match id2label"
    )
    return config
