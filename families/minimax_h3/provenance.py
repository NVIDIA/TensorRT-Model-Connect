# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Path-free public metadata for MiniMax-H3 native bundles."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from tensorrt_model_connect.bundle_writer import BUNDLE_MAGIC

from .config import (
    NATIVE_EXPLICIT_CANVAS_SIZES,
    SOL_ENGINE_1344X768_124_TO_345F,
    native_plan_filenames,
)


CHECKPOINT_REVISION = "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
CHECKPOINT_REPOSITORY = "MiniMaxAI/MiniMax-H3"

QUANTIZED_TRANSFORMER_CONFIG = {
    "format": "int8_tensorwise_convrot",
    "checkpoint_scope": "full_transformer_overlay",
    "quantized_weight_scope": (
        "blocks.*.{adaln_proj.linear,attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2}"
    ),
    "quantized_weight_count": 250,
    "activation_precision": "dynamic_int8_rowwise",
}

SUPER_RESOLUTION_PLAN_FILENAME = "video_super_resolution.plan"
SUPER_RESOLUTION_SECTION = "video_super_resolution_plan"
SUPER_RESOLUTION_MODEL = "realesr-general-x4v3"
SUPER_RESOLUTION_ARCHITECTURE = "SRVGGNetCompact"
SUPER_RESOLUTION_PRIMARY_FILENAME = "realesr-general-x4v3.pth"
SUPER_RESOLUTION_WEAK_FILENAME = "realesr-general-wdn-x4v3.pth"
SUPER_RESOLUTION_SOURCE_SHAPE = [480, 864]
SUPER_RESOLUTION_TARGET_SHAPE = [720, 1296]
SUPER_RESOLUTION_BATCH_PROFILE = [1, 4, 8]
SUPER_RESOLUTION_LEARNED_RESIDUAL_STRENGTH = 0.25


def checkpoint_snapshot_record(
    snapshot: Path,
    *,
    include_transformer_weights: bool = True,
) -> dict[str, object]:
    """Describe the selected public checkpoint without recording local paths."""

    root = Path(snapshot)
    required = ["text_encoder", "vae", "audio_vae", "tokenizer"]
    if include_transformer_weights:
        required.append("transformer")
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Incomplete MiniMax-H3 checkpoint; missing: " + ", ".join(missing)
        )
    return {
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "includes_transformer_weights": include_transformer_weights,
    }


def validate_checkpoint_snapshot_record(
    record: object,
    *,
    include_transformer_weights: bool = True,
) -> dict[str, object]:
    expected = {
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "includes_transformer_weights": include_transformer_weights,
    }
    if record != expected:
        raise ValueError("MiniMax-H3 checkpoint metadata does not match the public model")
    return dict(expected)


def _checkpoint_record(path: Path, role: str, filename: str) -> dict[str, object]:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"MiniMax-H3 {role} checkpoint is missing: {value}")
    if value.name != filename:
        raise ValueError(f"MiniMax-H3 {role} checkpoint must be named {filename}")
    return {"role": role, "filename": filename, "bytes": value.stat().st_size}


def super_resolution_source_identity(
    primary_checkpoint: Path,
    weak_checkpoint: Path | None,
    *,
    denoise_strength: float,
    learned_residual_strength: float = SUPER_RESOLUTION_LEARNED_RESIDUAL_STRENGTH,
) -> dict[str, object]:
    if isinstance(denoise_strength, bool) or not isinstance(denoise_strength, (int, float)):
        raise ValueError("MiniMax-H3 super-resolution denoise_strength must be numeric")
    if isinstance(learned_residual_strength, bool) or not isinstance(
        learned_residual_strength, (int, float)
    ):
        raise ValueError("MiniMax-H3 learned residual strength must be numeric")
    strength = float(denoise_strength)
    residual = float(learned_residual_strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("MiniMax-H3 super-resolution denoise_strength must be in [0, 1]")
    if not math.isfinite(residual) or not 0.0 <= residual <= 1.0:
        raise ValueError("MiniMax-H3 learned residual strength must be in [0, 1]")
    if weak_checkpoint is None and strength != 1.0:
        raise ValueError("MiniMax-H3 denoise blending requires the weak checkpoint")
    sources = [_checkpoint_record(primary_checkpoint, "primary", SUPER_RESOLUTION_PRIMARY_FILENAME)]
    if weak_checkpoint is not None:
        sources.append(_checkpoint_record(weak_checkpoint, "weak_denoise", SUPER_RESOLUTION_WEAK_FILENAME))
    return {
        "model": SUPER_RESOLUTION_MODEL,
        "architecture": SUPER_RESOLUTION_ARCHITECTURE,
        "denoise_strength": strength,
        "learned_residual_strength": residual,
        "sources": sources,
    }


def validate_super_resolution_source_identity(record: object) -> dict[str, object]:
    if not isinstance(record, dict):
        raise ValueError("MiniMax-H3 super-resolution source metadata must be an object")
    if record.get("model") != SUPER_RESOLUTION_MODEL:
        raise ValueError("MiniMax-H3 super-resolution model is unsupported")
    if record.get("architecture") != SUPER_RESOLUTION_ARCHITECTURE:
        raise ValueError("MiniMax-H3 super-resolution architecture is unsupported")
    sources = record.get("sources")
    if not isinstance(sources, list) or len(sources) not in (1, 2):
        raise ValueError("MiniMax-H3 super-resolution sources are invalid")
    expected = (
        ("primary", SUPER_RESOLUTION_PRIMARY_FILENAME),
        ("weak_denoise", SUPER_RESOLUTION_WEAK_FILENAME),
    )
    for source, (role, filename) in zip(sources, expected, strict=False):
        if (
            not isinstance(source, dict)
            or source.get("role") != role
            or source.get("filename") != filename
            or not isinstance(source.get("bytes"), int)
            or source["bytes"] <= 0
        ):
            raise ValueError("MiniMax-H3 super-resolution source metadata is invalid")
    strength = record.get("denoise_strength")
    residual = record.get("learned_residual_strength")
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        raise ValueError("MiniMax-H3 super-resolution denoise strength is invalid")
    if not isinstance(residual, (int, float)) or isinstance(residual, bool):
        raise ValueError("MiniMax-H3 super-resolution residual strength is invalid")
    if not 0.0 <= float(strength) <= 1.0 or not 0.0 <= float(residual) <= 1.0:
        raise ValueError("MiniMax-H3 super-resolution strengths must be in [0, 1]")
    if len(sources) == 1 and float(strength) != 1.0:
        raise ValueError("MiniMax-H3 denoise blending requires the weak checkpoint")
    return dict(record)


def super_resolution_bundle_config(source_identity: object) -> dict[str, object]:
    source = validate_super_resolution_source_identity(source_identity)
    return {
        "section": SUPER_RESOLUTION_SECTION,
        "mode": "explicit",
        "input_name": "frames",
        "output_name": "upscaled_frames",
        "model": source["model"],
        "architecture": source["architecture"],
        "source_shape": list(SUPER_RESOLUTION_SOURCE_SHAPE),
        "target_shape": list(SUPER_RESOLUTION_TARGET_SHAPE),
        "batch_profile": list(SUPER_RESOLUTION_BATCH_PROFILE),
        "denoise_strength": source["denoise_strength"],
        "learned_residual_strength": source["learned_residual_strength"],
        "sources": source["sources"],
    }


def validate_super_resolution_bundle_config(record: object) -> dict[str, object]:
    if not isinstance(record, dict) or record.get("section") != SUPER_RESOLUTION_SECTION:
        raise ValueError("MiniMax-H3 super-resolution bundle metadata is invalid")
    if record.get("mode") != "explicit":
        raise ValueError("MiniMax-H3 legacy or unsupported SR mode; rebuild with super_resolution=true")
    if record.get("input_name") != "frames" or record.get("output_name") != "upscaled_frames":
        raise ValueError("MiniMax-H3 super-resolution tensor names are invalid")
    if record.get("source_shape") != SUPER_RESOLUTION_SOURCE_SHAPE:
        raise ValueError("MiniMax-H3 super-resolution source shape is invalid")
    if record.get("target_shape") != SUPER_RESOLUTION_TARGET_SHAPE:
        raise ValueError("MiniMax-H3 super-resolution target shape is invalid")
    if record.get("batch_profile") != SUPER_RESOLUTION_BATCH_PROFILE:
        raise ValueError("MiniMax-H3 super-resolution batch profile is invalid")
    validate_super_resolution_source_identity(
        {
            "model": record.get("model"),
            "architecture": record.get("architecture"),
            "denoise_strength": record.get("denoise_strength"),
            "learned_residual_strength": record.get("learned_residual_strength"),
            "sources": record.get("sources"),
        }
    )
    return dict(record)


def validate_quantized_transformer_metadata(record: object) -> dict[str, object]:
    from .quantized_checkpoint import QUANTIZED_CHECKPOINT_IDENTITY

    expected = QUANTIZED_CHECKPOINT_IDENTITY.bundle_metadata()
    if record != expected:
        raise ValueError("MiniMax-H3 quantized transformer metadata does not match the public model")
    return expected


def validate_workspace_limit_bytes(
    record: object,
    *,
    profile=SOL_ENGINE_1344X768_124_TO_345F,
    additional_plan_filenames: tuple[str, ...] = (),
    excluded_plan_filenames: tuple[str, ...] = (),
) -> dict[str, object]:
    expected = tuple(
        filename
        for filename in native_plan_filenames()
        if filename not in excluded_plan_filenames
    ) + tuple(additional_plan_filenames)
    if not isinstance(record, dict) or set(record) != set(expected):
        raise ValueError("MiniMax-H3 workspace metadata must cover every native plan")
    for filename, value in record.items():
        if value not in ("tensorrt_default", None) and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"MiniMax-H3 workspace value for {filename} is invalid")
    del profile
    return dict(record)


def load_bundle_config(bundle: Path) -> dict[str, object]:
    """Load the canonical runtime section from a current ModelConnect bundle."""

    with Path(bundle).open("rb") as stream:
        if stream.read(8) != BUNDLE_MAGIC:
            raise ValueError("MiniMax-H3 bundle has invalid magic")
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("MiniMax-H3 bundle header is truncated")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = json.loads(stream.read(header_size))
        sections = header.get("sections", {})
        section = sections.get("runtime.json") or sections.get("config.json")
        if not isinstance(section, dict):
            raise ValueError("MiniMax-H3 bundle is missing runtime.json")
        length = section.get("length", section.get("size"))
        stream.seek(16 + header_size + int(section["offset"]))
        return json.loads(stream.read(int(length)))


def validate_native_bundle_config(bundle: Path, *, source_revision: str | None = None) -> dict:
    config = load_bundle_config(bundle)
    if config.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise ValueError("MiniMax-H3 bundle checkpoint revision is invalid")
    if config.get("explicit_canvas_sizes") != [list(size) for size in NATIVE_EXPLICIT_CANVAS_SIZES]:
        raise ValueError("MiniMax-H3 bundle canvas profile is invalid")
    del source_revision
    return config
