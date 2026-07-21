# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt official ELF YAML files to the TRTMC model-config schema."""

from pathlib import Path


_ELF_VARIANTS: dict[str, tuple[int, int, int]] = {
    "ELF-B": (12, 768, 12),
    "ELF-M": (24, 1056, 16),
    "ELF-L": (32, 1280, 16),
}


def _normalize_elf_variant(value: object) -> str:
    variant = str(value or "ELF-B").upper().replace("_", "-")
    return variant if variant in _ELF_VARIANTS else "ELF-B"


def _elf_yaml_to_config(raw: dict) -> dict:
    variant = _normalize_elf_variant(raw.get("model"))
    depth, hidden_size, num_heads = _ELF_VARIANTS[variant]
    converted = dict(raw)
    converted.update({
        "model_type": "elf",
        "model": variant,
        "elf_variant": variant,
        "hidden_size": int(raw.get("hidden_size") or raw.get("elf_hidden_size") or hidden_size),
        "num_hidden_layers": int(raw.get("depth") or raw.get("num_hidden_layers") or depth),
        "num_attention_heads": int(
            raw.get("num_heads") or raw.get("num_attention_heads") or num_heads
        ),
        "max_position_embeddings": int(
            raw.get("max_length") or raw.get("max_position_embeddings") or 128
        ),
        "text_encoder_dim": int(
            raw.get("text_encoder_dim")
            or raw.get("encoder_d_model")
            or raw.get("d_model")
            or 512
        ),
        "vocab_size": int(raw.get("vocab_size") or 0),
    })
    return converted


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Return a TRTMC config mapping for an official ELF YAML directory."""
    model_path = Path(model_dir)
    yaml_paths = [
        model_path / "config.yaml",
        model_path / "config.yml",
        *sorted(model_path.glob("*.yaml")),
        *sorted(model_path.glob("*.yml")),
    ]
    for yaml_path in yaml_paths:
        if not yaml_path.exists():
            continue
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load ELF YAML configs") from exc
        data = yaml.safe_load(yaml_path.read_text()) or {}
        variant = str(data.get("model", "")).upper().replace("_", "-") if isinstance(data, dict) else ""
        if isinstance(data, dict) and variant in _ELF_VARIANTS:
            return _elf_yaml_to_config(data)
    return None
