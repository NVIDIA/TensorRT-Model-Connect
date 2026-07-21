# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance checkpoint staging for the canonical family configuration entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol


LANCE_VARIANT_DIRS = {"image": "Lance_3B", "video": "Lance_3B_Video"}
_TOP_LEVEL_FILES = (
    "model.safetensors",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "tokenizer_config.json",
)


class ModelConfig(Protocol):
    """Lance builder view of the repository's parsed model configuration."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    attention_size: int
    raw: dict[str, Any]


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target, link)


def stage_model_dir(src: Path, out: Path, *, variant: str = "image") -> Path:
    """Stage one Lance variant and return its buildable model directory."""
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    llm_dir = src / LANCE_VARIANT_DIRS[variant]
    vit_path = src / "Qwen2.5-VL-ViT" / "vit.safetensors"

    if not (llm_dir / "llm_config.json").exists():
        raise FileNotFoundError(
            f"{llm_dir}/llm_config.json not found (bad source/variant?)"
        )
    if not vit_path.exists():
        raise FileNotFoundError(f"{vit_path} not found")

    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((llm_dir / "llm_config.json").read_text())
    config["model_type"] = "lance"
    (out / "config.json").write_text(json.dumps(config, indent=2))

    for name in _TOP_LEVEL_FILES:
        source = llm_dir / name
        if source.exists():
            _symlink(source, out / name)
        elif name == "model.safetensors":
            raise FileNotFoundError(f"{source} not found")

    _symlink(vit_path, out / "vision" / "model.safetensors")
    return out


def resolve_model_dir(src: Path) -> Path | None:
    """Stage a downloaded non-flat Lance repository for normal builds."""
    if not (src / "Lance_3B" / "llm_config.json").exists():
        return None
    digest = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:12]
    staging_root = Path(
        os.environ.get(
            "TRTMC_FAMILY_MODEL_ROOT",
            str(Path(tempfile.gettempdir()) / "trtmc-family-models"),
        )
    )
    return stage_model_dir(src, staging_root / f"lance-image-{digest}")
