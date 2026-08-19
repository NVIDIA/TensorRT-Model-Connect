# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Magpie-owned NeMo archive resolution."""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


_TARGET_MARKERS = ("magpietts", "magpie_tts")


def _read_model_config(nemo_path: Path) -> dict[str, Any]:
    import yaml

    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            if Path(member.name).name == "model_config.yaml":
                handle = tar.extractfile(member)
                if handle is not None:
                    return yaml.safe_load(handle.read()) or {}
    return {}


def _matches_magpie(cfg: dict[str, Any]) -> bool:
    target = " ".join(
        str(cfg.get(key, "") or "")
        for key in ("target", "_target_", "model_type")
    ).lower()
    return any(marker in target for marker in _TARGET_MARKERS)


def _symlink_archive(tmp_path: Path, nemo_path: Path) -> None:
    link = tmp_path / nemo_path.name
    if not link.exists():
        os.symlink(str(nemo_path.resolve()), str(link))


def resolve_nemo_archive(nemo_path: Path) -> str | None:
    """Create a Magpie synthetic model directory from a matching .nemo file."""
    cfg = _read_model_config(nemo_path)
    if not _matches_magpie(cfg):
        return None

    enc_cfg = cfg.get("encoder", {})
    dec_cfg = cfg.get("decoder", {})
    hidden = int(cfg.get("embedding_dim", enc_cfg.get("d_model", 768)))
    dec_layers = int(dec_cfg.get("n_layers", 12))
    dec_heads = int(dec_cfg.get("sa_n_heads", 12))
    dec_ffn = int(dec_cfg.get("d_ffn", 3072))
    vocab_size = int(cfg.get("text_vocab_size", 2380))

    tmp_dir = tempfile.mkdtemp(prefix="trtmc_nemo_magpie_tts_")
    tmp_path = Path(tmp_dir)
    synthetic_config = {
        "model_type": "magpie_tts",
        "hidden_size": hidden,
        "num_hidden_layers": dec_layers,
        "num_attention_heads": dec_heads,
        "intermediate_size": dec_ffn,
        "vocab_size": vocab_size,
        "rms_norm_eps": 1e-5,
        "_nemo_archive_path": str(nemo_path),
    }
    (tmp_path / "config.json").write_text(
        json.dumps(synthetic_config, indent=2),
        encoding="utf-8",
    )
    _symlink_archive(tmp_path, nemo_path)
    print(
        f"[trtmc build] Magpie NeMo resolved: tmp_dir={tmp_dir}",
        file=sys.stderr,
    )
    return tmp_dir
