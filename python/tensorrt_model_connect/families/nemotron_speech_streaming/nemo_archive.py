# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron speech streaming-owned NeMo archive resolution."""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


_TARGET_MARKERS = ("encdecrnnt", "transducer", "rnnt")


def _cfg_int(*values: object, default: int) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _read_model_config(nemo_path: Path) -> dict[str, Any]:
    import yaml

    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            if Path(member.name).name == "model_config.yaml":
                handle = tar.extractfile(member)
                if handle is not None:
                    return yaml.safe_load(handle.read()) or {}
    return {}


def _matches_nemotron_speech(cfg: dict[str, Any]) -> bool:
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
    """Create a Nemotron speech streaming model dir from a matching .nemo file."""
    cfg = _read_model_config(nemo_path)
    if not _matches_nemotron_speech(cfg):
        return None

    enc_cfg = cfg.get("encoder", {})
    defaults = cfg.get("model_defaults", {})
    dec_cfg = cfg.get("decoder", {})
    prednet = dec_cfg.get("prednet", {})
    if not isinstance(prednet, dict):
        prednet = {}
    hidden = _cfg_int(enc_cfg.get("d_model"), defaults.get("enc_hidden"), default=1024)
    pred_layers = _cfg_int(prednet.get("pred_rnn_layers"), default=1)
    pred_hidden = _cfg_int(prednet.get("pred_hidden"), defaults.get("pred_hidden"), default=640)
    vocab_size = _cfg_int(cfg.get("vocab_size"), defaults.get("vocab_size"), default=1024)

    tmp_dir = tempfile.mkdtemp(prefix="trtmc_nemo_nemotron_speech_")
    tmp_path = Path(tmp_dir)
    synthetic_config = {
        "model_type": "nemotron_speech_streaming",
        "hidden_size": hidden,
        "num_hidden_layers": pred_layers,
        "num_attention_heads": 1,
        "intermediate_size": pred_hidden,
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
        f"[trtmc build] Nemotron speech NeMo resolved: tmp_dir={tmp_dir}",
        file=sys.stderr,
    )
    return tmp_dir
