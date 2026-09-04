# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local config adapter for the official SANA-WM YAML layout."""

from __future__ import annotations

import json
from pathlib import Path


_VARIANTS: dict[str, tuple[int, int, int, tuple[int, int, int]]] = {
    # name suffix -> (depth, hidden_size, num_heads, patch_size)
    "600M_P1_D28": (28, 1152, 16, (1, 1, 1)),
    "600M_P2_D28": (28, 1152, 16, (1, 2, 2)),
    "1600M_P1_D20": (20, 2240, 20, (1, 1, 1)),
    "1600M_P2_D20": (20, 2240, 20, (1, 2, 2)),
    "1600M_P2S1_D20": (20, 2240, 20, (1, 1, 1)),
    "2000M_P2_D20": (20, 2304, 18, (1, 2, 2)),
    "2800M_P2_D28": (28, 2240, 20, (1, 2, 2)),
    "4000M_P2_D28": (28, 2560, 20, (1, 2, 2)),
    "4800M_P1_D60": (60, 2240, 20, (1, 1, 1)),
    "4800M_P2_D60": (60, 2240, 20, (1, 2, 2)),
}


def _arch_defaults(
    model_name: str,
    model: dict,
) -> tuple[int, int, int, tuple[int, int, int]]:
    suffix = model_name.removeprefix("SanaMSVideoCamCtrl_")
    if suffix in _VARIANTS:
        return _VARIANTS[suffix]

    depth = 20
    if "_D" in model_name:
        try:
            depth = int(model_name.rsplit("_D", 1)[1].split("_", 1)[0])
        except (ValueError, IndexError):
            pass
    num_heads = int(model.get("num_heads") or model.get("num_attention_heads") or 20)
    linear_head_dim = int(model.get("linear_head_dim", 112))
    hidden_size = int(model.get("hidden_size") or linear_head_dim * num_heads)
    return depth, hidden_size, num_heads, (1, 1, 1)


def _tokenizer_adds_special_tokens(model_path: Path) -> bool:
    config_path = model_path / "text_encoder" / "tokenizer_config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(config.get("add_bos_token") or config.get("add_eos_token"))


def config_from_dir(model_dir: str | Path) -> dict | None:
    model_path = Path(model_dir)
    config_path = model_path / "config.yaml"
    if not config_path.is_file():
        return None

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load SANA-WM config.yaml") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None

    model = raw.get("model")
    vae = raw.get("vae")
    if not isinstance(model, dict) or not isinstance(vae, dict):
        return None
    model_name = str(model.get("model", ""))
    camctrl_type = str(model.get("camctrl_type", ""))
    vae_type = str(vae.get("vae_type", ""))
    if not (
        model_name.startswith("SanaMSVideoCamCtrl")
        or "CamCtrl" in camctrl_type
        or vae_type == "LTX2VAE_diffusers"
    ):
        return None

    text_encoder = raw.get("text_encoder")
    if not isinstance(text_encoder, dict):
        text_encoder = {}
    model_name = model_name or "SanaMSVideoCamCtrl_1600M_P1_D20"
    depth, hidden_size, num_heads, patch_size = _arch_defaults(model_name, model)

    config = dict(raw)
    config.update(
        {
            "model_type": "sana_wm",
            "architectures": ["SanaWmWorldModel"],
            "hidden_size": hidden_size,
            "num_hidden_layers": depth,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_heads,
            "vocab_size": 0,
            "max_position_embeddings": int(text_encoder.get("model_max_length", 300)),
            "video_height": 704,
            "video_width": 1280,
            "video_num_frames": 321,
            "vae_latent_dim": int(vae.get("vae_latent_dim", 128)),
            "vae_downsample_rate": int(vae.get("vae_downsample_rate", 32)),
            "patch_size": list(patch_size),
            "linear_head_dim": int(
                model.get("linear_head_dim", hidden_size // max(num_heads, 1))
            ),
            "tokenizer_add_special_tokens": int(
                _tokenizer_adds_special_tokens(model_path)
            ),
            "sana_wm_config": raw,
        }
    )
    return config
