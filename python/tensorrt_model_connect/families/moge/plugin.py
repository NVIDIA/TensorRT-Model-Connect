# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoGe-2 ViT-L family plugin."""

from __future__ import annotations

from pathlib import Path


_CHECKPOINT = Path("model.pt")


def config_from_dir(model_dir: str | Path) -> dict | None:
    """Recognize the requested MoGe-2 ViT-L checkpoint snapshot."""

    if not (Path(model_dir) / _CHECKPOINT).is_file():
        return None
    return {
        "model_type": "moge",
        "architectures": ["MoGeModelV2"],
        "runtime_strategy": "moge_monocular_geometry",
        "vocab_size": 0,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "max_position_embeddings": 1841,
        "requires_tokenizer": False,
    }


class MoGePlugin:
    name = "moge"
    runtime_strategy = "moge_monocular_geometry"
    requires_tokenizer = False
    default_build_precision = "fp32"

    def matches(self, model_type: str) -> bool:
        normalized = (model_type or "").lower().replace("-", "").replace("_", "")
        return normalized in {"moge", "moge2", "mogemodelv2"}

    def load_weights(
        self,
        model_dir: str,
        config,
        *,
        precision: str = "fp32",
    ) -> dict[str, str]:
        del config, precision
        root = Path(model_dir).resolve()
        if not (root / _CHECKPOINT).is_file():
            raise FileNotFoundError(f"MoGe checkpoint not found: {root / _CHECKPOINT}")
        return {"model_dir": str(root)}

    def build_engine(
        self,
        config,
        weights: dict[str, str],
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del config, max_cache_length
        if quant_ctx is not None:
            raise ValueError("MoGe does not support quantized builds")
        model_dir = weights.get("model_dir")
        if not model_dir:
            raise RuntimeError("MoGe model directory was not loaded")
        from .model import build_moge_engine

        return build_moge_engine(model_dir, precision=precision, verbose=verbose)


plugin = MoGePlugin()
