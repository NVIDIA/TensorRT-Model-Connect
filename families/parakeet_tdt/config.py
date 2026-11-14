# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the pinned Parakeet TDT checkpoint contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParakeetTDTConfig:
    model_type: str
    architectures: tuple[str, ...]
    vocab_size: int
    blank_id: int
    pad_id: int
    durations: tuple[int, ...]
    max_symbols_per_step: int
    encoder_hidden_size: int
    encoder_layers: int
    encoder_heads: int
    encoder_ffn_size: int
    encoder_conv_kernel_size: int
    encoder_max_positions: int
    num_mel_bins: int
    subsampling_channels: int
    subsampling_factor: int
    decoder_hidden_size: int
    decoder_layers: int
    joint_activation: str
    raw: dict

    @classmethod
    def from_json(cls, text: str) -> "ParakeetTDTConfig":
        raw = json.loads(text)
        enc = raw.get("encoder_config") or {}
        return cls(
            model_type=str(raw.get("model_type", "")),
            architectures=tuple(str(item) for item in raw.get("architectures", ())),
            vocab_size=int(raw.get("vocab_size", 0)),
            blank_id=int(raw.get("blank_token_id", -1)),
            pad_id=int(raw.get("pad_token_id", -1)),
            durations=tuple(int(item) for item in raw.get("durations", ())),
            max_symbols_per_step=int(raw.get("max_symbols_per_step", 0)),
            encoder_hidden_size=int(enc.get("hidden_size", 0)),
            encoder_layers=int(enc.get("num_hidden_layers", 0)),
            encoder_heads=int(enc.get("num_attention_heads", 0)),
            encoder_ffn_size=int(enc.get("intermediate_size", 0)),
            encoder_conv_kernel_size=int(enc.get("conv_kernel_size", 0)),
            encoder_max_positions=int(enc.get("max_position_embeddings", 0)),
            num_mel_bins=int(enc.get("num_mel_bins", 0)),
            subsampling_channels=int(enc.get("subsampling_conv_channels", 0)),
            subsampling_factor=int(enc.get("subsampling_factor", 0)),
            decoder_hidden_size=int(raw.get("decoder_hidden_size", 0)),
            decoder_layers=int(raw.get("num_decoder_layers", 0)),
            joint_activation=str(raw.get("hidden_act", "")),
            raw=raw,
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ParakeetTDTConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Parakeet TDT model directory is missing config.json: {config_path}"
            )
        return cls.from_json(config_path.read_text(encoding="utf-8"))

    def validate_supported_checkpoint(self) -> None:
        expected = {
            "model_type": (self.model_type, "parakeet_tdt"),
            "architecture": (self.architectures, ("ParakeetForTDT",)),
            "vocab_size": (self.vocab_size, 8193),
            "blank_token_id": (self.blank_id, 8192),
            "durations": (self.durations, (0, 1, 2, 3, 4)),
            "encoder_hidden_size": (self.encoder_hidden_size, 1024),
            "encoder_layers": (self.encoder_layers, 24),
            "encoder_heads": (self.encoder_heads, 8),
            "encoder_ffn_size": (self.encoder_ffn_size, 4096),
            "encoder_conv_kernel_size": (self.encoder_conv_kernel_size, 9),
            "num_mel_bins": (self.num_mel_bins, 128),
            "subsampling_factor": (self.subsampling_factor, 8),
            "decoder_hidden_size": (self.decoder_hidden_size, 640),
            "decoder_layers": (self.decoder_layers, 2),
            "joint_activation": (self.joint_activation, "relu"),
            "max_symbols_per_step": (self.max_symbols_per_step, 10),
        }
        mismatches = [
            f"{name}={actual!r}, expected {wanted!r}"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise ValueError("Unsupported Parakeet TDT checkpoint contract: " + "; ".join(mismatches))
