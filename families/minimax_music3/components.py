# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and cross-check the MiniMax-Music3 component geometry.

Every component carries its own ``config.json``; nothing states the pipeline's
shape as a whole. This module reads each one and checks the places the numbers
have to line up, so a checkpoint that pairs mismatched components fails by name
rather than by a shape error deep inside a builder.

Values at revision ``fbdf52fbaaca799592917417eb05f1899f1255ec``::

    transformer         36 layers, 32 heads x 64 = 2048, ff 8192,
                        in_channels 128, condition_dim 2048, rotary_dim 32
    condition_encoder   weights 8 codebook streams, hidden 4096, out_dim 2048,
                        24000 Hz in / hop 960, 44100 Hz out / hop 512
    rvq_depth_decoder   4 layers, hidden 4096, 16 heads,
                        8 codebooks of 1024, max_position_embeddings 16
    vocoder             latent_channels 128, decoder 1024 -> 1536,
                        upsampling [8, 8, 4, 2] = 512x, 44100 Hz
    language_model      Qwen3, 36 layers, hidden 4096, 32/8 heads,
                        head_dim 128, ff 12288, rope_theta 1e6
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path

#: Upstream limits, from the model card.
MAX_AUDIO_FRAMES = 9000
AUDIO_FRAMES_PER_SECOND = 25
MAX_PROMPT_TOKENS = 5000

#: The model card's prose says 32 kHz in three places while every component
#: config says 44100. The configs win, and not by inference: in diffusers
#: v0.40.0 ``MiniMaxMusic3ModularPipeline.sampling_rate`` defaults to 44100 and
#: otherwise returns ``vocoder.config.sampling_rate``, the modular decoder
#: block documents itself as stitching "the final stereo waveform at 44.1 kHz",
#: and none of the three pipeline blocks resamples. The 32 kHz figure appears
#: to describe the SGLang-Omni serving path rather than the model.
DOCUMENTED_SAMPLING_RATE = 32000


class ComponentGeometryError(ValueError):
    """Raised when a component config is missing or the pipeline disagrees."""


@dataclass(frozen=True)
class PipelineGeometry:
    """Shape of the MiniMax-Music3 pipeline, read from its component configs."""

    transformer_layers: int
    transformer_hidden: int
    transformer_in_channels: int
    condition_dim: int
    condition_encoder_layers: int
    language_model_hidden: int
    language_model_layers: int
    depth_decoder_codebooks: int
    depth_decoder_vocab: int
    vocoder_latent_channels: int
    vocoder_upsample_factor: int
    sampling_rate: int

    @property
    def latent_frames_per_second(self) -> float:
        """Latent rate the vocoder consumes, in frames per second."""

        return self.sampling_rate / self.vocoder_upsample_factor

    @property
    def max_audio_seconds(self) -> float:
        """Longest generation the upstream frame limit allows."""

        return MAX_AUDIO_FRAMES / AUDIO_FRAMES_PER_SECOND


def _read_config(model_dir: Path, component: str, name: str = "config.json") -> dict:
    path = Path(model_dir) / component / name
    if not path.is_file():
        raise ComponentGeometryError(
            f"MiniMax-Music3 checkpoint is missing {component}/{name}"
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComponentGeometryError(
            f"{component}/{name} is not readable JSON"
        ) from exc
    if not isinstance(config, dict):
        raise ComponentGeometryError(f"{component}/{name} must be a JSON object")
    return config


def _require_int(config: Mapping, key: str, where: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComponentGeometryError(f"{where} must declare an integer {key!r}")
    return value


def detect_pipeline_geometry(model_dir: Path) -> PipelineGeometry:
    """Read every component config and check that the pipeline lines up."""

    transformer = _read_config(model_dir, "transformer")
    condition = _read_config(model_dir, "condition_encoder")
    depth = _read_config(model_dir, "rvq_depth_decoder")
    vocoder = _read_config(model_dir, "vocoder")
    language = _read_config(model_dir, "language_model")

    heads = _require_int(transformer, "num_attention_heads", "transformer")
    head_dim = _require_int(transformer, "attention_head_dim", "transformer")
    transformer_hidden = heads * head_dim
    condition_dim = _require_int(transformer, "condition_dim", "transformer")
    in_channels = _require_int(transformer, "in_channels", "transformer")

    condition_out = _require_int(condition, "out_dim", "condition_encoder")
    if condition_out != condition_dim:
        raise ComponentGeometryError(
            f"condition_encoder emits {condition_out} but the transformer "
            f"conditions on {condition_dim}"
        )
    if transformer_hidden != condition_dim:
        raise ComponentGeometryError(
            f"transformer width {transformer_hidden} disagrees with its "
            f"condition_dim {condition_dim}"
        )

    latent_channels = _require_int(vocoder, "latent_channels", "vocoder")
    if latent_channels != in_channels:
        raise ComponentGeometryError(
            f"vocoder consumes {latent_channels} latent channels but the "
            f"transformer emits {in_channels}"
        )

    ratios = vocoder.get("upsampling_ratios")
    if not isinstance(ratios, list) or not all(
        isinstance(r, int) and r > 0 for r in ratios
    ):
        raise ComponentGeometryError(
            "vocoder must declare positive integer upsampling_ratios"
        )
    upsample_factor = prod(ratios)

    sampling_rate = _require_int(vocoder, "sampling_rate", "vocoder")
    condition_out_rate = _require_int(
        condition, "output_sampling_rate", "condition_encoder"
    )
    if condition_out_rate != sampling_rate:
        raise ComponentGeometryError(
            f"condition_encoder targets {condition_out_rate} Hz but the "
            f"vocoder emits {sampling_rate} Hz"
        )

    hop = _require_int(condition, "output_hop_length", "condition_encoder")
    if hop != upsample_factor:
        raise ComponentGeometryError(
            f"condition_encoder hop {hop} disagrees with the vocoder's "
            f"{upsample_factor}x upsampling"
        )

    language_hidden = _require_int(language, "hidden_size", "language_model")
    depth_hidden = _require_int(depth, "hidden_size", "rvq_depth_decoder")
    if depth_hidden != language_hidden:
        raise ComponentGeometryError(
            f"rvq_depth_decoder consumes {depth_hidden} but the language model "
            f"emits {language_hidden}"
        )

    return PipelineGeometry(
        transformer_layers=_require_int(transformer, "num_layers", "transformer"),
        transformer_hidden=transformer_hidden,
        transformer_in_channels=in_channels,
        condition_dim=condition_dim,
        condition_encoder_layers=_require_int(
            condition, "num_condition_layers", "condition_encoder"
        ),
        language_model_hidden=language_hidden,
        language_model_layers=_require_int(
            language, "num_hidden_layers", "language_model"
        ),
        depth_decoder_codebooks=_require_int(
            depth, "num_codebooks", "rvq_depth_decoder"
        ),
        depth_decoder_vocab=_require_int(
            depth, "audio_vocab_size", "rvq_depth_decoder"
        ),
        vocoder_latent_channels=latent_channels,
        vocoder_upsample_factor=upsample_factor,
        sampling_rate=sampling_rate,
    )
