# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent defaults keyed only by the public E2E task contract."""

from __future__ import annotations

from dataclasses import dataclass


PERFORMANCE_MODES = frozenset({"decode", "diffusion", "enc_dec", "multi_stage", "single_pass"})


@dataclass(frozen=True)
class TaskDefaults:
    """Shared behavior that is invariant across owners implementing one task."""

    cli_commands: tuple[str, ...]
    performance_mode: str


TASK_DEFAULTS: dict[str, TaskDefaults] = {
    "diffusion_media_generation": TaskDefaults(("run", "generate-video"), "diffusion"),
    "diffusion_text_generation": TaskDefaults(("run",), "diffusion"),
    "embedding": TaskDefaults(("embed",), "single_pass"),
    "encoder_only_nlp": TaskDefaults(("encode",), "single_pass"),
    "image_classification": TaskDefaults(("classify",), "single_pass"),
    "image_feature_extraction": TaskDefaults(("extract-features",), "single_pass"),
    "neural_operator": TaskDefaults(("solve",), "single_pass"),
    "omni_multimodal": TaskDefaults(
        ("run", "embed", "transcribe", "generate-audio", "speak"),
        "multi_stage",
    ),
    "prompted_segmentation": TaskDefaults(("segment-prompted",), "single_pass"),
    "reranking": TaskDefaults(("rerank",), "single_pass"),
    "segmentation": TaskDefaults(("segment",), "single_pass"),
    "speech_to_speech": TaskDefaults(("speak",), "multi_stage"),
    "speech_to_text": TaskDefaults(("transcribe",), "enc_dec"),
    "stereo_disparity": TaskDefaults(("disparity",), "single_pass"),
    "text_generation_causal": TaskDefaults(("run",), "decode"),
    "text_to_audio": TaskDefaults(("generate-audio",), "multi_stage"),
    "vision_language_generation": TaskDefaults(("run",), "enc_dec"),
}


def task_defaults(task_strategy: str) -> TaskDefaults:
    """Return the required shared contract; unknown tasks fail closed."""
    try:
        return TASK_DEFAULTS[task_strategy]
    except KeyError as exc:
        raise ValueError(f"unknown task_strategy {task_strategy!r}") from exc
