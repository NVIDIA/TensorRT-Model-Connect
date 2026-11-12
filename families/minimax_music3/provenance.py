# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned sources for MiniMax-Music3 reference runs.

The checkpoint and the reference implementation are both pinned so that a
parity result names exactly what produced it.

The model card directs readers to install diffusers from the commit on
`huggingface/diffusers#14456 <https://github.com/huggingface/diffusers/pull/14456>`_
"until it is merged". It was merged on 2026-08-13, and diffusers 0.40.0,
released 2026-08-20, ships the pipeline, so the reference pins that release
rather than a pull-request commit. The checkpoint's components declare
``_diffusers_version = "0.40.0.dev0"``, which is the pre-release of that same
line.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Checkpoint, pinned to the revision the onboarding request names.
CHECKPOINT_REPOSITORY = "MiniMaxAI/MiniMax-Music3"
CHECKPOINT_REVISION = "fbdf52fbaaca799592917417eb05f1899f1255ec"
HF_CACHE_REPOSITORY = "models--MiniMaxAI--MiniMax-Music3"

#: Reference implementation.
DIFFUSERS_REFERENCE_DISTRIBUTION = "diffusers"
DIFFUSERS_REFERENCE_VERSION = "0.40.0"
DIFFUSERS_REFERENCE_REPOSITORY = "https://github.com/huggingface/diffusers.git"
DIFFUSERS_REFERENCE_TAG = "v0.40.0"

#: The pull request that added the pipeline, and the commit the model card
#: pins. Kept for traceability; the release above is what is installed.
DIFFUSERS_UPSTREAM_PULL_REQUEST = 14456
DIFFUSERS_UPSTREAM_COMMIT = "dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d"

#: Reference modules that must be importable for a parity run to be meaningful.
DIFFUSERS_REFERENCE_MODULES = (
    "diffusers.models.transformers.transformer_minimax_music3",
    "diffusers.models.transformers.minimax_music3_rvq_depth_decoder",
    "diffusers.modular_pipelines.minimax_music3.modular_pipeline",
)

#: Modular-pipeline class the checkpoint's index names.
PIPELINE_CLASS = "MiniMaxMusic3ModularPipeline"

#: Files a usable snapshot must contain. The checkpoint also ships `qwen_7B/`
#: and two root `.pth` files that the modular index never references; see the
#: family descriptor for why they are not downloaded.
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "modular_model_index.json",
    "condition_encoder/config.json",
    "condition_encoder/diffusion_pytorch_model.safetensors",
    "language_model/config.json",
    "language_model/model.safetensors.index.json",
    "rvq_depth_decoder/config.json",
    "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
)

#: Inputs of the reference run reproduced for parity. The upstream local
#: example names the description ``prompt`` and the lyrics ``lyrics``; this
#: family gives the lyrics the shared ``prompt`` field instead, because the
#: task contract scores a transcript against it. See the plugin docstring.
REFERENCE_CALL = {
    "audio_duration_seconds": 60.0,
    "seed": 7,
    "output": "audios",
    "dtype": "bfloat16",
}


def reference_requirement() -> str:
    """Return the pip requirement line for the reference profile."""

    return f"{DIFFUSERS_REFERENCE_DISTRIBUTION}=={DIFFUSERS_REFERENCE_VERSION}"


def missing_snapshot_files(present: Mapping[str, object] | set[str]) -> tuple[str, ...]:
    """Return the required files absent from a snapshot listing."""

    available = set(present)
    return tuple(name for name in REQUIRED_SNAPSHOT_FILES if name not in available)
