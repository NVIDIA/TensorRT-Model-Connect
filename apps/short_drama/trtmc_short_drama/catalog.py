# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tiers and the public checkpoints each one serves through trtmc-server."""

from __future__ import annotations

from dataclasses import dataclass, field
import subprocess


@dataclass(frozen=True)
class Model:
    """One bundle that trtmc-server loads for a role of the flow."""

    role: str
    name: str
    checkpoint: str
    build_args: tuple[str, ...] = ()
    # (library file under the runtime root, entry function, kernel name)
    byok: tuple[str, str, str] | None = None
    # "cli" builds with `python -m tensorrt_model_connect build`; anything else
    # names a family recipe that the README documents.
    builder: str = "cli"
    # Speech recognition language baked into the bundle's decoder prompt (Whisper).
    language: str | None = None

    def bundle_name(self, compute_capability: str) -> str:
        language = f"-{self.language}" if self.language else ""
        return f"{self.name}{language}-sm{compute_capability.replace('.', '')}.bundle"


@dataclass(frozen=True)
class Tier:
    name: str
    min_gib: int
    summary: str
    models: dict[str, Model] = field(default_factory=dict)


_WHISPER = Model("asr", "whisper", "openai/whisper-large-v3-turbo", language="zh")

TIERS = {
    "a": Tier("a", 140, "MiniMax H3 reference-to-video with native audio", {
        "llm": Model("llm", "qwen3.6-27b", "Qwen/Qwen3.6-27B", ("--max-sequence-length", "32768")),
        "image": Model("image", "qwen-image", "Qwen/Qwen-Image",
                       ("--image-height", "1664", "--image-width", "928")),
        "video": Model("video", "fasth3", "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
                       byok=("libtrtmc_minimax_h3_omniref_vsa.so", "run", "minimax_h3.omniref_vsa128"),
                       builder="minimax_h3"),
        "asr": _WHISPER,
    }),
    "b": Tier("b", 24, "Wan2.2-TI2V-5B image-to-video with Magpie text-to-speech", {
        "llm": Model("llm", "qwen3.5-4b", "Qwen/Qwen3.5-4B", ("--max-sequence-length", "32768")),
        "image": Model("image", "z-image-turbo", "Tongyi-MAI/Z-Image-Turbo",
                       ("--image-height", "1280", "--image-width", "720")),
        "video": Model("video", "wan2.2-ti2v-5b", "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                       ("--image-height", "1280", "--image-width", "704", "--video-num-frames", "121")),
        "tts": Model("tts", "magpie-tts", "nvidia/magpie_tts_multilingual_357m"),
        "asr": _WHISPER,
    }),
}

# The models each step loads together. Steps not listed need no model.
STAGES = {
    "plan": ("llm",),
    "stills": ("image",),
    "shots": ("video", "tts", "asr"),
    "subtitles": ("asr",),
}


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_gib: float
    compute_capability: str


def gpus() -> list[Gpu]:
    """Visible GPUs as nvidia-smi reports them; empty when there is no driver."""

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [parse_gpu(line) for line in output.splitlines() if line.strip()]


def parse_gpu(line: str) -> Gpu:
    index, name, memory, capability = (part.strip() for part in line.split(","))
    return Gpu(int(index), name, int(memory) / 1024, capability)


def pick_tier(gpu: Gpu, requested: str = "auto") -> Tier:
    if requested != "auto":
        return TIERS[requested]
    for tier in sorted(TIERS.values(), key=lambda tier: -tier.min_gib):
        if gpu.memory_gib >= tier.min_gib:
            return tier
    raise ValueError(f"{gpu.name} has {gpu.memory_gib:.0f} GiB; the smallest tier needs "
                     f"{min(tier.min_gib for tier in TIERS.values())} GiB")
