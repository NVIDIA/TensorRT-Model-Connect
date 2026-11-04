# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tiers and the public checkpoints each one serves through trtmc-server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    # A video model that animates one first frame, without sound, instead of filming each shot from
    # its reference pictures; the tier's "edit" model draws that frame from them, and a tier without
    # one starts the shot on a reference picture itself.
    first_frame: bool = False

    def bundle_name(self, compute_capability: str) -> str:
        language = f"-{self.language}" if self.language else ""
        return f"{self.name}{language}-sm{compute_capability.replace('.', '')}.bundle"

    @property
    def picture_size(self) -> str | None:
        """WIDTHxHEIGHT from the build arguments: what an image bundle renders (an edit bundle: takes)."""

        arguments = list(self.build_args)
        if "--image-width" not in arguments or "--image-height" not in arguments:
            return None
        return f"{arguments[arguments.index('--image-width') + 1]}x{arguments[arguments.index('--image-height') + 1]}"


@dataclass(frozen=True)
class Tier:
    name: str
    min_gib: int
    summary: str
    models: dict[str, Model] = field(default_factory=dict)
    hardware: str = ""  # the GPUs the tier is sized for
    video: str = ""  # how a shot is made
    speech: str = ""  # where the voices come from
    # What TRTMC still lacks before the tier runs end to end; empty when it does.
    missing: tuple[str, ...] = ()
    # The memory size cards are sold with; nvidia-smi reports a little less (a 24 GB card: 23.99 GiB),
    # so `min_gib` sits just under it.
    nominal_gb: int = 0


_WHISPER = Model("asr", "whisper", "openai/whisper-large-v3-turbo", language="zh")

# 121 frames: about 5 s at 24 fps, so a longer shot ends after 5 s.
_WAN = Model("video", "wan2.2-ti2v-5b", "Wan-AI/Wan2.2-TI2V-5B",
             ("--task", "initial_image_text_to_video", "--precision", "bf16", "--image-height", "1280",
              "--image-width", "704", "--video-num-frames", "121"), first_frame=True)

_I2V = "image-to-video for Wan2.2-TI2V-5B in TRTMC (its family renders from text only)"
_EDIT = "vertical pictures from Qwen-Image-Edit in TRTMC (its edit bundles render 1024x1024 only)"
_SILENT = "none yet: each line is a subtitle until TRTMC has Chinese text-to-speech"
# On an RTX PRO 6000 with TensorRT 11.4, TRTMC's fp16 Z-Image-Turbo renders white pictures, also at
# its tested 1024x1024; fp32 renders, but needs about 31 GiB even with one engine loaded at a time.
_ZIMAGE = "Z-Image-Turbo pictures within 24 GB in TRTMC (fp16 renders white, fp32 needs about 31 GiB)"

# Every model here has a permissive license (Apache-2.0 or MIT) except MiniMax H3, whose community
# license excludes some territories; see the README before shipping tier A.
TIERS = {
    "a": Tier("a", 140, "MiniMax H3 reference-to-video with native audio", {
        "llm": Model("llm", "qwen3.6-27b", "Qwen/Qwen3.6-27B", ("--max-sequence-length", "32768")),
        "image": Model("image", "qwen-image", "Qwen/Qwen-Image",
                       ("--precision", "bf16", "--image-height", "1664", "--image-width", "928")),
        "video": Model("video", "fasth3", "FastVideo/FastVideo-FastH3-Omni-8-Step-V1",
                       byok=("libtrtmc_minimax_h3_omniref_vsa.so", "run", "minimax_h3.omniref_vsa128"),
                       builder="minimax_h3"),
        "asr": _WHISPER,
        # Served only with --vision local: looks at the shots during the analysis.
        "vision": Model("vision", "qwen3.5-9b-vl", "Qwen/Qwen3.5-9B",
                        ("--task", "vision_language_generation", "--precision", "fp16",
                         "--max-sequence-length", "32768")),
    }, hardware="GB300, B200, H200", video="MiniMax H3 films each shot from up to 6 reference pictures",
        speech="native audio from the video model", nominal_gb=140),
    "b": Tier("b", 78, "keyframes from the reference pictures and Wan2.2-TI2V-5B image-to-video, without voices", {
        "llm": Model("llm", "qwen3.5-4b", "Qwen/Qwen3.5-4B", ("--max-sequence-length", "32768")),
        "image": Model("image", "qwen-image", "Qwen/Qwen-Image",
                       ("--precision", "bf16", "--image-height", "1664", "--image-width", "928")),
        "edit": Model("edit", "qwen-image-edit", "Qwen/Qwen-Image-Edit-2511",
                      ("--task", "image_edit", "--precision", "bf16", "--image-height", "1280", "--image-width", "704")),
        "video": _WAN,
        "asr": _WHISPER,
    }, hardware="RTX PRO 6000, H100, DGX Spark",
        video="Qwen-Image-Edit draws each shot's first frame from the reference pictures; Wan2.2-TI2V-5B animates it",
        speech=_SILENT, missing=(_I2V, _EDIT), nominal_gb=80),
    # No edit model fits beside the others in 24 GB (Qwen-Image-Edit holds about 65 GB), so a shot
    # starts on its reference picture; FLUX.2 klein 4B could draw first frames once TRTMC takes
    # reference pictures for it.
    "c": Tier("c", 23, "Wan2.2-TI2V-5B image-to-video straight from the reference pictures, without voices", {
        "llm": Model("llm", "qwen3.5-4b", "Qwen/Qwen3.5-4B", ("--max-sequence-length", "32768")),
        # fp16 with the layers TRTMC's own Z-Image test keeps in fp32: the VAE, the noise refiners and
        # the first two blocks.
        "image": Model("image", "z-image-turbo", "Tongyi-MAI/Z-Image-Turbo",
                       ("--precision", "fp16", "--fp32-layer", "2", "--fp32-layer", "3", "--fp32-layer", "4",
                        "--fp32-layer", "7", "--fp32-layer", "8", "--image-height", "1280", "--image-width", "720")),
        "video": _WAN,
        "asr": _WHISPER,
    }, hardware="RTX 4090, RTX 5090, L40S",
        video="Wan2.2-TI2V-5B animates each shot from its speaker's reference picture",
        speech=_SILENT, missing=(_I2V, _ZIMAGE), nominal_gb=24),
}


def describe_tiers() -> list[dict]:
    """The tiers for the page: what each GPU size gets and what is still missing."""

    return [{"name": t.name, "min_gib": t.min_gib, "min_gb": t.nominal_gb or t.min_gib, "hardware": t.hardware, "summary": t.summary,
             "video": t.video, "speech": t.speech, "ready": not t.missing, "missing": list(t.missing)}
            for t in sorted(TIERS.values(), key=lambda tier: -tier.min_gib)]


# The models each step loads together. Steps not listed need no model.
STAGES = {
    "plan": ("llm",),
    "stills": ("image",),
    "shots": ("edit", "video", "asr"),
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
    try:
        gib = int(memory) / 1024
    except ValueError:  # DGX Spark and Jetson Thor share system memory: nvidia-smi says [N/A]
        gib = system_memory_gib()
    return Gpu(int(index), name, gib, capability)


def system_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    """Total system memory, which a unified-memory GPU uses as its own."""

    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def pick_tier(gpu: Gpu, requested: str = "auto") -> Tier:
    if requested != "auto":
        return TIERS[requested]
    for tier in sorted(TIERS.values(), key=lambda tier: -tier.min_gib):
        if gpu.memory_gib >= tier.min_gib:
            return tier
    raise ValueError(f"{gpu.name} has {gpu.memory_gib:.0f} GiB; the smallest tier needs "
                     f"{min(tier.min_gib for tier in TIERS.values())} GiB")
