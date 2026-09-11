# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time source resolution for the single quantized H3 delivery."""

from __future__ import annotations

from pathlib import Path


COMFY_REPOSITORY = "Comfy-Org/MiniMax-H3"
COMFY_REVISION = "4cc1d817b6184899b41293954329f576cb5ae86b"
SR_SOURCE_SHAPE = (480, 864)
QUANTIZED_SOURCES = {
    "quantized_transformer": "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
    "quantized_ref_transformer": "diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
    "quantized_text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
}


def resolve_quantized_sources(model_dir: Path, options: dict) -> dict[str, Path]:
    """Prefer explicit/offline files, otherwise use the pinned public HF cache."""

    from huggingface_hub import hf_hub_download

    for option in QUANTIZED_SOURCES:
        if options.get(option) and not Path(options[option]).is_file():
            raise FileNotFoundError(f"MiniMax-H3 {option} checkpoint is missing: {options[option]}")
    sources = {}
    for option, filename in QUANTIZED_SOURCES.items():
        explicit = options.get(option)
        local = Path(model_dir) / filename
        if explicit:
            source = Path(explicit).absolute()
        elif local.is_file():
            source = local.absolute()
        else:
            source = Path(hf_hub_download(COMFY_REPOSITORY, filename, revision=COMFY_REVISION))
        if not source.is_file():
            raise FileNotFoundError(f"MiniMax-H3 {option} checkpoint is missing: {source}")
        # Preserve the public filename when HF cache entries are symlinks.
        sources[option] = source.absolute()
    return sources


def resolve_super_resolution_sources(options: dict) -> tuple[Path | None, Path | None]:
    """Only the explicit build flag may enable the fixed-base SR delivery."""

    enabled = options.get("super_resolution", False)
    keys = ("super_resolution_model", "super_resolution_weak_model")
    if not enabled:
        if any(options.get(key) for key in keys):
            raise ValueError("MiniMax-H3 SR checkpoint overrides require super_resolution=true")
        return None, None
    from torch.hub import download_url_to_file, get_dir

    result = []
    for key, filename in zip(keys, ("realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth")):
        explicit = options.get(key)
        path = Path(explicit).absolute() if explicit else Path(get_dir()) / "checkpoints" / filename
        if explicit and not path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 SR checkpoint is missing: {path}")
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            download_url_to_file(
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/" + filename,
                str(path),
            )
        result.append(path.absolute())
    return result[0], result[1]
