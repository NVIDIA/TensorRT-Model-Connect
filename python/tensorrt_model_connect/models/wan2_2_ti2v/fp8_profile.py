# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged FP8 scale profile for native Wan2.2 TI2V-5B.

The activation scales were measured over the full 50-step conditional and
unconditional denoising trajectory at 1280x704/121 frames, CFG 5, flow shift
5, and seed 42. Activation amax uses a 10 percent margin before E4M3
normalization; weight scales use per-tensor max-absolute E4M3 normalization.
This fixed map was collected with PyTorch on GB300; it was not produced by
ModelOpt or runtime calibration. Consumption with the official TensorRT
11.1.0.106 release requires a fresh full-resolution Jetson Thor visual
qualification. A visual receipt collected with a different TensorRT build does
not qualify this official-release path.

The data is checkpoint/profile calibration, not a serialized TensorRT plan.
Each target still builds its own TensorRT engines.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from .model_config import WAN22_TI2V_5B, select_generation_profile


PACKAGED_HF_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
PACKAGED_HF_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
PACKAGED_FP8_SCALE_FILENAME = "wan22-ti2v-5b-921dbaf3-fp8-scales.json"
PACKAGED_FP8_SCALE_SHA256 = (
    "a6c0b5b5a450312e804f978bee7be4c45a3c766d0f10297d9c823542328d0b0a"
)
PACKAGED_FP8_SCALE_PATH = Path(__file__).with_name("data") / PACKAGED_FP8_SCALE_FILENAME

# Exact platforms on which Model Connect may consume this packaged map.
PACKAGED_FP8_TARGETS = {
    ((10, 3), False): "GB300 calibration host",
    ((11, 0), True): "Jetson Thor visual-requalification target",
}

# Content identities at the pinned Hugging Face revision. Hashing is a one-time
# build check and prevents a same-shape or locally modified checkpoint from
# silently consuming activation scales measured on different weights.
PACKAGED_CHECKPOINT_FILES = {
    "config.json": (
        "d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc",
        251,
    ),
    "diffusion_pytorch_model.safetensors.index.json": (
        "bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe",
        72_865,
    ),
    "diffusion_pytorch_model-00001-of-00003.safetensors": (
        "720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1",
        9_825_014_472,
    ),
    "diffusion_pytorch_model-00002-of-00003.safetensors": (
        "09ec5ef720d8396f6cfa51fbdcbdb2327e37722afd6e89fd38f1e7e5e782c283",
        9_995_661_736,
    ),
    "diffusion_pytorch_model-00003-of-00003.safetensors": (
        "6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49",
        178_558_176,
    ),
    "Wan2.2_VAE.pth": (
        "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
        2_818_839_170,
    ),
    "models_t5_umt5-xxl-enc-bf16.pth": (
        "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
        11_361_920_418,
    ),
    "google/umt5-xxl/tokenizer.json": (
        "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b",
        16_837_417,
    ),
}


def _current_device_profile() -> tuple[tuple[int, int], bool]:
    # Reuse the family-owned CUDA query used by the VAE precision selector.
    from .vae_step_builder import _current_cuda_device_profile

    return _current_cuda_device_profile()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_contents(model_dir: str) -> None:
    root = Path(model_dir)
    for relative_path, (expected_sha256, expected_size) in PACKAGED_CHECKPOINT_FILES.items():
        checkpoint = root / relative_path
        if not checkpoint.is_file():
            raise ValueError(
                "Wan2.2 packaged FP8 scales require the exact checkpoint files "
                f"from {PACKAGED_HF_MODEL_ID}@{PACKAGED_HF_REVISION}; missing "
                f"{checkpoint}"
            )
        actual_size = checkpoint.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "Wan2.2 checkpoint content does not match the packaged FP8 scale "
                f"provenance for {relative_path}: expected {expected_size} bytes, "
                f"found {actual_size}"
            )
        actual_sha256 = _sha256_file(checkpoint)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Wan2.2 checkpoint content does not match the packaged FP8 scale "
                f"provenance for {relative_path}: expected SHA256 {expected_sha256}, "
                f"found {actual_sha256}"
            )


def _validate_packaged_request(model_dir: str, config) -> str:
    profile = select_generation_profile(config.raw)
    if profile != WAN22_TI2V_5B:
        raise ValueError(
            "Wan2.2 packaged FP8 scales require the official "
            "1280x704, 121-frame, 50-step profile; omit --fp8 for the "
            "reduced BF16 profile"
        )

    compute_capability, integrated = _current_device_profile()
    target = PACKAGED_FP8_TARGETS.get((compute_capability, integrated))
    if target is None:
        enabled = "GB300 SM 10.3 or integrated Jetson Thor SM 11.0"
        raise ValueError(
            "Wan2.2 packaged FP8 scales are enabled only on "
            f"{enabled}; found SM {compute_capability[0]}.{compute_capability[1]} "
            f"(integrated={int(integrated)}). Omit --fp8 to build the portable "
            "BF16 path"
        )
    print(
        "[trtmc build] Verifying pinned Wan2.2 checkpoint contents for FP8 ...",
        file=sys.stderr,
    )
    _validate_checkpoint_contents(model_dir)
    return target


def _load_scale_asset() -> dict[str, dict[str, float]]:
    try:
        payload = PACKAGED_FP8_SCALE_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "The installed Wan2.2 FP8 scale asset is missing: "
            f"{PACKAGED_FP8_SCALE_PATH}"
        ) from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest != PACKAGED_FP8_SCALE_SHA256:
        raise RuntimeError(
            "The installed Wan2.2 FP8 scale asset failed its SHA256 "
            f"check: expected {PACKAGED_FP8_SCALE_SHA256}, found {digest}"
        )
    try:
        scales = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The Wan2.2 FP8 scale asset is not valid JSON") from exc
    if not isinstance(scales, dict):
        raise RuntimeError("The Wan2.2 FP8 scale asset must contain a JSON object")

    expected = {
        name
        for index in range(WAN22_TI2V_5B.num_layers)
        for name in (
            f"blocks.{index}.ffn.net.0.proj",
            f"blocks.{index}.ffn.net.2",
            f"blocks.{index}.attn2.to_q",
            f"blocks.{index}.attn2.to_out.0",
        )
    }
    provided = set(scales)
    if provided != expected:
        raise RuntimeError(
            "The Wan2.2 FP8 scale asset has an unexpected layer map: "
            f"missing={sorted(expected - provided)}, "
            f"unexpected={sorted(provided - expected)}"
        )
    for name, entry in scales.items():
        if not isinstance(entry, dict) or set(entry) != {"input_scale", "weight_scale"}:
            raise RuntimeError(
                f"Wan2.2 FP8 scale entry {name!r} must contain only "
                "input_scale and weight_scale"
            )
        for field, value in entry.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RuntimeError(
                    f"Wan2.2 FP8 scale {name}.{field} must be finite and positive"
                )
    return scales


def load_packaged_fp8_scales(model_dir: str, config) -> dict[str, dict[str, float]]:
    """Validate the request and load the immutable packaged scale map."""

    target = _validate_packaged_request(model_dir, config)
    scales = _load_scale_asset()
    print(
        "[trtmc build] Wan2.2 packaged FP8 profile: "
        f"checkpoint={PACKAGED_HF_REVISION[:8]} target={target} "
        f"asset_sha256={PACKAGED_FP8_SCALE_SHA256}",
        file=sys.stderr,
    )
    return scales
