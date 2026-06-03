"""VoxCPM2 family plugin.

VoxCPM2 is not Bark or Magpie.  Upstream exposes a tokenizer-free
diffusion-autoregressive TTS stack:

    LocEnc -> TSLM -> RALM -> LocDiT -> AudioVAE V2

This plugin provides model detection, config metadata, and a dedicated runtime
strategy so the model fails explicitly until those TensorRT builders/runtime
stages exist.
"""

from __future__ import annotations

from typing import Any

from ...config import ModelConfig


VOXCPM2_RUNTIME_STRATEGY = "text_to_audio_voxcpm2"
VOXCPM2_ARCHITECTURE_STAGES = ("LocEnc", "TSLM", "RALM", "LocDiT", "AudioVAE V2")
VOXCPM2_RUNTIME_LIMITATION = (
    "VoxCPM2 model detection is registered, but TensorRT build/runtime support "
    "is not implemented yet. The upstream model is a tokenizer-free "
    "diffusion-autoregressive TTS stack (LocEnc -> TSLM -> RALM -> LocDiT -> "
    "AudioVAE V2) served through the external voxcpm library; TensorRT-Model-"
    "Connect still needs dedicated builders and a runtime pipeline for those "
    "stages before openbmb/VoxCPM2 can produce audio."
)


def _normalize_model_type(model_type: str) -> str:
    return model_type.strip().lower().replace("-", "_")


def _int_from_mapping(mapping: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


class VoxCPM2Plugin:
    name = "voxcpm2"
    runtime_strategy = VOXCPM2_RUNTIME_STRATEGY

    def matches(self, model_type: str) -> bool:
        return _normalize_model_type(model_type) in {"voxcpm2", "vox_cpm2"}

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> dict[str, Any]:
        del model_dir, precision
        return {
            "_voxcpm2_config": config.raw,
            "_voxcpm2_runtime_limitation": VOXCPM2_RUNTIME_LIMITATION,
        }

    def build_engine(
        self,
        config: ModelConfig,
        weights: dict[str, Any],
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx: Any | None = None,
        verbose: bool = False,
        parallel_config: Any | None = None,
    ) -> bytes:
        del config, weights, max_cache_length, precision, quant_ctx, verbose, parallel_config
        raise NotImplementedError(VOXCPM2_RUNTIME_LIMITATION)

    def get_audio_config(self, config: ModelConfig) -> dict[str, Any]:
        raw = config.raw
        audio_vae = raw.get("audio_vae_config")
        if not isinstance(audio_vae, dict):
            audio_vae = {}

        return {
            "voxcpm2": True,
            "sample_rate": _int_from_mapping(audio_vae, "out_sample_rate", 48000),
            "reference_sample_rate": _int_from_mapping(audio_vae, "sample_rate", 16000),
            "max_length": _int_from_mapping(raw, "max_length", config.max_position_embeddings),
            "dtype": str(raw.get("dtype", "bfloat16")),
            "architecture": str(raw.get("architecture", config.model_type)),
            "architecture_stages": list(VOXCPM2_ARCHITECTURE_STAGES),
            "runtime_limitation": VOXCPM2_RUNTIME_LIMITATION,
        }


plugin = VoxCPM2Plugin()
