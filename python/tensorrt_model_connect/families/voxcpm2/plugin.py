"""VoxCPM2 family detection and metadata.

VoxCPM2 is not a Bark or Magpie variant. The upstream checkpoint combines a
MiniCPM-style language model, residual autoregressive modeling, LocDiT
diffusion, and AudioVAE. This plugin intentionally stops before weight loading
so users get an explicit unsupported-runtime error instead of a misleading
decoder or text-to-audio build attempt.
"""

from __future__ import annotations

from typing import Any

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


_UNSUPPORTED_MESSAGE = (
    "VoxCPM2 is detected, but TensorRT runtime support is not implemented yet. "
    "The upstream openbmb/VoxCPM2 architecture requires LocEnc, TSLM/RALM, "
    "LocDiT diffusion, and AudioVAE support; the existing Bark and Magpie TTS "
    "pipelines do not match this checkpoint."
)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VoxCPM2Plugin:
    name = "voxcpm2"
    runtime_strategy = "text_to_audio_voxcpm2"
    runtime_config_namespace = "audio_voxcpm2"

    def matches(self, model_type: str) -> bool:
        return str(model_type or "").lower().replace("-", "_") in {
            "voxcpm2",
            "vox_cpm2",
        }

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE)

    def get_audio_config(self, config: ModelConfig) -> dict:
        raw = config.raw or {}
        lm_config = raw.get("lm_config") if isinstance(raw.get("lm_config"), dict) else {}
        audio_vae = (
            raw.get("audio_vae_config")
            if isinstance(raw.get("audio_vae_config"), dict)
            else {}
        )
        dit_config = raw.get("dit_config") if isinstance(raw.get("dit_config"), dict) else {}

        return {
            "voxcpm2": True,
            "voxcpm2_runtime_supported": False,
            "voxcpm2_architecture": str(raw.get("architecture") or config.model_type),
            "voxcpm2_reference_sample_rate": _as_int(audio_vae.get("sample_rate"), 16000),
            "sample_rate": _as_int(audio_vae.get("out_sample_rate"), 48000),
            "voxcpm2_max_length": _as_int(raw.get("max_length"), 8192),
            "voxcpm2_dtype": str(raw.get("dtype") or "bfloat16"),
            "voxcpm2_lm_hidden_size": _as_int(
                lm_config.get("hidden_size"), config.hidden_size
            ),
            "voxcpm2_lm_layers": _as_int(
                lm_config.get("num_hidden_layers"), config.num_hidden_layers
            ),
            "voxcpm2_dit_hidden_dim": _as_int(dit_config.get("hidden_dim"), 1024),
            "voxcpm2_dit_layers": _as_int(dit_config.get("num_layers"), 12),
        }


plugin = VoxCPM2Plugin()
