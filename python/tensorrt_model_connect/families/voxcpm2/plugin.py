"""VoxCPM2 family plugin.

openbmb/VoxCPM2 is a tokenizer-free diffusion autoregressive TTS model:
LocEnc -> TSLM -> RALM -> LocDiT, with AudioVAE V2 converting 16 kHz
reference/audio latents to 48 kHz output.  It is served upstream through the
``voxcpm`` Python package rather than the HF Transformers Bark API.

This plugin intentionally stops at detection and metadata extraction.  The
current native TRT text-to-audio runtimes are Bark and Magpie-specific; routing
VoxCPM2 through either would produce an invalid bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


class VoxCPM2Plugin:
    name = "voxcpm2"
    runtime_strategy = "text_to_audio_voxcpm2"

    _SUPPORTED_MODEL_TYPES = {
        "voxcpm2",
        "vox_cpm2",
        "vox-cpm2",
        "openbmb/voxcpm2",
    }

    def matches(self, model_type: str) -> bool:
        mt = str(model_type or "").strip().lower()
        return mt in self._SUPPORTED_MODEL_TYPES

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_path = Path(model_dir)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(
                f"VoxCPM2 requires config.json in {model_dir}"
            )

        raw = json.loads(config_path.read_text())
        architecture = str(raw.get("architecture") or raw.get("model_type") or "")
        if architecture.lower() != "voxcpm2":
            raise ValueError(
                f"Expected VoxCPM2 config architecture 'voxcpm2', got {architecture!r}"
            )

        lm_config = raw.get("lm_config", {})
        encoder_config = raw.get("encoder_config", {})
        dit_config = raw.get("dit_config", {})
        audio_vae_config = raw.get("audio_vae_config", {})

        config.raw.setdefault("model_type", "voxcpm2")
        config.raw.setdefault("runtime_strategy", self.runtime_strategy)
        config.raw["_voxcpm2_architecture"] = {
            "architecture": architecture,
            "lm_hidden_size": int(lm_config.get("hidden_size", 0) or 0),
            "lm_layers": int(lm_config.get("num_hidden_layers", 0) or 0),
            "lm_attention_heads": int(lm_config.get("num_attention_heads", 0) or 0),
            "residual_lm_num_layers": int(raw.get("residual_lm_num_layers", 0) or 0),
            "locenc_layers": int(encoder_config.get("num_layers", 0) or 0),
            "locdit_layers": int(dit_config.get("num_layers", 0) or 0),
            "audio_vae_sample_rate": int(audio_vae_config.get("sample_rate", 0) or 0),
            "audio_vae_out_sample_rate": int(
                audio_vae_config.get("out_sample_rate", 0) or 0
            ),
            "dtype": str(raw.get("dtype", "")),
        }

        weights = WeightDict()
        weights["_model_format"] = "voxcpm2"
        weights["_model_dir"] = str(model_path)
        weights["_config"] = raw
        weights["_requires_python_package"] = "voxcpm"
        weights["_unsupported_reason"] = _unsupported_runtime_message()
        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del config, weights, max_cache_length, precision, quant_ctx, verbose
        raise NotImplementedError(_unsupported_runtime_message())

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        arch = config.raw.get("_voxcpm2_architecture", {})
        return {
            "model_type": "voxcpm2",
            "runtime_strategy": self.runtime_strategy,
            "audio_task": "text_to_audio",
            "voxcpm2": {
                "upstream_library": "voxcpm",
                "upstream_entrypoint": (
                    'VoxCPM.from_pretrained("openbmb/VoxCPM2")'
                ),
                "architecture": "LocEnc -> TSLM -> RALM -> LocDiT",
                "sample_rate": int(arch.get("audio_vae_out_sample_rate", 48000) or 48000),
                "reference_sample_rate": int(
                    arch.get("audio_vae_sample_rate", 16000) or 16000
                ),
                "dtype": str(arch.get("dtype", "bfloat16") or "bfloat16"),
                "unsupported_runtime": _unsupported_runtime_message(),
            },
        }


def _unsupported_runtime_message() -> str:
    return (
        "VoxCPM2 is detected, but native TensorRT runtime support is not "
        "implemented yet. It requires a new text_to_audio_voxcpm2 runtime for "
        "the tokenizer-free LocEnc/TSLM/RALM/LocDiT stack and AudioVAE V2; "
        "the existing Bark and Magpie TTS runtimes are not compatible."
    )


plugin = VoxCPM2Plugin()
