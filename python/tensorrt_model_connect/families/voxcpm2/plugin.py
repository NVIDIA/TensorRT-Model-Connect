"""VoxCPM2 family registration.

VoxCPM2 is a tokenizer-free diffusion/autoregressive TTS stack. This plugin
registers the model family and exposes the generation defaults used by the E2E
contract. Full TRT export requires dedicated builders for the LocEnc, TSLM,
RALM, and LocDiT components; the build boundary fails explicitly until those
engines exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


_DEFAULT_SAMPLE_RATE = 48000
_DEFAULT_CFG_VALUE = 2.0
_DEFAULT_INFERENCE_TIMESTEPS = 10


def _raw_config_value(config: ModelConfig, key: str, default: Any) -> Any:
    raw = config.raw if isinstance(config.raw, dict) else {}
    return raw.get(key, default)


class VoxCPM2Plugin:
    name = "voxcpm2"
    runtime_strategy = "text_to_audio_voxcpm2"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().replace("-", "_") in {"voxcpm2", "vox_cpm2"}

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        """Record metadata needed by the future VoxCPM2 runtime builders."""
        architecture = str(_raw_config_value(config, "architecture", config.model_type))
        if not self.matches(architecture):
            raise ValueError(
                "VoxCPM2 plugin expected architecture/model_type 'voxcpm2', "
                f"got {architecture!r}"
            )

        weights = WeightDict()
        weights["_model_dir"] = str(Path(model_dir))
        weights["_architecture"] = architecture
        weights["_voxcpm2_components"] = ("locenc", "tslm", "ralm", "locdit")
        weights["_sample_rate"] = int(_raw_config_value(config, "sample_rate", _DEFAULT_SAMPLE_RATE))
        weights["_cfg_value"] = float(_raw_config_value(config, "cfg_value", _DEFAULT_CFG_VALUE))
        weights["_inference_timesteps"] = int(
            _raw_config_value(config, "inference_timesteps", _DEFAULT_INFERENCE_TIMESTEPS)
        )
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
        parallel_config=None,
    ) -> bytes:
        raise NotImplementedError(
            "VoxCPM2 TRT export is not implemented yet. Full support requires "
            "dedicated LocEnc, TSLM, RALM, and LocDiT TensorRT builders plus "
            "a native text-to-audio runtime that consumes audio_voxcpm2 settings."
        )

    def get_audio_config(self, config: ModelConfig) -> dict:
        """Return VoxCPM2 audio defaults injected into bundle config.json."""
        return {
            "sample_rate": int(_raw_config_value(config, "sample_rate", _DEFAULT_SAMPLE_RATE)),
            "reference_sample_rate": int(
                _raw_config_value(config, "reference_sample_rate", 16000)
            ),
            "voxcpm2_cfg_value": float(
                _raw_config_value(config, "cfg_value", _DEFAULT_CFG_VALUE)
            ),
            "voxcpm2_inference_timesteps": int(
                _raw_config_value(
                    config, "inference_timesteps", _DEFAULT_INFERENCE_TIMESTEPS
                )
            ),
            "voxcpm2_architecture": str(
                _raw_config_value(config, "architecture", config.model_type or "voxcpm2")
            ),
        }


plugin = VoxCPM2Plugin()
