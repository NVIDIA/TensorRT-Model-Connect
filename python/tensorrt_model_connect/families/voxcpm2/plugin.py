"""VoxCPM2 family plugin -- detection-only text-to-audio support.

openbmb/VoxCPM2 is a tokenizer-free diffusion autoregressive TTS model:
LocEnc -> TSLM -> RALM -> LocDiT, followed by AudioVAE V2.  It is not a
Bark-style semantic/coarse/fine codebook stack and it is not Magpie's
NeMo/NanoCodec encoder-decoder path.  The upstream runtime is the external
``voxcpm`` package, so this plugin intentionally stops raw TRT builds until
the repo grows a matching LocEnc/TSLM/RALM/LocDiT runtime strategy.
"""

from __future__ import annotations

from pathlib import Path

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


_UNSUPPORTED_RUNTIME_MESSAGE = (
    "VoxCPM2 detection is supported, but raw TensorRT runtime support is not "
    "implemented. openbmb/VoxCPM2 uses the tokenizer-free LocEnc -> TSLM -> "
    "RALM -> LocDiT architecture with AudioVAE V2 and currently runs through "
    "the external voxcpm.VoxCPM API. Add a dedicated text_to_audio_voxcpm2 "
    "runtime before building bundles or adding E2E manifests."
)


def _audio_vae_config(config: ModelConfig) -> dict:
    value = config.raw.get("audio_vae_config", {})
    return value if isinstance(value, dict) else {}


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class VoxCPM2Plugin:
    name = "voxcpm2"
    runtime_strategy = "text_to_audio_voxcpm2"

    def matches(self, model_type: str) -> bool:
        normalized = model_type.lower().replace("-", "_")
        return normalized in {"voxcpm2", "vox_cpm2"}

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        """Record model metadata without reading the 2B BF16 checkpoint.

        The follow-up runtime work needs separate builders for the LocEnc,
        TSLM, RALM, LocDiT, and AudioVAE V2 components. Until then, loading
        checkpoint tensors would only slow the builder before the explicit
        unsupported-runtime error.
        """
        audio_vae = _audio_vae_config(config)
        weights = WeightDict()
        weights["_model_format"] = "voxcpm2"
        weights["_model_dir"] = str(Path(model_dir))
        weights["_requires_python_package"] = "voxcpm"
        weights["_unsupported_runtime_reason"] = _UNSUPPORTED_RUNTIME_MESSAGE
        weights["_precision"] = precision
        weights["_sample_rate"] = _as_int(audio_vae.get("out_sample_rate"), 48000)
        weights["_reference_sample_rate"] = _as_int(audio_vae.get("sample_rate"), 16000)
        weights["_architecture"] = "LocEnc -> TSLM -> RALM -> LocDiT"
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
        raise NotImplementedError(_UNSUPPORTED_RUNTIME_MESSAGE)

    def get_audio_config(self, config: ModelConfig) -> dict:
        audio_vae = _audio_vae_config(config)
        return {
            "voxcpm2": True,
            "sample_rate": _as_int(audio_vae.get("out_sample_rate"), 48000),
            "reference_sample_rate": _as_int(audio_vae.get("sample_rate"), 16000),
            "architecture": "LocEnc -> TSLM -> RALM -> LocDiT",
            "requires_python_package": "voxcpm",
            "runtime_supported": False,
            "runtime_blocker": _UNSUPPORTED_RUNTIME_MESSAGE,
        }


plugin = VoxCPM2Plugin()
