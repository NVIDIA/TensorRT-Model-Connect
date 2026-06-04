"""VoxCPM2 family registration.

VoxCPM2 is a tokenizer-free diffusion/autoregressive TTS stack. This plugin
registers the model family and exposes the generation defaults used by the E2E
contract. Full TRT export requires dedicated builders for LocEnc, TSLM, RALM,
LocDiT, and the AudioVAE waveform decoder; the build boundary fails explicitly
until those engines exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig


_DEFAULT_SAMPLE_RATE = 48000
_DEFAULT_CFG_VALUE = 2.0
_DEFAULT_INFERENCE_TIMESTEPS = 10
_VOXCPM2_COMPONENTS = (
    "locenc",
    "tslm",
    "ralm",
    "locdit",
    "audiovae",
)
_VOXCPM2_ENGINE_SECTIONS = tuple(
    f"{component}_engine_plan" for component in _VOXCPM2_COMPONENTS
)
_VOXCPM2_PREBUILT_ENGINE_FILENAMES = {
    component: (
        f"{component}_engine_plan",
        f"{component}_engine_plan.plan",
        f"{component}.plan",
        f"{component}.engine",
        f"{component}.trtplan",
    )
    for component in _VOXCPM2_COMPONENTS
}


def _raw_config_value(config: ModelConfig, key: str, default: Any) -> Any:
    raw = config.raw if isinstance(config.raw, dict) else {}
    return raw.get(key, default)


def _find_prebuilt_component_plans(model_dir: Path) -> dict[str, Path]:
    plans: dict[str, Path] = {}
    for component, candidates in _VOXCPM2_PREBUILT_ENGINE_FILENAMES.items():
        for filename in candidates:
            path = model_dir / filename
            if path.is_file():
                plans[component] = path
                break
    return plans


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
        weights["_voxcpm2_components"] = _VOXCPM2_COMPONENTS
        weights["_voxcpm2_engine_sections"] = _VOXCPM2_ENGINE_SECTIONS
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
    ) -> bytes | dict[str, bytes]:
        model_dir = Path(str(weights.get("_model_dir", "")))
        prebuilt_plans = _find_prebuilt_component_plans(model_dir)
        if len(prebuilt_plans) == len(_VOXCPM2_COMPONENTS):
            return {
                f"{component}_engine_plan": prebuilt_plans[component].read_bytes()
                for component in _VOXCPM2_COMPONENTS
            }

        missing = [
            component
            for component in _VOXCPM2_COMPONENTS
            if component not in prebuilt_plans
        ]
        expected = {
            component: list(_VOXCPM2_PREBUILT_ENGINE_FILENAMES[component])
            for component in missing
        }
        raise NotImplementedError(
            "VoxCPM2 TRT export is not implemented yet. Full support requires "
            "dedicated LocEnc, TSLM, RALM, LocDiT, and AudioVAE TensorRT "
            "builders plus a native text-to-audio runtime that consumes "
            "audio_voxcpm2 settings. This build can package prebuilt native "
            "component plans, but is missing artifacts for "
            f"{', '.join(missing)} under {model_dir}. Expected filenames: "
            f"{expected}."
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
