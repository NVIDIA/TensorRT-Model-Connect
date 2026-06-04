"""VoxCPM2 family registration.

VoxCPM2 is a tokenizer-free diffusion/autoregressive TTS stack. This plugin
registers the model family and exposes the generation defaults used by the E2E
contract. Full TRT export requires dedicated builders for LocEnc, TSLM, RALM,
LocDiT, and the AudioVAE waveform decoder; the build boundary fails explicitly
until those engines exist.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...checkpoint_mapper import WeightDict
from ...config import ModelConfig
from . import component_builders as voxcpm2_component_builders


_DEFAULT_SAMPLE_RATE = 48000
_DEFAULT_CFG_VALUE = 2.0
_DEFAULT_INFERENCE_TIMESTEPS = 10
_VOXCPM2_COMPONENTS = tuple(
    spec.name for spec in voxcpm2_component_builders.VOXCPM2_COMPONENT_SPECS
)
_VOXCPM2_ENGINE_SECTIONS = tuple(
    spec.engine_section for spec in voxcpm2_component_builders.VOXCPM2_COMPONENT_SPECS
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
_VOXCPM2_RAW_COMPONENT_CONFIG_KEYS = {
    "locenc": ("encoder_config", "patch_size", "feat_dim"),
    "tslm": ("lm_config", "max_length"),
    "ralm": (
        "lm_config",
        "residual_lm_num_layers",
        "scalar_quantization_latent_dim",
        "scalar_quantization_scale",
    ),
    "locdit": ("dit_config", "dit_config.cfm_config", "patch_size", "feat_dim"),
    "audiovae": (
        "audio_vae_config",
        "audio_vae_config.sample_rate",
        "audio_vae_config.out_sample_rate",
    ),
}


@dataclass(frozen=True)
class VoxCPM2RawComponentSource:
    """Raw checkpoint inputs that a future native component builder consumes."""

    config_keys: tuple[str, ...]
    config_values: dict[str, Any]
    weight_files: tuple[str, ...]


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


def _find_safetensors_files(model_dir: Path) -> tuple[str, ...]:
    files: list[str] = []
    seen: set[str] = set()
    for pattern in ("model.safetensors", "model-*.safetensors", "*.safetensors"):
        for path in sorted(model_dir.glob(pattern)):
            if path.is_file() and path.name not in seen:
                seen.add(path.name)
                files.append(path.name)
    return tuple(files)


def _has_raw_config_key(raw_config: dict[str, Any], key: str) -> bool:
    value: Any = raw_config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _raw_config_get(raw_config: dict[str, Any], key: str) -> Any:
    value: Any = raw_config
    for part in key.split("."):
        value = value[part]
    return copy.deepcopy(value)


def _find_raw_component_sources(
    model_dir: Path, config: ModelConfig
) -> dict[str, VoxCPM2RawComponentSource]:
    raw_config = config.raw if isinstance(config.raw, dict) else {}
    safetensors_files = _find_safetensors_files(model_dir)
    audio_vae_files = ("audiovae.pth",) if (model_dir / "audiovae.pth").is_file() else ()

    sources: dict[str, VoxCPM2RawComponentSource] = {}
    for component in _VOXCPM2_COMPONENTS:
        config_keys = _VOXCPM2_RAW_COMPONENT_CONFIG_KEYS[component]
        if not all(_has_raw_config_key(raw_config, key) for key in config_keys):
            continue

        weight_files = audio_vae_files if component == "audiovae" else safetensors_files
        if not weight_files:
            continue

        sources[component] = VoxCPM2RawComponentSource(
            config_keys=config_keys,
            config_values={
                key: _raw_config_get(raw_config, key) for key in config_keys
            },
            weight_files=weight_files,
        )
    return sources


def _format_raw_component_sources(
    sources: dict[str, VoxCPM2RawComponentSource],
) -> str:
    return "; ".join(
        f"{component}(config: {', '.join(source.config_keys)}; "
        f"weights: {', '.join(source.weight_files)})"
        for component, source in sources.items()
    )


class VoxCPM2Plugin:
    name = "voxcpm2"
    runtime_strategy = "text_to_audio_voxcpm2"
    component_builders = voxcpm2_component_builders.DEFAULT_COMPONENT_BUILDERS

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
        weights["_voxcpm2_raw_component_sources"] = _find_raw_component_sources(
            Path(model_dir), config
        )
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

        raw_sources = weights.get("_voxcpm2_raw_component_sources", {})
        if isinstance(raw_sources, dict) and len(raw_sources) == len(_VOXCPM2_COMPONENTS):
            try:
                return voxcpm2_component_builders.build_voxcpm2_component_plans(
                    model_dir,
                    config,
                    raw_sources,
                    precision=precision,
                    verbose=verbose,
                    builders=self.component_builders,
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    "VoxCPM2 raw checkpoint sources are present for "
                    f"{', '.join(_VOXCPM2_COMPONENTS)}, but native TRT builders "
                    "are incomplete. Builder inputs discovered: "
                    f"{_format_raw_component_sources(raw_sources)}. Full support "
                    "still requires LocEnc, TSLM, RALM, LocDiT, and AudioVAE "
                    "builders plus a native text-to-audio runtime that writes "
                    f"the TRT WAV artifact. First incomplete builder: {exc}"
                ) from exc

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
