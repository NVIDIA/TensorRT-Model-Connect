"""Manifest loader — load, validate, and normalize E2E model manifests.

Reads per-model JSON manifests from tests/e2e/models/ and returns
normalized E2ECase dataclass instances. Supports both v1 (current) and
v2 (target) manifest schemas with backward compatibility.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

from .contracts import (
    CILane,
    E2ECase,
    MODEL_REFERENCE_FAMILY,
    OracleLevel,
    PreflightRequirement,
    REFERENCE_FAMILY_TO_USER_CONTRACT,
    RUNTIME_TO_TASK_STRATEGY,
    StageSpec,
)
from .python_profiles import PROFILE_PHASES, normalize_execution_profiles

logger = logging.getLogger(__name__)

# Default manifest directory (relative to project root)
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "e2e" / "models"

# ---------------------------------------------------------------------------
# Reference backend defaults per task strategy
# ---------------------------------------------------------------------------

_DEFAULT_REFERENCE_BACKEND: dict[str, str] = {
    "text_generation_causal": "hf_transformers",
    "vision_language_generation": "hf_transformers",
    "speech_to_text": "hf_transformers",
    "text_to_audio": "hf_transformers",
    "speech_to_speech": "torch_reference",
    "segmentation": "hf_transformers",
    "prompted_segmentation": "hf_transformers",
    "image_classification": "hf_transformers",
    "object_detection": "hf_transformers",
    "diffusion_media_generation": "hf_diffusers",
    "diffusion_text_generation": "invariant_only",
    "embedding": "hf_transformers",
    "reranking": "hf_transformers",
    "encoder_only_nlp": "hf_transformers",
    "neural_operator": "torch_reference",
    "omni_multimodal": "torch_reference",
    "composite_pipeline": "hf_diffusers",
}

_DEFAULT_ORACLE_LEVEL: dict[str, str] = {
    "hf_transformers": OracleLevel.L1_EXTERNAL_REFERENCE.value,
    "hf_diffusers": OracleLevel.L1_EXTERNAL_REFERENCE.value,
    "nemo": OracleLevel.L1_EXTERNAL_REFERENCE.value,
    "torch_reference": OracleLevel.L2_INTERNAL_REFERENCE.value,
    "custom_python": OracleLevel.L2_INTERNAL_REFERENCE.value,
    "golden_snapshot": OracleLevel.L3_SNAPSHOT_REGRESSION.value,
    "invariant_only": OracleLevel.L4_INVARIANTS.value,
}

# ---------------------------------------------------------------------------
# Default stage specs per task strategy
# ---------------------------------------------------------------------------

_DEFAULT_STAGES: dict[str, list[dict[str, Any]]] = {
    "text_generation_causal": [
        {"name": "full_generation", "required": True},
    ],
    "vision_language_generation": [
        {"name": "vision_encode", "required": True},
        {"name": "full_generation", "required": True},
    ],
    "speech_to_text": [
        {"name": "full_generation", "required": True},
    ],
    "text_to_audio": [
        {"name": "full_generation", "required": True},
    ],
    "speech_to_speech": [
        {"name": "full_generation", "required": True},
    ],
    "segmentation": [
        {"name": "full_inference", "required": True},
    ],
    "prompted_segmentation": [
        {"name": "full_inference", "required": True},
    ],
    "image_classification": [
        {"name": "full_inference", "required": True},
    ],
    "object_detection": [
        {"name": "full_inference", "required": True},
    ],
    "diffusion_media_generation": [
        {"name": "t5_encode", "required": False},
        {"name": "dit_step", "required": False},
        {"name": "vae_decode", "required": False},
        {"name": "end_to_end", "required": True},
    ],
    "diffusion_text_generation": [
        {"name": "decoded_text", "required": True},
    ],
    "embedding": [
        {"name": "full_inference", "required": True},
    ],
    "reranking": [
        {"name": "full_inference", "required": True},
    ],
    "encoder_only_nlp": [
        {"name": "full_inference", "required": True},
    ],
    "neural_operator": [
        {"name": "full_inference", "required": True},
    ],
    "omni_multimodal": [
        {"name": "thinker_decode", "required": True},
        {"name": "vision_encode", "required": False},
        {"name": "audio_encode", "required": False},
        {"name": "talker_decode", "required": True},
        {"name": "end_to_end", "required": True},
    ],
    "composite_pipeline": [
        {"name": "end_to_end", "required": True},
    ],
}


# ---------------------------------------------------------------------------
# v1 -> v2 field inference
# ---------------------------------------------------------------------------

def _infer_task_strategy(manifest: dict) -> str:
    """Infer task_strategy from runtime_strategy."""
    if "task_strategy" in manifest:
        return manifest["task_strategy"]
    rs = manifest.get("runtime_strategy", "decoder_kv_cache")
    return RUNTIME_TO_TASK_STRATEGY.get(rs, "text_generation_causal")


def _infer_reference_backend(manifest: dict, task_strategy: str) -> str:
    """Infer reference_backend from manifest or task_strategy defaults."""
    if "reference_backend" in manifest:
        return manifest["reference_backend"]
    return _DEFAULT_REFERENCE_BACKEND.get(task_strategy, "hf_transformers")


def _infer_oracle_level(manifest: dict, reference_backend: str) -> str:
    """Infer oracle_level from reference_backend."""
    if "oracle_level" in manifest:
        return manifest["oracle_level"]
    return _DEFAULT_ORACLE_LEVEL.get(
        reference_backend, OracleLevel.L2_INTERNAL_REFERENCE.value)


def _infer_reference_family(manifest: dict) -> str:
    """Infer reference_family from manifest name or explicit field."""
    if "reference_family" in manifest:
        return manifest["reference_family"]
    return MODEL_REFERENCE_FAMILY.get(manifest.get("name", ""), "")


def _infer_user_contract(manifest: dict, reference_family: str) -> str:
    """Infer user_contract from reference_family or explicit field."""
    if "user_contract" in manifest:
        return manifest["user_contract"]
    if reference_family:
        return REFERENCE_FAMILY_TO_USER_CONTRACT.get(reference_family, "")
    return ""


def _infer_ci_lane(manifest: dict) -> str:
    """Infer ci_lane from manifest or default to acceptance."""
    if "ci_lane" in manifest:
        return manifest["ci_lane"]
    return CILane.ACCEPTANCE.value


def _build_preflight(manifest: dict, task_strategy: str) -> list[PreflightRequirement]:
    """Build preflight requirements from manifest or infer defaults."""
    if "preflight_requirements" in manifest:
        return [
            PreflightRequirement(
                kind=req["kind"],
                args=req.get("args", {}),
                gating=req.get("gating", True),
            )
            for req in manifest["preflight_requirements"]
        ]

    # Auto-generate preflight from manifest fields
    reqs: list[PreflightRequirement] = []

    # All models need the binary
    reqs.append(PreflightRequirement(kind="binary_exists", args={}, gating=True))

    # Vision/segmentation need test image
    if manifest.get("test_image"):
        reqs.append(PreflightRequirement(
            kind="asset_exists",
            args={"path": manifest["test_image"]},
            gating=True,
        ))

    # Audio models need test audio
    if manifest.get("test_input_audio"):
        reqs.append(PreflightRequirement(
            kind="asset_exists",
            args={"path": manifest["test_input_audio"]},
            gating=True,
        ))

    # Speech reference tokens
    if manifest.get("speech_reference_tokens"):
        reqs.append(PreflightRequirement(
            kind="asset_exists",
            args={"path": manifest["speech_reference_tokens"]},
            gating=True,
        ))

    # Diffusion needs diffusers
    if task_strategy == "diffusion_media_generation":
        reqs.append(PreflightRequirement(
            kind="python_module_available",
            args={"module": "diffusers", "phase": "build"},
            gating=True,
        ))
        reqs.append(PreflightRequirement(
            kind="python_module_available",
            args={"module": "ftfy", "phase": "build"},
            gating=True,
        ))

    if task_strategy == "image_classification":
        reqs.append(PreflightRequirement(
            kind="python_module_available",
            args={"module": "timm", "phase": "reference"},
            gating=True,
        ))

    # HF auth for gated models.  trust_remote_code still gets a non-gating
    # diagnostic because many public repos use remote code, but gated repos
    # should skip cleanly when a CI runner has no HF token.
    if manifest.get("gated"):
        reqs.append(PreflightRequirement(
            kind="hf_auth_token_present",
            args={},
            gating=True,
        ))
    elif manifest.get("trust_remote_code"):
        reqs.append(PreflightRequirement(
            kind="hf_auth_token_present",
            args={},
            gating=False,
        ))

    return reqs


def _build_stages(manifest: dict, task_strategy: str) -> list[StageSpec]:
    """Build stage specs from manifest or infer defaults."""
    if "stages" in manifest:
        return [
            StageSpec(
                name=s["name"],
                required=s.get("required", True),
                runner_override=s.get("runner_override"),
                comparator_override=s.get("comparator_override"),
                artifact_type=s.get("artifact_type", ""),
                comparison_mode=s.get("comparison_mode", ""),
                ci_lanes=s.get("ci_lanes", [CILane.ACCEPTANCE.value]),
            )
            for s in manifest["stages"]
        ]

    defaults = _DEFAULT_STAGES.get(task_strategy, [{"name": "full_generation", "required": True}])
    return [
        StageSpec(name=s["name"], required=s.get("required", True))
        for s in defaults
    ]


def _build_inputs(manifest: dict) -> dict:
    """Extract input specification from manifest."""
    inputs: dict[str, Any] = {}

    explicit_inputs = manifest.get("inputs")
    if isinstance(explicit_inputs, dict):
        inputs.update(explicit_inputs)

    # Text prompt
    prompt = manifest.get("prompt") or manifest.get("test_prompt", "")
    if prompt:
        inputs["prompt"] = prompt

    # Image for VL/segmentation
    if manifest.get("test_image"):
        inputs["image"] = manifest["test_image"]

    # Audio input
    if manifest.get("test_input_audio"):
        inputs["audio"] = manifest["test_input_audio"]

    # Point prompts for SAM
    if "point_x" in manifest:
        inputs["point_x"] = manifest["point_x"]
        inputs["point_y"] = manifest["point_y"]
    if "num_expected_masks" in manifest:
        inputs["num_expected_masks"] = manifest["num_expected_masks"]

    # Max new tokens
    inputs["max_new_tokens"] = manifest.get("max_new_tokens", 30)
    inputs["max_cache_length"] = manifest.get(
        "max_cache_length",
        manifest.get("build_args", {}).get("max_cache_length", 256),
    )

    # Generation parameters (optional, default to each runtime's configured value)
    for key in ("temperature", "top_p", "top_k", "min_p", "seed", "guidance_scale"):
        if key in manifest:
            inputs[key] = manifest[key]

    # Diffusion-specific
    if manifest.get("video_num_frames"):
        inputs["video_num_frames"] = manifest["video_num_frames"]
        inputs["video_height"] = manifest.get("video_height", 480)
        inputs["video_width"] = manifest.get("video_width", 832)
        inputs["num_inference_steps"] = manifest.get("num_inference_steps", 30)

    # Image dimensions for image-only diffusion.
    if manifest.get("image_height"):
        inputs["image_height"] = manifest["image_height"]
        inputs["image_width"] = manifest.get("image_width", manifest["image_height"])

    # Qwen-Image (and other image-only diffusion) extras.
    for key in ("negative_prompt", "cfg_scale", "height", "width",
                "num_inference_steps"):
        if key in manifest and key not in inputs:
            inputs[key] = manifest[key]

    # Numeric tensor inputs for neural-operator / one-shot dense models
    for key in ("field_input", "branch_input", "trunk_input", "output_field"):
        if key in manifest:
            inputs[key] = manifest[key]

    return inputs


def _build_threshold_overrides(manifest: dict) -> dict:
    """Extract threshold overrides from manifest."""
    overrides: dict[str, Any] = {}

    if "threshold_overrides" in manifest:
        overrides.update(manifest["threshold_overrides"])

    # v1 compat: map specific fields to threshold overrides
    if "logit_atol" in manifest:
        overrides["logit_atol"] = manifest["logit_atol"]
    if "layer_atol" in manifest:
        overrides["layer_atol"] = manifest["layer_atol"]
    if "min_pixel_agreement" in manifest:
        overrides["min_pixel_agreement"] = manifest["min_pixel_agreement"]
    if "min_pixel_mean" in manifest:
        overrides["min_pixel_mean"] = manifest["min_pixel_mean"]
    if "max_pixel_mean" in manifest:
        overrides["max_pixel_mean"] = manifest["max_pixel_mean"]
    if "min_pixel_std" in manifest:
        overrides["min_pixel_std"] = manifest["min_pixel_std"]
    if "reference_min_pixel_std_for_ratio" in manifest:
        overrides["reference_min_pixel_std_for_ratio"] = (
            manifest["reference_min_pixel_std_for_ratio"])
    if "min_reference_std_ratio" in manifest:
        overrides["min_reference_std_ratio"] = manifest["min_reference_std_ratio"]
    if "speech_min_token_match" in manifest:
        overrides["speech_min_token_match"] = manifest["speech_min_token_match"]
    if "speech_min_frame_exact" in manifest:
        overrides["speech_min_frame_exact"] = manifest["speech_min_frame_exact"]
    if "speech_min_rms" in manifest:
        overrides["speech_min_rms"] = manifest["speech_min_rms"]
    if "num_expected_masks" in manifest:
        overrides["num_expected_masks"] = manifest["num_expected_masks"]

    return overrides


def _build_determinism(manifest: dict) -> dict:
    """Extract determinism settings from manifest."""
    if "determinism" in manifest:
        return manifest["determinism"]

    return {
        "seed": manifest.get("seed", 42),
        "reruns": manifest.get("determinism_reruns", 0),
    }


def _build_execution_profiles(
    manifest: dict,
    *,
    reference_backend: str = "",
) -> dict[str, str]:
    """Extract execution profile selections from the manifest."""
    return normalize_execution_profiles(
        manifest.get("execution_profiles"),
        family=str(manifest.get("family", "") or ""),
        runtime_strategy=str(manifest.get("runtime_strategy", "") or ""),
        reference_backend=reference_backend or str(manifest.get("reference_backend", "") or ""),
    )


def _build_metadata(manifest: dict) -> dict:
    """Collect all non-standard fields into metadata."""
    standard_fields = {
        "name", "hf_id", "model_id", "bundle", "family", "runtime_strategy",
        "task_strategy", "reference_backend", "oracle_level", "prompt",
        "test_prompt", "max_new_tokens", "max_cache_length", "precision",
        "quantization",
        "logit_atol", "layer_atol", "trust_remote_code", "skip",
        "skip_comparison", "test_image",
        "test_input_audio", "speech_reference_tokens", "speech_test_max_frames",
        "speech_min_token_match", "speech_min_frame_exact", "speech_min_rms",
        "point_x", "point_y", "num_expected_masks", "min_pixel_agreement",
        "min_pixel_mean", "max_pixel_mean", "min_pixel_std",
        "reference_min_pixel_std_for_ratio", "min_reference_std_ratio",
        "video_num_frames", "video_height", "video_width",
        "num_inference_steps", "build_args", "preflight_requirements",
        "stages", "comparison_profile", "threshold_overrides", "determinism",
        "inputs", "metadata", "reference_family", "user_contract", "ci_lane",
        "execution_profiles", "temperature", "top_p", "top_k", "min_p", "seed",
        "negative_prompt", "cfg_scale", "height", "width",
        "image_height", "image_width",
    }

    meta = manifest.get("metadata", {}).copy()
    for k, v in manifest.items():
        if k not in standard_fields:
            meta[k] = v

    # Explicitly propagate trust_remote_code to metadata so reference runners
    # can access it via case.metadata["trust_remote_code"].
    if "trust_remote_code" in manifest:
        meta["trust_remote_code"] = manifest["trust_remote_code"]

    # Propagate precision to metadata so the orchestrator can pass it to
    # the trtmc build CLI when building bundles.
    if "precision" in manifest:
        meta["precision"] = manifest["precision"]

    if "quantization" in manifest:
        meta["quantization"] = manifest["quantization"]

    # Propagate build_args so the orchestrator can select the correct backend.
    if "build_args" in manifest:
        meta["build_args"] = manifest["build_args"]

    return meta


def _convert_skip_to_known_limitation(manifest: dict) -> dict | None:
    """Convert v1 skip field to known_limitations entry."""
    skip_reason = manifest.get("skip")
    if skip_reason:
        return {
            "reason": skip_reason,
            "source": "v1_skip_migration",
        }
    return None


# ---------------------------------------------------------------------------
# Manifest schema validation
# ---------------------------------------------------------------------------

_KNOWN_RUNTIME_STRATEGIES = frozenset({
    "decoder_kv_cache",
    "decoder_moe",
    "nemotron_labs_diffusion",
    "ssm_recurrent",
    "rwkv_recurrent",
    "hybrid_mamba_attention",
    "vision_language",
    "speech_to_text",
    "speech_to_text_rnnt",
    "text_to_audio",              # legacy alias
    "text_to_audio_bark",
    "text_to_audio_magpie",
    "speech_to_speech",
    "diffusion",                  # legacy alias
    "diffusion_flux",
    "diffusion_ltx",
    "diffusion_wan",
    "diffusion_zimage",
    "diffusion_qwen_image",
    "diffusion_pixart",
    "segmentation",
    "prompted_segmentation",
    "image_classification",
    "encoder_only",
    "embedding",
    "reranking",
    "text_to_text",
    "marian_translation",
    "seq2seq_encoder_decoder",
    "object_detection",
    "omni_multimodal",
    "neural_operator",
    "elf_flow",
})


def _validate_manifest(raw: dict, path: str) -> None:
    """Validate manifest schema before loading.

    Raises:
        ValueError: If required fields are missing.
        TypeError: If typed fields have wrong types.

    Warns on unknown runtime_strategy values.
    """
    # 1. 'name' is always required
    if "name" not in raw:
        raise ValueError(
            f"Manifest {path!r} is missing required field 'name'"
        )

    # 2. When not skipped, hf_id and family are required
    if not raw.get("skip"):
        if "hf_id" not in raw and "model_id" not in raw:
            raise ValueError(
                f"Manifest {path!r} (name={raw['name']!r}) is missing "
                f"required field 'hf_id' (and no 'skip' is set)"
            )
        if "family" not in raw:
            raise ValueError(
                f"Manifest {path!r} (name={raw['name']!r}) is missing "
                f"required field 'family' (and no 'skip' is set)"
            )

    # 3. Type checks for int fields
    for field_name in ("max_new_tokens", "max_cache_length"):
        if field_name in raw:
            val = raw[field_name]
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(
                    f"Manifest {path!r}: '{field_name}' must be int, "
                    f"got {type(val).__name__} ({val!r})"
                )

    # 4. Warn on unknown runtime_strategy
    rs = raw.get("runtime_strategy")
    if rs is not None and rs not in _KNOWN_RUNTIME_STRATEGIES:
        warnings.warn(
            f"Manifest {path!r} (name={raw.get('name')!r}): unknown "
            f"runtime_strategy {rs!r}. Known values: "
            f"{sorted(_KNOWN_RUNTIME_STRATEGIES)}",
            stacklevel=2,
        )

    execution_profiles = raw.get("execution_profiles")
    if execution_profiles is not None:
        if not isinstance(execution_profiles, dict):
            raise TypeError(
                f"Manifest {path!r}: 'execution_profiles' must be an object, "
                f"got {type(execution_profiles).__name__}"
            )
        for phase, profile in execution_profiles.items():
            if phase not in PROFILE_PHASES:
                raise ValueError(
                    f"Manifest {path!r}: execution_profiles contains unsupported "
                    f"phase {phase!r}; expected one of {PROFILE_PHASES}"
                )
            if not isinstance(profile, str) or not profile.strip():
                raise TypeError(
                    f"Manifest {path!r}: execution_profiles[{phase!r}] must be "
                    f"a non-empty string"
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_manifest(
    manifest_path: str | Path,
) -> E2ECase:
    """Load a single model manifest and return an E2ECase.

    Supports both v1 (current) and v2 (target) manifest formats.
    V1 fields are auto-inferred to fill v2 requirements.
    """
    path = Path(manifest_path)
    with open(path) as f:
        raw = json.load(f)

    _validate_manifest(raw, str(path))

    task_strategy = _infer_task_strategy(raw)
    reference_backend = _infer_reference_backend(raw, task_strategy)
    oracle_level = _infer_oracle_level(raw, reference_backend)

    # Infer reference family, user contract, and CI lane
    reference_family = _infer_reference_family(raw)
    user_contract = _infer_user_contract(raw, reference_family)
    ci_lane = _infer_ci_lane(raw)

    # Handle skip -> known_limitations migration
    known_limitation = _convert_skip_to_known_limitation(raw)
    metadata = _build_metadata(raw)
    if known_limitation:
        metadata["known_limitations"] = [known_limitation]
        metadata["skip_reason"] = raw["skip"]

    # Partial skip: run TRT but skip HF reference/comparison.
    if raw.get("skip_comparison"):
        reason = raw["skip_comparison"]
        if isinstance(reason, bool):
            reason = "comparison skipped"
        metadata["skip_comparison_reason"] = reason

    return E2ECase(
        name=raw["name"],
        hf_id=raw.get("hf_id", raw.get("model_id", "")),
        family=raw.get("family", ""),
        runtime_strategy=raw.get("runtime_strategy", "decoder_kv_cache"),
        task_strategy=task_strategy,
        reference_backend=reference_backend,
        oracle_level=oracle_level,
        reference_family=reference_family,
        user_contract=user_contract,
        ci_lane=ci_lane,
        bundle=raw.get("bundle", f"{raw['name']}.trtfb"),
        inputs=_build_inputs(raw),
        preflight=_build_preflight(raw, task_strategy),
        stages=_build_stages(raw, task_strategy),
        comparison_profile=raw.get("comparison_profile", "default"),
        threshold_overrides=_build_threshold_overrides(raw),
        determinism=_build_determinism(raw),
        execution_profiles=_build_execution_profiles(
            raw,
            reference_backend=reference_backend,
        ),
        metadata=metadata,
    )


def load_all_manifests(
    models_dir: str | Path | None = None,
    task_strategy_filter: str | None = None,
) -> list[E2ECase]:
    """Load all model manifests from a directory.

    Args:
        models_dir: Directory containing *.json manifests.
            Defaults to tests/e2e/models/.
        task_strategy_filter: If set, only return cases matching this
            task_strategy.

    Returns:
        List of E2ECase instances, sorted by name.
    """
    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR

    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        logger.warning("Models directory not found: %s", models_dir)
        return []

    cases: list[E2ECase] = []
    for manifest_path in sorted(models_dir.glob("*.json")):
        try:
            case = load_manifest(manifest_path)
            if task_strategy_filter and case.task_strategy != task_strategy_filter:
                continue
            cases.append(case)
        except Exception as e:
            logger.error("Failed to load manifest %s: %s", manifest_path, e)
            continue

    return cases


def get_case_names(
    models_dir: str | Path | None = None,
    task_strategy_filter: str | None = None,
) -> list[str]:
    """Return list of model case names for pytest parametrization."""
    cases = load_all_manifests(models_dir, task_strategy_filter)
    return [c.name for c in cases]


def get_case_by_name(
    name: str,
    models_dir: str | Path | None = None,
) -> E2ECase | None:
    """Look up a single case by name."""
    cases = load_all_manifests(models_dir)
    for c in cases:
        if c.name == name:
            return c
    return None
