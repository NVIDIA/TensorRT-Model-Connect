"""Load mutation-testing policies and normalize E2E model manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from model_connect_ci.types import MandatoryPolicy, ModelCase, ModelInventory, PolicyBundle


BUCKET_DECODER = "decoder-only LLM"
BUCKET_ENCODER = "encoder / embedding"
BUCKET_ENCODER_DECODER = "encoder-decoder"
BUCKET_VISION = "vision"
BUCKET_DIFFUSION = "diffusion"
BUCKET_SPEECH = "speech"
BUCKET_MULTIMODAL = "multimodal"

_RUNTIME_TO_TASK = {
    "decoder_kv_cache": "text_generation_causal",
    "decoder_static": "text_generation_causal",
    "encoder_only": "encoder_only_nlp",
    "encoder_decoder": "encoder_decoder",
    "seq2seq": "encoder_decoder",
    "diffusion_flux": "diffusion_media_generation",
    "diffusion_wan": "diffusion_media_generation",
    "diffusion_z_image": "diffusion_media_generation",
    "prompted_segmentation": "prompted_segmentation",
}

_DIFFUSION_FAMILIES = {
    "elf_flow",
    "flux",
    "ltx_video",
    "nemotron_labs_diffusion",
    "pixart",
    "qwen_image",
    "wan_t2v",
    "z_image",
}
_SPEECH_FAMILIES = {
    "bark",
    "canary",
    "magpie_tts",
    "nemotron_speech_streaming",
    "personaplex",
    "whisper",
}
_MULTIMODAL_FAMILIES = {
    "deepseek_ocr",
    "eagle_vlm",
    "internvl",
    "lance",
    "nemotron_embed_vl",
    "nemotron_rerank_vl",
    "phi4mm",
    "qwen_vl",
}
_VISION_FAMILIES = {
    "sam",
    "sam3",
    "segformer",
    "timm_vit",
    "yolox",
}
_ENCODER_DECODER_FAMILIES = {
    "bart",
    "marian",
    "m2m_100",
    "nllb",
    "riva_translate",
    "t5",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _tuple_mapping(raw: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(key): tuple(str(item) for item in value or [])
        for key, value in raw.items()
    }


def load_policy_bundle(manifest_dir: Path) -> PolicyBundle:
    """Load all mutation-testing YAML policies."""

    supported_models = _read_yaml(manifest_dir / "supported_hf_models.yaml")
    reference_corpus = _read_yaml(manifest_dir / "reference_corpus.yaml")
    mutation_catalog = _read_yaml(manifest_dir / "mutation_catalog.yaml")
    tolerance_policy = _read_yaml(manifest_dir / "tolerance_policy.yaml")
    mandatory_raw = _read_yaml(manifest_dir / "mandatory_matrix.yaml")
    gate = mandatory_raw.get("gate_thresholds", {})

    mandatory = MandatoryPolicy(
        required_buckets=tuple(str(item) for item in mandatory_raw.get("required_buckets", [])),
        tier_a_models=tuple(str(item) for item in mandatory_raw.get("tier_a_models", [])),
        negative_test_models=_tuple_mapping(mandatory_raw.get("negative_test_models", {})),
        metamorphic_test_models=_tuple_mapping(mandatory_raw.get("metamorphic_test_models", {})),
        assertion_strength_min=float(gate.get("assertion_strength_min", 1.0)),
        negative_test_count_min=int(gate.get("negative_test_count_min", 0)),
        skip_xfail_delta_max=int(gate.get("skip_xfail_delta_max", 0)),
        report_integrity_min=float(gate.get("report_integrity_min", 1.0)),
    )

    return PolicyBundle(
        manifest_dir=manifest_dir,
        supported_models=supported_models,
        reference_corpus=reference_corpus,
        mutation_catalog=mutation_catalog,
        tolerance_policy=tolerance_policy,
        mandatory=mandatory,
    )


def _infer_task_strategy(raw: dict[str, Any]) -> str:
    explicit = raw.get("task_strategy")
    if explicit:
        return str(explicit)
    runtime = str(raw.get("runtime_strategy", "decoder_kv_cache"))
    return _RUNTIME_TO_TASK.get(runtime, runtime)


def _classify_bucket(raw: dict[str, Any], task_strategy: str) -> str:
    family = str(raw.get("family", "")).lower()
    runtime = str(raw.get("runtime_strategy", "")).lower()
    reference_family = str(raw.get("reference_family", "")).lower()
    test_type = str(raw.get("test_type", "")).lower()

    if family in _DIFFUSION_FAMILIES or "diffusion" in task_strategy or "diffusion" in runtime:
        return BUCKET_DIFFUSION
    if family in _SPEECH_FAMILIES or task_strategy in {
        "speech_to_text",
        "speech_to_speech",
        "text_to_audio",
    }:
        return BUCKET_SPEECH
    if (
        family in _MULTIMODAL_FAMILIES
        or task_strategy in {"vision_language_generation", "omni_multimodal"}
        or "vl_" in reference_family
        or "ocr" in reference_family
    ):
        return BUCKET_MULTIMODAL
    if (
        family in _VISION_FAMILIES
        or task_strategy in {
            "image_classification",
            "object_detection",
            "prompted_segmentation",
            "segmentation",
        }
        or test_type in {"prompted_segmentation", "segmentation"}
    ):
        return BUCKET_VISION
    if family in _ENCODER_DECODER_FAMILIES or task_strategy in {
        "encoder_decoder",
        "translation",
    }:
        return BUCKET_ENCODER_DECODER
    if task_strategy in {
        "embedding",
        "encoder_only_nlp",
        "neural_operator",
        "reranking",
    } or "encoder" in runtime:
        return BUCKET_ENCODER
    return BUCKET_DECODER


def _assign_tier(raw: dict[str, Any], name: str, mandatory: MandatoryPolicy) -> str:
    if name in mandatory.tier_a_models:
        return "A"
    ci_tier = str(raw.get("ci_tier", "acceptance"))
    if ci_tier in {"weekly_only", "deep_mutation"}:
        return "C"
    return "B"


def _load_model_case(path: Path, policies: PolicyBundle) -> ModelCase:
    raw = json.loads(path.read_text(encoding="utf-8"))
    name = str(raw.get("name", path.stem))
    task_strategy = _infer_task_strategy(raw)
    bucket = _classify_bucket(raw, task_strategy)
    return ModelCase(
        name=name,
        hf_id=str(raw.get("hf_id") or raw.get("model_id") or name),
        family=str(raw.get("family", "")),
        runtime_strategy=str(raw.get("runtime_strategy", "")),
        task_strategy=task_strategy,
        reference_family=str(raw.get("reference_family", "")),
        ci_tier=str(raw.get("ci_tier", "acceptance")),
        architecture_bucket=bucket,
        tier=_assign_tier(raw, name, policies.mandatory),
        source_path=path,
        raw=raw,
    )


def load_model_inventory(models_dir: Path, policies: PolicyBundle) -> ModelInventory:
    """Load every per-model E2E manifest as the supported-model inventory."""

    if not models_dir.is_dir():
        raise FileNotFoundError(f"models directory not found: {models_dir}")
    models = tuple(
        _load_model_case(path, policies)
        for path in sorted(models_dir.glob("*.json"))
    )
    return ModelInventory(models)
