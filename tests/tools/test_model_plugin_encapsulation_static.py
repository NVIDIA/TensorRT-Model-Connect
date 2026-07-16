# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static regression checks for model plugin encapsulation boundaries.

Trace: ARCH-MODPLUG-001
Intent: keep model builder, runtime, and E2E ownership independently testable.
Preconditions: model-owned builder/runtime/E2E folders are present.
Postconditions: model folders do not import/include sibling model
implementations, and each model-owned manifest has local E2E sidecars.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODELS = REPO_ROOT / "src" / "runtime" / "models"
FAMILIES = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
E2E_MODELS = REPO_ROOT / "tests" / "e2e" / "models"
CMAKE_ROOT = REPO_ROOT / "CMakeLists.txt"
CONFIG_SCHEMA_CMAKE = REPO_ROOT / "cmake" / "trtmc_config_schemas.cmake"
SHARED_CONFIG_SCHEMAS = REPO_ROOT / "src" / "runtime" / "config" / "schemas"
SHARED_CONFIG_SCHEMA_INCLUDES = REPO_ROOT / "include" / "trtmc" / "config" / "schemas"
PY_RUNTIME_CONFIG_SCHEMAS = (
    REPO_ROOT / "python" / "tensorrt_model_connect" / "runtime_config" / "schemas"
)
BUNDLE_WRITER = REPO_ROOT / "python" / "tensorrt_model_connect" / "bundle_writer.py"
CONFIG_PY = REPO_ROOT / "python" / "tensorrt_model_connect" / "config.py"
CHECKPOINT_MAPPER = REPO_ROOT / "python" / "tensorrt_model_connect" / "checkpoint_mapper.py"
PYTHON_PROFILES = REPO_ROOT / "python" / "tensorrt_model_connect" / "python_profiles.toml"
ENGINE_BUILDER = REPO_ROOT / "python" / "tensorrt_model_connect" / "engine_builder.py"
BUILD_CLI = REPO_ROOT / "python" / "tensorrt_model_connect" / "build_cli.py"
DEBUG_RUNNER = REPO_ROOT / "python" / "tensorrt_model_connect" / "debug_runner.py"
REMOVED_ROOT_GRAPH_HELPERS = (
    REPO_ROOT / "python" / "tensorrt_model_connect" / "graph_ops.py",
    REPO_ROOT / "python" / "tensorrt_model_connect" / "graph_blocks.py",
)
QUANTIZATION = REPO_ROOT / "python" / "tensorrt_model_connect" / "quantization"
DEBUG_RUNNER_TEST = REPO_ROOT / "tests" / "builder" / "test_debug_runner.py"
DEBUG_RUNNER_EXTENDED_TEST = REPO_ROOT / "tests" / "builder" / "test_debug_runner_extended.py"
SHARED_MANIFEST_VALIDATION_TEST = REPO_ROOT / "tests" / "builder" / "test_manifest_validation.py"
FP8_CALIBRATE = REPO_ROOT / "python" / "tensorrt_model_connect" / "fp8_calibrate.py"
SHARED_GENERIC_HELPER_FILES = (
    DEBUG_RUNNER,
    REPO_ROOT / "python" / "tensorrt_model_connect" / "triattention_export.py",
    BUILD_CLI,
    REPO_ROOT / "tests" / "builder" / "test_graph_ops_extended.py",
)
REMOVED_SHARED_BUILDER_FILES = (
    REPO_ROOT / "python" / "tensorrt_model_connect" / "builders" / "__init__.py",
    REPO_ROOT / "python" / "tensorrt_model_connect" / "builders" / "default_decoder.py",
    REPO_ROOT
    / "python"
    / "tensorrt_model_connect"
    / "builders"
    / "default_dual_profile_decoder.py",
    REPO_ROOT
    / "python"
    / "tensorrt_model_connect"
    / "builders"
    / "default_dual_profile_decoder_tp.py",
    REPO_ROOT / "python" / "tensorrt_model_connect" / "builders" / "utils.py",
)
CHAT_TEMPLATE_CORE_FILES = (
    REPO_ROOT / "src" / "runtime" / "core" / "chat_template.h",
    REPO_ROOT / "src" / "runtime" / "core" / "chat_template.cpp",
)
SHARED_CLI_FILES = (
    REPO_ROOT / "src" / "cli" / "main.cpp",
    REPO_ROOT / "src" / "cli" / "args.cpp",
    REPO_ROOT / "src" / "cli" / "args.h",
)
RUNTIME_DOMAINS = REPO_ROOT / "src" / "runtime" / "domains"
RUNTIME_DOMAIN_INCLUDES = REPO_ROOT / "include" / "trtmc" / "runtime" / "domains"
RUNTIME_DIFFUSION_DOMAINS = RUNTIME_DOMAINS / "diffusion"
RUNTIME_AUDIO_DOMAIN_DIR = RUNTIME_DOMAINS / "audio"
VL_RUNTIME_FAMILIES = (
    "qwen_vl",
    "internvl",
    "deepseek_ocr",
    "locateanything",
    "lance",
    "phi4_multimodal",
)
SEGMENTATION_RUNTIME_STRATEGIES = {
    "segformer": "segformer_segmentation",
    "sam": "sam_prompted_segmentation",
    "sam3": "sam3_prompted_segmentation",
}
ENCODER_RUNTIME_STRATEGIES = {
    "albert": "albert_encoder_only",
    "bert": "bert_encoder_only",
    "convbert": "convbert_encoder_only",
    "deberta": "deberta_encoder_only",
    "distilbert": "distilbert_encoder_only",
    "dpr": "dpr_encoder_only",
    "eagle_vlm": ("eagle_vlm_embedding", "eagle_vlm_reranking"),
    "electra": "electra_encoder_only",
    "fnet": "fnet_encoder_only",
    "modernbert": "modernbert_encoder_only",
    "mpnet": "mpnet_encoder_only",
    "roberta": "roberta_encoder_only",
    "xlnet": "xlnet_encoder_only",
}
RECURRENT_RUNTIME_STRATEGIES = {
    "mamba": "mamba_ssm_recurrent",
    "rwkv": "rwkv_recurrent",
    "nemotron_h": "nemotron_h_hybrid_mamba_attention",
    "qwen3_5": "qwen3_5_hybrid_mamba_attention",
}
SPEECH_TO_TEXT_RUNTIME_STRATEGIES = {
    "whisper": "whisper_speech_to_text",
    "canary": "canary_speech_to_text",
}
NEMOTRON_SPEECH_STREAMING_RUNTIME_STRATEGY = "nemotron_speech_streaming_speech_to_text_rnnt"
PERSONAPLEX_RUNTIME_STRATEGY = "personaplex_speech_to_speech"
QWEN3_OMNI_RUNTIME_STRATEGY = "qwen3_omni_multimodal"
PIPELINE_FACTORY = REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp"
RUNTIME_STRATEGY_MATRIX = REPO_ROOT / "tests" / "runtime_strategy_matrix.yaml"
RUNTIME_STRATEGY_DEFAULT_FILES = (
    REPO_ROOT / "include" / "trtmc" / "bundle.h",
    REPO_ROOT / "src" / "bundle" / "bundle_format.cpp",
    REPO_ROOT / "include" / "trtmc" / "runtime" / "pipeline_plugin.h",
    REPO_ROOT / "include" / "trtmc" / "runtime" / "pipeline_registry.h",
    REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp",
    REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_plugin.cpp",
)
E2E_CONTRACTS = REPO_ROOT / "tests" / "e2e_harness" / "contracts.py"
E2E_MANIFEST_LOADER = REPO_ROOT / "tests" / "e2e_harness" / "manifest_loader.py"
E2E_ORCHESTRATOR = REPO_ROOT / "tests" / "e2e_harness" / "orchestrator.py"
E2E_SHARED_HARNESS_FILES = (
    E2E_ORCHESTRATOR,
    REPO_ROOT / "tests" / "e2e_harness" / "plugins" / "segmentation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "hf_transformers.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "segmentation.py",
)
E2E_SHARED_TEXT_GENERATION_RUNNER = (
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "text_generation.py"
)
SHARED_RUNTIME_LEAK_FILES = ()
E2E_SHARED_DIFFUSION_FILES = (
    REPO_ROOT / "tests" / "e2e_harness" / "orchestrator.py",
    E2E_CONTRACTS,
    E2E_MANIFEST_LOADER,
    REPO_ROOT / "scripts" / "generate_e2e_report.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "hf_diffusers.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "diffusion.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "diffusion.py",
    REPO_ROOT / "tests" / "e2e_harness" / "plugins" / "diffusion.py",
)
E2E_SHARED_AUDIO_FILES = (
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "audio_speech.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "text_to_audio.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "torch_reference.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "nemo_reference.py",
)
E2E_SHARED_HF_TRANSFORMERS = (
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "hf_transformers.py"
)
E2E_SHARED_TORCH_REFERENCE = (
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "torch_reference.py"
)
E2E_SHARED_CONTRACT_PLUGINS = REPO_ROOT / "tests" / "e2e_harness" / "plugins"
PERSONAPLEX_E2E_MANIFESTS = (
    E2E_MODELS / "personaplex" / "manifests" / "personaplex-7b.json",
    E2E_MODELS / "personaplex" / "manifests" / "personaplex-7b-l0.json",
    E2E_MODELS / "personaplex" / "manifests" / "personaplex-7b-l0-tp4.json",
)
E2E_SHARED_SEGMENTATION_CONTRACT_FILES = (
    E2E_CONTRACTS,
    E2E_SHARED_CONTRACT_PLUGINS / "segmentation.py",
)
E2E_SHARED_PROMPTED_SEGMENTATION_REFERENCE_FILES = (E2E_SHARED_HF_TRANSFORMERS,)
E2E_SHARED_PROMPTED_SEGMENTATION_RUNTIME_FILES = (
    E2E_ORCHESTRATOR,
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "segmentation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "segmentation.py",
)
E2E_SHARED_DOC_FILES = (
    E2E_CONTRACTS,
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "vision_language.py",
)
E2E_SHARED_PLACEHOLDER_SIDECARS = (
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "audio_speech.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "diffusion.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "diffusion_text_generation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "encoder_only.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "embedding.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "image_classification.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "neural_operator.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "object_detection.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "omni.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "reranking.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "segmentation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "text_generation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "vision_language.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "audio.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "diffusion.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "diffusion_text_generation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "embedding.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "encoder_only.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "image_classification.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "neural_operator.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "omni.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "reranking.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "segmentation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "speech_to_speech.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "speech_to_text.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "text.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "text_to_audio.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "vision_language.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "custom_python.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "golden_snapshot.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "hf_diffusers.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "hf_transformers.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "invariant_only.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "nemo_reference.py",
    REPO_ROOT / "tests" / "e2e_harness" / "references" / "torch_reference.py",
)
TEST_IMPACT = REPO_ROOT / "tools" / "test_impact.py"
SHARED_DIFF_VL_TOOL = REPO_ROOT / "tools" / "diff_vl.py"
SHARED_DIFF_LOGITS_TOOL = REPO_ROOT / "tools" / "diff_logits.py"
SHARED_TOOL_HELPERS = REPO_ROOT / "tools" / "tool_helpers.py"
SHARED_DIFF_LAYERS_TOOL = REPO_ROOT / "tools" / "diff_layers.py"
SHARED_DIFF_AUDIO_TOOL = REPO_ROOT / "tools" / "diff_audio.py"
SHARED_DIFF_T5_TOOL = REPO_ROOT / "tools" / "diff_t5.py"
SHARED_QWEN_AIME_BENCHMARK_TOOL = REPO_ROOT / "tools" / "benchmark_qwen3_8b_aime25_vs_hf.py"
SHARED_QWEN_FLASHINFER_BENCHMARK_TOOL = REPO_ROOT / "tools" / "bench_flashinfer_e2e.py"
AUTOPILOT_DISCOVER = REPO_ROOT / "scripts" / "autopilot" / "discover.py"
AUTOPILOT_AUTORUN = REPO_ROOT / "scripts" / "autopilot" / "autorun.py"
WARM_HF_CACHE = REPO_ROOT / "scripts" / "warm_hf_cache.py"
SHARED_FAMILY_REGISTRY_TEST = REPO_ROOT / "tests" / "builder" / "test_families.py"
SHARED_GENERIC_FIXTURE_TEST_FILES = (
    REPO_ROOT / "tests" / "builder" / "test_bundle_writer.py",
    REPO_ROOT / "tests" / "builder" / "test_cli.py",
    REPO_ROOT / "tests" / "builder" / "test_cli_coverage.py",
    REPO_ROOT / "tests" / "builder" / "test_config.py",
    REPO_ROOT / "tests" / "builder" / "test_config_coverage.py",
    REPO_ROOT / "tests" / "builder" / "test_quantization.py",
    REPO_ROOT / "tests" / "builder" / "test_parallel_config.py",
    REPO_ROOT / "tests" / "builder" / "test_trt_compat_boundary.py",
    REPO_ROOT / "tests" / "builder" / "test_triattention_export.py",
)
ROOT_MODEL_SCRIPT_WRAPPERS = {
    REPO_ROOT / "scripts" / "magpie_tokenizer.py": (
        FAMILIES / "magpie_tts" / "magpie_tokenizer.py"
    ),
    REPO_ROOT / "scripts" / "magpie_codec_bridge.py": (FAMILIES / "magpie_tts" / "codec_bridge.py"),
    REPO_ROOT / "scripts" / "profile_magpie_tts.py": (FAMILIES / "magpie_tts" / "profile.py"),
    REPO_ROOT / "scripts" / "prepare_lance_model.py": (FAMILIES / "lance" / "prepare_model.py"),
    REPO_ROOT / "scripts" / "_build_fp8_onnx_monolithic.py": (
        FAMILIES / "flux" / "build_fp8_onnx_monolithic.py"
    ),
    REPO_ROOT / "scripts" / "_inject_fp8_qdq_proto.py": (
        FAMILIES / "flux" / "inject_fp8_qdq_proto.py"
    ),
    REPO_ROOT / "scripts" / "_mk_fp8_bf16_bundle.py": (FAMILIES / "flux" / "mk_fp8_bf16_bundle.py"),
    REPO_ROOT / "tools" / "diff_personaplex.py": (FAMILIES / "personaplex" / "diff_personaplex.py"),
}
MODEL_OWNED_BUILDER_TESTS = {
    REPO_ROOT / "tests" / "builder" / "test_engine_qwen.py": (
        E2E_MODELS / "qwen" / "test_qwen_builder_engine.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_engine_qwen_vl_tp.py": (
        E2E_MODELS / "qwen_vl" / "test_qwen_vl_builder_tp.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_engine_qwen_moe_tp.py": (
        E2E_MODELS / "qwen_moe" / "test_qwen_moe_builder_tp.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_qwen_image_dit_batch_profile.py": (
        E2E_MODELS / "qwen_image" / "test_qwen_image_dit_batch_profile.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_flux_dit_batch_profile.py": (
        E2E_MODELS / "flux" / "test_flux_dit_batch_profile.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_z_image_dit_batch_profile.py": (
        E2E_MODELS / "z_image" / "test_z_image_dit_batch_profile.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_standard_decoder.py": (
        E2E_MODELS / "llama" / "test_llama_standard_decoder.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_bark_tokenizer.py": (
        E2E_MODELS / "bark" / "test_bark_tokenizer.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_engine_magpie_tts.py": (
        E2E_MODELS / "magpie_tts" / "test_magpie_tts_builder_tp.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_engine_nemotron_speech_streaming.py": (
        E2E_MODELS / "nemotron_speech_streaming" / "test_nemotron_speech_streaming_builder.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_engine_nemotron_speech_streaming_tp.py": (
        E2E_MODELS / "nemotron_speech_streaming" / "test_nemotron_speech_streaming_builder_tp.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_family_qwen_moe.py": (
        E2E_MODELS / "qwen_moe" / "test_qwen_moe_family_plugin.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_family_qwen3_5.py": (
        E2E_MODELS / "qwen3_5" / "test_qwen3_5_family_plugin.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_vision_compute.py": (
        E2E_MODELS / "qwen_vl" / "test_qwen_vl_vision_compute.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_vision_compute_extended.py": (
        E2E_MODELS / "qwen_vl" / "test_qwen_vl_vision_compute_extended.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_owned_qwen3_t5_helpers.py": (
        E2E_MODELS / "z_image" / "test_z_image_qwen3_t5_helpers.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_owned_builder_mocked_paths.py": (
        E2E_MODELS / "bark" / "test_bark_encodec_builder.py",
        E2E_MODELS / "bert" / "test_bert_encoder_builder_mocked.py",
        E2E_MODELS / "qwen_vl" / "test_qwen_vl_onnx_vision_builder.py",
    ),
    REPO_ROOT / "tests" / "builder" / "test_owned_encoder_builders_coverage.py": (
        E2E_MODELS / "flux" / "test_flux_owned_encoder_builders_coverage.py",
        E2E_MODELS / "bert" / "test_bert_owned_encoder_builders_coverage.py",
        E2E_MODELS / "z_image" / "test_z_image_qwen3_encoder_builder.py",
    ),
    REPO_ROOT / "tests" / "builder" / "test_family_z_image.py": (
        E2E_MODELS / "z_image" / "test_z_image_family_plugin.py"
    ),
}
MODEL_OWNED_BUILDER_TESTS.update(
    {
        REPO_ROOT / "tests" / "builder" / "test_engine_internvl_tp.py": E2E_MODELS
        / "internvl"
        / "test_internvl_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_xlnet_tp.py": E2E_MODELS
        / "xlnet"
        / "test_xlnet_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_t5_tp.py": E2E_MODELS
        / "t5"
        / "test_t5_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_segformer_tp.py": E2E_MODELS
        / "segformer"
        / "test_segformer_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_sam_tp.py": E2E_MODELS
        / "sam"
        / "test_sam_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_rwkv_tp.py": E2E_MODELS
        / "rwkv"
        / "test_rwkv_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_roberta_tp.py": E2E_MODELS
        / "roberta"
        / "test_roberta_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_phi_moe_tp.py": E2E_MODELS
        / "phi_moe"
        / "test_phi_moe_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_personaplex_tp.py": E2E_MODELS
        / "personaplex"
        / "test_personaplex_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_opt_tp.py": E2E_MODELS
        / "opt"
        / "test_opt_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_olmo_tp.py": E2E_MODELS
        / "olmo"
        / "test_olmo_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_olmo2_tp.py": E2E_MODELS
        / "olmo2"
        / "test_olmo2_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_mpnet_tp.py": E2E_MODELS
        / "mpnet"
        / "test_mpnet_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_modernbert_tp.py": E2E_MODELS
        / "modernbert"
        / "test_modernbert_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_mixtral_tp.py": E2E_MODELS
        / "mixtral"
        / "test_mixtral_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_marian_tp.py": E2E_MODELS
        / "marian"
        / "test_marian_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_mamba_tp.py": E2E_MODELS
        / "mamba"
        / "test_mamba_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_granite_tp.py": E2E_MODELS
        / "granite"
        / "test_granite_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_gpt_oss_tp.py": E2E_MODELS
        / "gpt_oss"
        / "test_gpt_oss_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_gemma_tp.py": E2E_MODELS
        / "gemma"
        / "test_gemma_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_fnet_tp.py": E2E_MODELS
        / "fnet"
        / "test_fnet_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_falcon_tp.py": E2E_MODELS
        / "falcon"
        / "test_falcon_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_electra_tp.py": E2E_MODELS
        / "electra"
        / "test_electra_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_eagle_vlm_tp.py": E2E_MODELS
        / "eagle_vlm"
        / "test_eagle_vlm_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_dpr_tp.py": E2E_MODELS
        / "dpr"
        / "test_dpr_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_distilbert_tp.py": E2E_MODELS
        / "distilbert"
        / "test_distilbert_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_deepseek_v2_tp.py": E2E_MODELS
        / "deepseek_v2"
        / "test_deepseek_v2_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_deepseek_ocr_tp.py": E2E_MODELS
        / "deepseek_ocr"
        / "test_deepseek_ocr_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_deberta_tp.py": E2E_MODELS
        / "deberta"
        / "test_deberta_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_convbert_tp.py": E2E_MODELS
        / "convbert"
        / "test_convbert_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_bloom_tp.py": E2E_MODELS
        / "bloom"
        / "test_bloom_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_bert_tp.py": E2E_MODELS
        / "bert"
        / "test_bert_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_bart_tp.py": E2E_MODELS
        / "bart"
        / "test_bart_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_albert_tp.py": E2E_MODELS
        / "albert"
        / "test_albert_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_xglm.py": E2E_MODELS
        / "xglm"
        / "test_xglm_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_whisper.py": E2E_MODELS
        / "whisper"
        / "test_whisper_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_starcoder2.py": E2E_MODELS
        / "starcoder2"
        / "test_starcoder2_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_stablelm.py": E2E_MODELS
        / "stablelm"
        / "test_stablelm_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_segformer.py": E2E_MODELS
        / "segformer"
        / "test_segformer_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_rwkv.py": E2E_MODELS
        / "rwkv"
        / "test_rwkv_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_phi_moe.py": E2E_MODELS
        / "phi_moe"
        / "test_phi_moe_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_phi.py": E2E_MODELS
        / "phi"
        / "test_phi_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_opt.py": E2E_MODELS
        / "opt"
        / "test_opt_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_olmo.py": E2E_MODELS
        / "olmo"
        / "test_olmo_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_nemotron.py": E2E_MODELS
        / "nemotron"
        / "test_nemotron_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_mixtral.py": E2E_MODELS
        / "mixtral"
        / "test_mixtral_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_mistral.py": E2E_MODELS
        / "mistral"
        / "test_mistral_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_mamba.py": E2E_MODELS
        / "mamba"
        / "test_mamba_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_llama.py": E2E_MODELS
        / "llama"
        / "test_llama_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_internlm.py": E2E_MODELS
        / "internlm"
        / "test_internlm_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_granite.py": E2E_MODELS
        / "granite"
        / "test_granite_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_gpt_neox.py": E2E_MODELS
        / "gpt_neox"
        / "test_gpt_neox_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_gpt_neo.py": E2E_MODELS
        / "gpt_neo"
        / "test_gpt_neo_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_gpt2.py": E2E_MODELS
        / "gpt2"
        / "test_gpt2_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_gemma.py": E2E_MODELS
        / "gemma"
        / "test_gemma_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_falcon.py": E2E_MODELS
        / "falcon"
        / "test_falcon_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_codegen.py": E2E_MODELS
        / "codegen"
        / "test_codegen_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_bloom.py": E2E_MODELS
        / "bloom"
        / "test_bloom_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_engine_bark.py": E2E_MODELS
        / "bark"
        / "test_bark_builder_engine.py",
        REPO_ROOT / "tests" / "builder" / "test_family_marian_debug_runner.py": E2E_MODELS
        / "marian"
        / "test_marian_debug_runner.py",
        REPO_ROOT / "tests" / "builder" / "test_family_sam3.py": E2E_MODELS
        / "sam3"
        / "test_sam3_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_timm_vit.py": E2E_MODELS
        / "timm_vit"
        / "test_timm_vit_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_nemotron_h_tp.py": E2E_MODELS
        / "nemotron_h"
        / "test_nemotron_h_builder_tp.py",
        REPO_ROOT / "tests" / "builder" / "test_family_yolox.py": E2E_MODELS
        / "yolox"
        / "test_yolox_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_wan_t2v.py": E2E_MODELS
        / "wan_t2v"
        / "test_wan_t2v_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_sam.py": E2E_MODELS
        / "sam"
        / "test_sam_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_roberta.py": E2E_MODELS
        / "roberta"
        / "test_roberta_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_pixart.py": E2E_MODELS
        / "pixart"
        / "test_pixart_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_phi4mm.py": E2E_MODELS
        / "phi4_multimodal"
        / "test_phi4_multimodal_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_nemotron_h.py": E2E_MODELS
        / "nemotron_h"
        / "test_nemotron_h_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_mpnet.py": E2E_MODELS
        / "mpnet"
        / "test_mpnet_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_ltx_video.py": E2E_MODELS
        / "ltx_video"
        / "test_ltx_video_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_gpt_oss.py": E2E_MODELS
        / "gpt_oss"
        / "test_gpt_oss_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_glm.py": E2E_MODELS
        / "glm"
        / "test_glm_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_distilbert.py": E2E_MODELS
        / "distilbert"
        / "test_distilbert_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_deepseek_v2.py": E2E_MODELS
        / "deepseek_v2"
        / "test_deepseek_v2_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_bert.py": E2E_MODELS
        / "bert"
        / "test_bert_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_family_elf.py": E2E_MODELS
        / "elf_flow"
        / "test_elf_flow_family_plugin.py",
        REPO_ROOT / "tests" / "builder" / "test_build_engine_std_decoders.py": (
            E2E_MODELS / "gpt2" / "test_gpt2_build_engine_integration.py",
            E2E_MODELS / "gpt_neo" / "test_gpt_neo_build_engine_integration.py",
            E2E_MODELS / "gpt_neox" / "test_gpt_neox_build_engine_integration.py",
            E2E_MODELS / "internlm" / "test_internlm_build_engine_integration.py",
            E2E_MODELS / "codegen" / "test_codegen_build_engine_integration.py",
        ),
        REPO_ROOT / "tests" / "builder" / "test_build_engine_decoders.py": (
            E2E_MODELS / "t5" / "test_t5_build_engine_integration.py",
            E2E_MODELS / "convbert" / "test_convbert_build_engine_integration.py",
            E2E_MODELS / "dpr" / "test_dpr_build_engine_integration.py",
            E2E_MODELS / "distilbert" / "test_distilbert_build_engine_integration.py",
        ),
        REPO_ROOT / "tests" / "builder" / "test_build_engine_enc_dec.py": (
            E2E_MODELS / "bart" / "test_bart_build_engine_integration.py",
            E2E_MODELS / "m2m_100" / "test_m2m_100_build_engine_integration.py",
        ),
        REPO_ROOT / "tests" / "builder" / "test_build_engine_integration.py": (
            E2E_MODELS / "olmo2" / "test_olmo2_build_engine_integration.py",
            E2E_MODELS / "modernbert" / "test_modernbert_build_engine_integration.py",
            E2E_MODELS / "deberta" / "test_deberta_build_engine_integration.py",
            E2E_MODELS / "electra" / "test_electra_build_engine_integration.py",
            E2E_MODELS / "fnet" / "test_fnet_build_engine_integration.py",
            E2E_MODELS / "albert" / "test_albert_build_engine_integration.py",
            E2E_MODELS / "xlnet" / "test_xlnet_build_engine_integration.py",
        ),
        REPO_ROOT / "tests" / "builder" / "test_engine_gpt_tp.py": (
            E2E_MODELS / "gpt2" / "test_gpt2_builder_tp_dispatch.py",
            E2E_MODELS / "gpt_neo" / "test_gpt_neo_builder_tp_dispatch.py",
            E2E_MODELS / "gpt_neox" / "test_gpt_neox_builder_tp_dispatch.py",
        ),
        REPO_ROOT / "tests" / "builder" / "test_engine_decoder_family_tp.py": (
            E2E_MODELS / "codegen" / "test_codegen_builder_tp.py",
            E2E_MODELS / "glm" / "test_glm_builder_tp.py",
            E2E_MODELS / "internlm" / "test_internlm_builder_tp.py",
            E2E_MODELS / "phi" / "test_phi_builder_tp.py",
            E2E_MODELS / "stablelm" / "test_stablelm_builder_tp.py",
            E2E_MODELS / "starcoder2" / "test_starcoder2_builder_tp.py",
            E2E_MODELS / "xglm" / "test_xglm_builder_tp.py",
        ),
    }
)
SHARED_PLUGIN_WEIGHT_TEST_FILES = (
    REPO_ROOT / "tests" / "builder" / "test_family_plugins.py",
    REPO_ROOT / "tests" / "builder" / "test_family_plugins_extended.py",
    REPO_ROOT / "tests" / "builder" / "test_family_plugins_extended2.py",
)
MODEL_OWNED_PLUGIN_WEIGHT_TESTS = {
    "TestQwenPlugin": E2E_MODELS / "qwen" / "test_qwen_family_plugin_weights.py",
    "TestNemotronLabsDiffusionPlugin": (
        E2E_MODELS
        / "nemotron_labs_diffusion"
        / "test_nemotron_labs_diffusion_family_plugin_weights.py"
    ),
    "TestGemmaPlugin": E2E_MODELS / "gemma" / "test_gemma_family_plugin_weights.py",
    "TestPhiPlugin": E2E_MODELS / "phi" / "test_phi_family_plugin_weights.py",
    "TestFalconPlugin": E2E_MODELS / "falcon" / "test_falcon_family_plugin_weights.py",
    "TestMambaPlugin": E2E_MODELS / "mamba" / "test_mamba_family_plugin_weights.py",
    "TestMixtralPlugin": E2E_MODELS / "mixtral" / "test_mixtral_family_plugin_weights.py",
    "TestBloomPlugin": E2E_MODELS / "bloom" / "test_bloom_family_plugin_weights.py",
    "TestLlamaPlugin": E2E_MODELS / "llama" / "test_llama_family_plugin_weights.py",
    "TestDeepSeekV2Plugin": (
        E2E_MODELS / "deepseek_v2" / "test_deepseek_v2_family_plugin_weights.py"
    ),
    "TestQwenVLPlugin": E2E_MODELS / "qwen_vl" / "test_qwen_vl_family_plugin_weights.py",
    "TestInternVLPlugin": E2E_MODELS / "internvl" / "test_internvl_family_plugin_weights.py",
    "TestLocateAnythingPlugin": (
        E2E_MODELS / "locateanything" / "test_locateanything_family_plugin_weights.py"
    ),
    "TestEagleVLMPlugin": E2E_MODELS / "eagle_vlm" / "test_eagle_vlm_family_plugin_weights.py",
    "TestGlmPlugin": E2E_MODELS / "glm" / "test_glm_family_plugin_weights.py",
    "TestCanaryPlugin": E2E_MODELS / "canary" / "test_canary_family_plugin_weights.py",
    "TestT5Plugin": E2E_MODELS / "t5" / "test_t5_family_plugin_weights.py",
    "TestBartPlugin": E2E_MODELS / "bart" / "test_bart_family_plugin_weights.py",
    "TestOlmo2Plugin": E2E_MODELS / "olmo2" / "test_olmo2_family_plugin_weights.py",
    "TestModernbertPlugin": (
        E2E_MODELS / "modernbert" / "test_modernbert_family_plugin_weights.py"
    ),
    "TestDebertaPlugin": E2E_MODELS / "deberta" / "test_deberta_family_plugin_weights.py",
    "TestElectraPlugin": E2E_MODELS / "electra" / "test_electra_family_plugin_weights.py",
    "TestFNetPlugin": E2E_MODELS / "fnet" / "test_fnet_family_plugin_weights.py",
    "TestDPRPlugin": E2E_MODELS / "dpr" / "test_dpr_family_plugin_weights.py",
    "TestConvBERTPlugin": E2E_MODELS / "convbert" / "test_convbert_family_plugin_weights.py",
    "TestXLNetPlugin": E2E_MODELS / "xlnet" / "test_xlnet_family_plugin_weights.py",
    "TestAlbertPlugin": E2E_MODELS / "albert" / "test_albert_family_plugin_weights.py",
    "TestM2M100SinusoidalPosEmbed": (
        E2E_MODELS / "m2m_100" / "test_m2m_100_family_plugin_weights.py"
    ),
    "TestM2M100Plugin": E2E_MODELS / "m2m_100" / "test_m2m_100_family_plugin_weights.py",
    "TestMarianPlugin": E2E_MODELS / "marian" / "test_marian_family_plugin_weights.py",
    "TestOlmoPlugin": E2E_MODELS / "olmo" / "test_olmo_family_plugin_weights.py",
    "TestStablelmPlugin": (E2E_MODELS / "stablelm" / "test_stablelm_family_plugin_weights.py"),
    "TestStarcoder2Plugin": (
        E2E_MODELS / "starcoder2" / "test_starcoder2_family_plugin_weights.py"
    ),
    "TestGranitePlugin": E2E_MODELS / "granite" / "test_granite_family_plugin_weights.py",
    "TestXglmPlugin": E2E_MODELS / "xglm" / "test_xglm_family_plugin_weights.py",
}
MODEL_OWNED_REGISTRY_CONTRACT_TESTS = {
    "mamba": E2E_MODELS / "mamba" / "test_mamba_registry_contract.py",
    "mixtral": E2E_MODELS / "mixtral" / "test_mixtral_registry_contract.py",
    "gpt_oss": E2E_MODELS / "gpt_oss" / "test_gpt_oss_registry_contract.py",
    "qwen_vl": E2E_MODELS / "qwen_vl" / "test_qwen_vl_registry_contract.py",
    "internvl": E2E_MODELS / "internvl" / "test_internvl_registry_contract.py",
    "locateanything": (E2E_MODELS / "locateanything" / "test_locateanything_registry_contract.py"),
    "qwen3_omni": E2E_MODELS / "qwen3_omni" / "test_qwen3_omni_registry_contract.py",
    "personaplex": E2E_MODELS / "personaplex" / "test_personaplex_registry_contract.py",
    "nemotron_h": E2E_MODELS / "nemotron_h" / "test_nemotron_h_registry_contract.py",
    "canary": E2E_MODELS / "canary" / "test_canary_registry_contract.py",
    "nemotron_speech_streaming": (
        E2E_MODELS
        / "nemotron_speech_streaming"
        / "test_nemotron_speech_streaming_registry_contract.py"
    ),
    "patchtst": E2E_MODELS / "patchtst" / "test_patchtst_registry_contract.py",
    "patchtsmixer": E2E_MODELS / "patchtsmixer" / "test_patchtsmixer_registry_contract.py",
    "timesfm": E2E_MODELS / "timesfm" / "test_timesfm_registry_contract.py",
    "chronos_bolt": E2E_MODELS / "chronos_bolt" / "test_chronos_bolt_registry_contract.py",
    "qwen": E2E_MODELS / "qwen" / "test_qwen_registry_contract.py",
    "llama": E2E_MODELS / "llama" / "test_llama_registry_contract.py",
    "mistral": E2E_MODELS / "mistral" / "test_mistral_registry_contract.py",
    "gemma": E2E_MODELS / "gemma" / "test_gemma_registry_contract.py",
    "phi": E2E_MODELS / "phi" / "test_phi_registry_contract.py",
    "gpt2": E2E_MODELS / "gpt2" / "test_gpt2_registry_contract.py",
    "opt": E2E_MODELS / "opt" / "test_opt_registry_contract.py",
    "phi_moe": E2E_MODELS / "phi_moe" / "test_phi_moe_registry_contract.py",
}
SHARED_TIMM_VIT_TRT_PATH_TOOL = (
    REPO_ROOT / "tools" / "validation" / "timm_vit" / "benchmark_trt_paths.py"
)
SHARED_RUNNER_PARITY_TOOL = REPO_ROOT / "tools" / "test_runner_parity.py"
SHARED_PERF_COMPARE_TOOL = REPO_ROOT / "tools" / "perf_compare.py"
SHARED_CPU_PROFILE_TOOL = REPO_ROOT / "tools" / "cpu_profile.py"
SHARED_CPU_PROFILE_MATRIX_TOOL = REPO_ROOT / "tools" / "cpu_profile_matrix.py"
SHARED_TRTMC_PROFILE_TOOL = REPO_ROOT / "tools" / "trtmc_profile.py"
SHARED_PERF_PROFILE_TEST_FILES = (
    REPO_ROOT / "tests" / "tools" / "test_profile.py",
    REPO_ROOT / "tests" / "tools" / "test_perf_compare.py",
    REPO_ROOT / "tests" / "tools" / "test_perfdb.py",
    REPO_ROOT / "tests" / "tools" / "test_profile_report.py",
    REPO_ROOT / "tests" / "tools" / "test_cpu_profile.py",
)
SHARED_REPORT_AND_IMPACT_TEST_FILES = (
    REPO_ROOT / "tests" / "test_e2e.py",
    REPO_ROOT / "tests" / "test_e2e_selection.py",
    REPO_ROOT / "tests" / "tools" / "test_generate_report.py",
    REPO_ROOT / "tests" / "tools" / "test_generate_ci_summary.py",
    REPO_ROOT / "tests" / "tools" / "test_layer_profiler.py",
    REPO_ROOT / "tests" / "tools" / "test_e2e_detailed_timing.py",
    REPO_ROOT / "tests" / "tools" / "test_test_impact.py",
    REPO_ROOT / "tests" / "tools" / "test_count_diffusion_frame_pairs.py",
    REPO_ROOT / "tests" / "tools" / "test_diff_vl.py",
    REPO_ROOT / "tests" / "tools" / "test_e2e_origin_main_parity.py",
    REPO_ROOT / "tests" / "tools" / "test_perf_evolve_prompt.py",
    REPO_ROOT / "tests" / "tools" / "test_sampling_contract_plugin.py",
    REPO_ROOT / "tests" / "tools" / "test_schedule_e2e.py",
    REPO_ROOT / "tests" / "tools" / "test_sol_estimate.py",
    REPO_ROOT / "tests" / "tools" / "test_task_eval.py",
)
SHARED_REPORT_AND_PROFILE_TOOLS = (
    REPO_ROOT / "scripts" / "perf_evolve_prompt.py",
    REPO_ROOT / "tools" / "perfdb.py",
    REPO_ROOT / "tools" / "profile_report.py",
    REPO_ROOT / "tools" / "test_runner_parity.py",
    REPO_ROOT / "tools" / "nsys_to_layer_timing.py",
)
SHARED_SOL_ESTIMATE_TOOL = REPO_ROOT / "tools" / "sol_estimate.py"
SHARED_AUTO_PERF_TUNE_TOOL = REPO_ROOT / "tools" / "auto_perf_tune.py"
SHARED_CLASSIFY_BOTTLENECK_TOOL = REPO_ROOT / "tools" / "classify_bottleneck.py"
SHARED_DEBUG_DIFFUSION_PIPELINE_TOOL = REPO_ROOT / "tools" / "debug_diffusion_pipeline.py"
SHARED_VALIDATE_T5_TOOL = REPO_ROOT / "tools" / "validate_t5.py"
SHARED_VALIDATE_DIT_TOOL = REPO_ROOT / "tools" / "validate_dit.py"
SHARED_GITHUB_CI_FILES = (
    REPO_ROOT / ".github" / "workflows" / "nightly.yml",
    REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml",
    REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh",
    REPO_ROOT / ".github" / "scripts" / "start-gha-container.sh",
    REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh",
)
DIFFUSION_VLM_SIMILARITY_TOOL = REPO_ROOT / "tools" / "evaluate_diffusion_vlm_similarity.py"
MODEL_OWNED_DIFF_VL_HANDLERS = (
    FAMILIES / "qwen_vl" / "diff_vl.py",
    FAMILIES / "locateanything" / "diff_vl.py",
)
MODEL_OWNED_DIFF_LOGITS_HANDLERS = (FAMILIES / "whisper" / "diff_logits.py",)
MODEL_OWNED_DIFF_AUDIO_HANDLERS = (FAMILIES / "bark" / "diff_audio.py",)
MODEL_OWNED_DIFF_T5_HANDLERS = (FAMILIES / "wan_t2v" / "diff_t5.py",)
MODEL_OWNED_DEBUG_DIFFUSION_PIPELINE_HANDLERS = (
    FAMILIES / "wan_t2v" / "debug_diffusion_pipeline.py",
)
MODEL_OWNED_E2E_CONTRACT_PLUGINS = (
    E2E_MODELS / "sam" / "e2e_plugins" / "contract.py",
    E2E_MODELS / "sam3" / "e2e_plugins" / "contract.py",
)
MODEL_OWNED_E2E_PROMPTED_SEGMENTATION_RUNTIME_PLUGINS = (
    E2E_MODELS / "sam" / "e2e_plugins" / "runner.py",
    E2E_MODELS / "sam" / "e2e_plugins" / "comparator.py",
    E2E_MODELS / "sam" / "e2e_plugins" / "repro.py",
    E2E_MODELS / "sam3" / "e2e_plugins" / "runner.py",
    E2E_MODELS / "sam3" / "e2e_plugins" / "comparator.py",
    E2E_MODELS / "sam3" / "e2e_plugins" / "repro.py",
)
MODEL_OWNED_DIFFUSION_VALIDATE_HANDLERS = (
    FAMILIES / "wan_t2v" / "validate_t5.py",
    FAMILIES / "wan_t2v" / "validate_dit.py",
)
CPP_TESTS = REPO_ROOT / "tests" / "cpp"

_RUNTIME_INCLUDE_RE = re.compile(
    r'#\s*include\s+[<"](?P<path>[^">]*runtime/models/(?P<model>[^/]+)/[^">]+)[">]'
)
_FORBIDDEN_E2E_IMPORT_RE = re.compile(r"tests\.e2e_harness\.(?:runners|comparators|references)")
_FORBIDDEN_SHARED_BUILDER_MODULES = {
    "checkpoint_mapper",
    "config",
    "graph_blocks",
    "graph_ops",
    "utils",
}


def _runtime_model_ids() -> set[str]:
    return {path.name for path in RUNTIME_MODELS.iterdir() if path.is_dir()}


def _family_model_ids() -> set[str]:
    return {
        path.name for path in FAMILIES.iterdir() if path.is_dir() and not path.name.startswith("__")
    }


def _format_violations(violations: list[tuple[Path, int, str]]) -> str:
    return "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{line}: {detail}" for path, line, detail in violations
    )


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _cpp_files_under(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".h", ".hpp", ".cpp", ".cu", ".cc"}
    ]


def _strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def _model_prefix(model: str) -> str:
    return "".join(part.capitalize() for part in model.split("_"))


def test_runtime_models_do_not_include_sibling_model_folders() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prove each runtime model implementation includes only local model code.
    Preconditions: src/runtime/models/<model> folders exist.
    Postconditions: no file under one runtime model includes another model folder.
    """
    violations: list[tuple[Path, int, str]] = []
    for owner in sorted(_runtime_model_ids()):
        for path in (RUNTIME_MODELS / owner).rglob("*"):
            if path.suffix not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                match = _RUNTIME_INCLUDE_RE.search(line)
                if match and match.group("model") != owner:
                    violations.append((path, line_no, match.group("path")))

    assert not violations, _format_violations(violations)


def test_shared_runtime_build_files_do_not_name_model_owned_sources() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific runtime implementation files owned by model DSOs.
    Preconditions: CMakeLists.txt declares shared and model runtime build inputs.
    Postconditions: shared build files do not list model-specific source paths.
    """
    text = CMAKE_ROOT.read_text(encoding="utf-8")
    forbidden = [
        "src/runtime/domains/diffusion/qwen_image_types.cpp",
        "src/runtime/domains/audio/magpie_kernels.cu",
        "src/runtime/domains/audio/audio_bundle_validation.cpp",
    ]
    violations = [
        (CMAKE_ROOT, 0, f"shared CMake names model-owned source {needle}")
        for needle in forbidden
        if needle in text
    ]

    literal_model_source_re = re.compile(r"src/runtime/models/(?!\$\{_trtmc_model\}|\*)([^/]+)/")
    cmake_files = (
        CMAKE_ROOT,
        REPO_ROOT / "cmake" / "trtmc_pipeline_plugins.cmake",
    )
    for path in cmake_files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = literal_model_source_re.search(line)
            if match:
                violations.append(
                    (
                        path,
                        line_no,
                        f"shared CMake names model source folder {match.group(1)}",
                    )
                )

    assert not violations, _format_violations(violations)


def test_model_config_schemas_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep single-model runtime config schemas with the owning model DSO.
    Preconditions: runtime model manifests declare model-local config schemas.
    Postconditions: shared config schema manifests/directories do not name Bark
    or Magpie schema sources.
    """
    violations = []
    shared_schema_text = CONFIG_SCHEMA_CMAKE.read_text(encoding="utf-8")
    for needle in ("audio_bark", "audio_magpie", "bark", "magpie"):
        if needle in shared_schema_text:
            violations.append(
                (
                    CONFIG_SCHEMA_CMAKE,
                    0,
                    f"shared config schema manifest contains model-owned schema {needle}",
                )
            )

    forbidden_paths = (
        SHARED_CONFIG_SCHEMAS / "audio_bark.cpp",
        SHARED_CONFIG_SCHEMAS / "audio_magpie.cpp",
        SHARED_CONFIG_SCHEMA_INCLUDES / "audio_bark.h",
        SHARED_CONFIG_SCHEMA_INCLUDES / "audio_magpie.h",
        PY_RUNTIME_CONFIG_SCHEMAS / "audio_bark.py",
        PY_RUNTIME_CONFIG_SCHEMAS / "audio_magpie.py",
    )
    violations.extend(
        (path, 0, "single-model config schema must live under src/runtime/models")
        for path in forbidden_paths
        if path.exists()
    )

    expected_schema_owners = {
        "bark": 'runtime_config_schemas = ["config_schema.cpp|register_audio_bark_schema"]',
        "magpie": 'runtime_config_schemas = ["config_schema.cpp|register_audio_magpie_schema"]',
    }
    for model, manifest_line in expected_schema_owners.items():
        manifest = RUNTIME_MODELS / model / "MODEL.toml"
        if manifest_line not in manifest.read_text(encoding="utf-8"):
            violations.append((manifest, 0, f"missing model-owned schema entry {manifest_line}"))

    expected_python_schemas = (
        FAMILIES / "bark" / "runtime_config_schema.py",
        FAMILIES / "magpie_tts" / "runtime_config_schema.py",
    )
    violations.extend(
        (path, 0, "missing model-owned Python runtime config schema")
        for path in expected_python_schemas
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_model_owned_cpp_tests_do_not_live_in_shared_cpp_root() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep single-owner C++ tests beside the owning runtime model tests.
    Preconditions: runtime model manifests declare model-owned C++ tests.
    Postconditions: shared tests/cpp root has no single-owner model test files.
    """
    forbidden_shared_tests = (
        "test_audio_pipeline_new.cpp",
        "test_bark_generation_plan.cpp",
        "test_chat_template.cpp",
        "test_decode_runtime.cpp",
        "test_diffusion_generation_plan.cpp",
        "test_diffusion_batch_utils.cpp",
        "test_diffusion_pipeline_new.cpp",
        "test_elf_flow_pipeline.cpp",
        "test_flux_denoising_step_seam.cpp",
        "test_ltx_video_pipeline.cpp",
        "test_magpie_codec_plan.cpp",
        "test_magpie_decode_policy.cpp",
        "test_magpie_decoder_plan.cpp",
        "test_magpie_text_completion_policy.cpp",
        "test_omni_audio_plan.cpp",
        "test_plugin_helpers.cpp",
        "test_qwen_image_cfg_renorm.cpp",
        "test_rnnt_decode_policy.cpp",
        "test_rnnt_streaming_contract.cpp",
        "test_sam_prompt_seam.cpp",
        "test_sam3_pipeline.cpp",
        "test_sam_image_preprocess_seam.cpp",
        "test_segformer_preprocess_seam.cpp",
        "test_segformer_postprocess_seam.cpp",
        "test_perception_preprocess_seams.cpp",
        "test_neural_operator_config.cpp",
        "test_speech_decode_stop_policy.cpp",
        "test_speech_depth_plan.cpp",
        "test_speech_generation_helpers.cpp",
        "test_speech_mimi_decode_plan.cpp",
        "test_speech_pipeline.cpp",
        "test_speech_runtime_plan.cpp",
        "test_speech_subprocess_seam.cpp",
        "test_speech_temporal_embed_plan.cpp",
        "test_trt_engine_lifecycle.cpp",
        "test_trt_engine_lifecycle_fake_engine.cpp",
        "test_wan_generation_conditioning.cpp",
        "test_wan_generation_plan.cpp",
        "test_whisper_decode_policy.cpp",
        "test_whisper_host_plan.cpp",
    )
    violations = [
        (REPO_ROOT / "tests" / "cpp" / filename, 0, "single-owner C++ test must be model-owned")
        for filename in forbidden_shared_tests
        if (REPO_ROOT / "tests" / "cpp" / filename).exists()
    ]

    assert not violations, _format_violations(violations)


def test_shared_cpp_tests_do_not_regress_known_model_owned_cases() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep moved model-specific C++ assertions beside their owning tests.
    Preconditions: decoder, audio, and diffusion model tests have local homes.
    Postconditions: shared C++ tests do not name the model-owned cases moved out.
    """
    forbidden_by_file = {
        "test_chat_template.cpp": (
            "Nemotron",
            "nemotron",
            "truncate_history_thinking",
            "SPECIAL_10",
        ),
        "test_recurrent_pipeline.cpp": (
            "Nemotron",
            "nemotron",
            "SPECIAL_10",
        ),
        "test_recurrent_step_contracts.cpp": (
            "initialize_rwkv_outputs",
            "initialize_mamba_outputs",
            "rwkv",
            "mamba",
        ),
        "test_model_plugin_loader.cpp": (
            "qwen_decoder_kv_cache",
            "llama",
            "libtrtmc_model_llama.so",
            "nemotron_labs_diffusion",
            "diffusion_flux",
            "speech_to_text",
            "flux",
            "whisper",
        ),
        "test_c_abi_runtime_regression.cpp": (
            '"runtime_strategy": "diffusion"',
            "test_diffusion_bundle_missing_required_section_reports_error",
            "denoiser_plan",
        ),
        "test_bundle_format.cpp": (
            "qwen",
            "flux",
        ),
        "test_encoder_pipeline.cpp": (
            "SegmentPipeline",
            "SamPipeline",
            "SamConfig",
            "build_segment_engine",
            "build_sam",
        ),
        "test_diffusion_math.cpp": (
            "Wan",
            "FLUX",
            "Z-Image",
        ),
        "test_ipa_tokenizer.cpp": ("MagpieTTS",),
        "test_perception_preprocess_seams.cpp": (
            "SAM",
            "Sam",
            "sam_",
            "SegmentationConfig",
            "preprocess_segmentation_image",
        ),
        "test_neural_operator_config.cpp": (
            "SegmentationLogitsShape",
            "SegmentationPostprocessStatus",
            "compute_segmentation_class_map_from_logits",
        ),
    }
    violations = []
    for filename, forbidden in forbidden_by_file.items():
        path = CPP_TESTS / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared C++ test contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_top_level_cmake_does_not_hardcode_model_owned_cpp_tests() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model C++ test registration in model manifests.
    Preconditions: runtime model manifests declare model-owned runtime_tests.
    Postconditions: top-level CMake does not name model-owned pipeline tests.
    """
    text = CMAKE_ROOT.read_text(encoding="utf-8")
    forbidden = (
        "test_llama_pipeline",
        "test_recurrent_pipeline",
        "test_encoder_pipeline",
        "test_vl_pipeline",
        "test_perception_preprocess_seams",
        "test_neural_operator_config",
        "add_dependencies(test_model_plugin_loader trtmc_model_llama)",
    )
    violations = [
        (CMAKE_ROOT, 0, f"top-level CMake hardcodes model-owned test detail {needle}")
        for needle in forbidden
        if needle in text
    ]

    expected_manifest_entries = {
        "llama": "test_llama_pipeline|test_llama_pipeline.cpp",
        "nemotron_labs_diffusion": "test_nemotron_labs_diffusion_chat_template|test_nemotron_labs_diffusion_chat_template.cpp",
        "qwen_vl": "test_qwen_vl_vl_pipeline|test_qwen_vl_vl_pipeline.cpp",
        "internvl": "test_internvl_vl_pipeline|test_internvl_vl_pipeline.cpp",
        "deepseek_ocr": "test_deepseek_ocr_vl_pipeline|test_deepseek_ocr_vl_pipeline.cpp",
        "locateanything": "test_locateanything_vl_pipeline|test_locateanything_vl_pipeline.cpp",
        "lance": "test_lance_vl_pipeline|test_lance_vl_pipeline.cpp",
        "phi4_multimodal": "test_phi4_multimodal_vl_pipeline|test_phi4_multimodal_vl_pipeline.cpp",
        "whisper": "test_whisper_pipeline|test_whisper_pipeline.cpp",
        "canary": "test_canary_pipeline|test_canary_pipeline.cpp",
    }
    for family in ENCODER_RUNTIME_STRATEGIES:
        expected_manifest_entries[family] = (
            f"test_{family}_encoder_pipeline|test_{family}_encoder_pipeline.cpp"
        )
    for family in RECURRENT_RUNTIME_STRATEGIES:
        expected_manifest_entries[family] = (
            f"test_{family}_recurrent_pipeline|test_{family}_recurrent_pipeline.cpp"
        )
    for model, entry in expected_manifest_entries.items():
        manifest = RUNTIME_MODELS / model / "MODEL.toml"
        if entry not in manifest.read_text(encoding="utf-8"):
            violations.append((manifest, 0, f"missing model-owned C++ test {entry}"))

    assert not violations, _format_violations(violations)


def test_shared_chat_template_core_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep chat template detection and rendering in runtime model-owned files.
    Preconditions: decoder and recurrent models own their template helpers.
    Postconditions: the old shared chat_template registry is absent and unused.
    """
    violations = []
    for path in CHAT_TEMPLATE_CORE_FILES:
        if path.exists():
            violations.append((path, 0, "shared chat template registry must be deleted"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8")
    for needle in ("src/runtime/core/chat_template.cpp", "trtmc_add_test(test_chat_template)"):
        if needle in cmake_text:
            violations.append(
                (CMAKE_ROOT, 0, f"shared chat template build reference remains: {needle}")
            )

    legacy_patterns = {
        "runtime/core/chat_template.h": re.compile(r"runtime/core/chat_template\.h"),
        "register_chat_template_format": re.compile(r"\bregister_chat_template_format\s*\("),
        "detect_chat_template_format": re.compile(
            r"(?<![A-Za-z0-9_])detect_chat_template_format\s*\("
        ),
        "apply_chat_template": re.compile(
            r"(?<![A-Za-z0-9_])apply_chat_template\s*\([^,\n]+,[^,\n]+,"
        ),
        "ChatTemplateFormat": re.compile(r"\bChatTemplateFormat\b"),
        "ChatTemplateApplyFn": re.compile(r"\bChatTemplateApplyFn\b"),
    }
    scan_roots = (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp")
    for root in scan_roots:
        for path in root.rglob("*"):
            if path.suffix not in {".cpp", ".h", ".cu", ".cuh", ".hpp"}:
                continue
            text = path.read_text(encoding="utf-8")
            violations.extend(
                (path, 0, f"legacy shared chat template API remains: {name}")
                for name, pattern in legacy_patterns.items()
                if pattern.search(text)
            )

    for source in sorted(RUNTIME_MODELS.glob("*/chat_templates.cpp")):
        family = source.parent.name
        header = source.with_suffix(".h")
        if not header.is_file():
            violations.append((header, 0, "missing model-owned chat template header"))
            continue
        source_text = source.read_text(encoding="utf-8")
        header_text = header.read_text(encoding="utf-8")
        expected_symbols = (
            f"{family}_detect_chat_template_format",
            f"{family}_apply_chat_template",
        )
        for symbol in expected_symbols:
            if symbol not in source_text:
                violations.append((source, 0, f"missing model-owned chat template symbol {symbol}"))
            if symbol not in header_text:
                violations.append((header, 0, f"missing model-owned chat template symbol {symbol}"))

    assert not violations, _format_violations(violations)


def test_production_models_do_not_include_shared_trt_common() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model runtime behavior from depending on shared TRT logger state.
    Preconditions: model plugins receive resolved runtime config at construction.
    Postconditions: production model sources do not include trt_common or query its log state.
    """
    forbidden_patterns = {
        "runtime/core/trt_common.h": re.compile(r"runtime/core/trt_common\.h"),
        "trt_log_to_stderr_enabled": re.compile(r"\btrt_log_to_stderr_enabled\s*\("),
        "trt_log_stderr_min_severity": re.compile(r"\btrt_log_stderr_min_severity\s*\("),
        "configure_trt_logger": re.compile(r"\bconfigure_trt_logger\s*\("),
        "TrtLogSeverity": re.compile(r"\bTrtLogSeverity\b"),
    }
    violations = []
    for path in RUNTIME_MODELS.rglob("*"):
        if path.suffix not in {".cpp", ".h", ".cu", ".cuh", ".hpp"}:
            continue
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"production model source uses shared TRT common API: {name}")
            for name, pattern in forbidden_patterns.items()
            if pattern.search(text)
        )

    assert not violations, _format_violations(violations)


def test_production_models_do_not_include_runtime_core_headers() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep production model implementation independent from shared runtime core headers.
    Preconditions: model-owned helpers wrap any CUDA/TRT/image behavior needed by models.
    Postconditions: production model sources include no runtime/core headers.
    """
    include_pattern = re.compile(r'#include\s+"runtime/core/')
    violations = []
    for path in RUNTIME_MODELS.rglob("*"):
        if path.suffix not in {".cpp", ".h", ".cu", ".cuh", ".hpp"}:
            continue
        text = path.read_text(encoding="utf-8")
        if include_pattern.search(text):
            violations.append(
                (path, 0, "production model source includes shared runtime/core header")
            )

    assert not violations, _format_violations(violations)


def test_model_cuda_helpers_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep production model CUDA ownership helpers in model-owned folders.
    Preconditions: VL and Magpie models own their CUDA stream/buffer helpers.
    Postconditions: production model sources do not include shared cuda_common.
    """
    expected_files = [
        RUNTIME_MODELS / "magpie" / "cuda_common.h",
        RUNTIME_MODELS / "magpie" / "cuda_common.cpp",
    ]
    expected_files.extend(
        RUNTIME_MODELS / family / "cuda_stream.h"
        for family in (
            "deepseek_ocr",
            "internvl",
            "lance",
            "locateanything",
            "phi4_multimodal",
            "qwen_vl",
        )
    )
    violations = [
        (path, 0, "missing model-owned CUDA helper")
        for path in expected_files
        if not path.is_file()
    ]

    for path in RUNTIME_MODELS.rglob("*"):
        if path.suffix not in {".cpp", ".h", ".cu", ".cuh", ".hpp"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "runtime/core/cuda_common.h" in text:
            violations.append((path, 0, "production model source includes shared cuda_common"))

    assert not violations, _format_violations(violations)


def test_flux_gpu_matmul_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Flux preprocessor GPU math with the Flux runtime model.
    Preconditions: Flux owns its GPU matmul helper source.
    Postconditions: the old shared gpu_matmul core source is absent and unused.
    """
    retired_paths = (
        REPO_ROOT / "src" / "runtime" / "core" / "gpu_matmul.h",
        REPO_ROOT / "src" / "runtime" / "core" / "gpu_matmul.cpp",
    )
    violations = [
        (path, 0, "shared gpu_matmul core helper must be deleted")
        for path in retired_paths
        if path.exists()
    ]

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8")
    if "src/runtime/core/gpu_matmul.cpp" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "shared gpu_matmul core source remains in CMake"))

    flux_header = RUNTIME_MODELS / "flux" / "gpu_matmul.h"
    flux_source = RUNTIME_MODELS / "flux" / "gpu_matmul.cpp"
    flux_manifest = RUNTIME_MODELS / "flux" / "MODEL.toml"
    for path in (flux_header, flux_source):
        if not path.is_file():
            violations.append((path, 0, "missing Flux-owned gpu_matmul file"))
    if not flux_manifest.is_file():
        violations.append((flux_manifest, 0, "missing Flux model manifest"))
    elif 'runtime_link_libraries = ["cublas"]' not in flux_manifest.read_text(encoding="utf-8"):
        violations.append((flux_manifest, 0, "Flux cuBLAS dependency must be model-owned"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in root.rglob("*"):
            if path.suffix not in {".cpp", ".h", ".cu", ".cuh", ".hpp"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "runtime/core/gpu_matmul.h" in text:
                violations.append((path, 0, "shared gpu_matmul include remains"))

    if flux_source.is_file():
        text = flux_source.read_text(encoding="utf-8")
        for symbol in ("flux_gpu_matmul_init", "flux_gpu_matmul_shutdown", "flux_gpu_matmul_bias"):
            if symbol not in text:
                violations.append((flux_source, 0, f"missing Flux-owned symbol {symbol}"))

    assert not violations, _format_violations(violations)


def test_decoded_image_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep decoded image preprocessing value types with model families.
    Preconditions: vision/perception model families own their preprocessing headers.
    Postconditions: the old shared decoded_image core header is absent and unused.
    """
    shared_header = REPO_ROOT / "src" / "runtime" / "core" / "decoded_image.h"
    violations = []
    if shared_header.exists():
        violations.append((shared_header, 0, "shared decoded_image header must be deleted"))

    expected_families = (
        "deepseek_ocr",
        "internvl",
        "lance",
        "locateanything",
        "phi4_multimodal",
        "qwen_vl",
        "sam",
        "segformer",
    )
    for family in expected_families:
        header = RUNTIME_MODELS / family / "decoded_image.h"
        if not header.is_file():
            violations.append((header, 0, "missing model-owned decoded image header"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in root.rglob("*"):
            if path.suffix not in {".cpp", ".h", ".cu", ".cuh", ".hpp"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "runtime/core/decoded_image.h" in text:
                violations.append((path, 0, "shared decoded_image include remains"))

    assert not violations, _format_violations(violations)


def test_shared_runtime_core_and_domains_do_not_name_single_family_defaults() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model defaults/comments in model-owned runtime files.
    Preconditions: shared runtime core/domain files provide generic utilities.
    Postconditions: shared runtime files contain no single-family default strings.
    """
    forbidden_by_file = {
        "hybrid_state.h": ("Nemotron", "nemotron"),
    }
    violations = []
    for path in SHARED_RUNTIME_LEAK_FILES:
        text = path.read_text(encoding="utf-8")
        forbidden = forbidden_by_file.get(path.name, ())
        violations.extend(
            (path, 0, f"shared runtime file contains single-family term {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_hybrid_state_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep hybrid attention+recurrent state behavior with the hybrid
    model families that own it.
    Preconditions: Nemotron-H and Qwen3.5 provide local hybrid state classes.
    Postconditions: the old shared HybridState source/header are absent,
    CMake no longer links them into core, and non-hybrid families/tests do not
    include or instantiate hybrid state.
    """
    retired_paths = (
        REPO_ROOT / "include" / "trtmc" / "runtime" / "hybrid_state.h",
        REPO_ROOT / "src" / "runtime" / "core" / "hybrid_state.cpp",
    )
    required_owned = {
        "nemotron_h": (
            REPO_ROOT / "src" / "runtime" / "models" / "nemotron_h" / "hybrid_state.h",
            REPO_ROOT / "src" / "runtime" / "models" / "nemotron_h" / "hybrid_state.cpp",
            "NemotronHHybridState",
        ),
        "qwen3_5": (
            REPO_ROOT / "src" / "runtime" / "models" / "qwen3_5" / "hybrid_state.h",
            REPO_ROOT / "src" / "runtime" / "models" / "qwen3_5" / "hybrid_state.cpp",
            "Qwen35HybridState",
        ),
    }
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared HybridState artifact must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    if "src/runtime/core/hybrid_state.cpp" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "CMake links retired shared hybrid_state.cpp"))

    for family, (header, source, class_name) in required_owned.items():
        for path in (header, source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned hybrid state file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if class_name not in text:
                violations.append((path, 0, f"missing {class_name}"))
        plugin = REPO_ROOT / "src" / "runtime" / "models" / family / "plugin.cpp"
        plugin_text = plugin.read_text(encoding="utf-8", errors="ignore")
        if class_name not in plugin_text:
            violations.append((plugin, 0, f"plugin does not instantiate {class_name}"))

    forbidden_include = "trtmc/runtime/hybrid_state.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            if forbidden_include in path.read_text(encoding="utf-8", errors="ignore"):
                violations.append((path, 0, "includes retired shared hybrid state header"))

    pure_recurrent_tests = (
        REPO_ROOT / "tests" / "cpp" / "models" / "mamba" / "test_mamba_recurrent_pipeline.cpp",
        REPO_ROOT / "tests" / "cpp" / "models" / "rwkv" / "test_rwkv_recurrent_pipeline.cpp",
    )
    for path in pure_recurrent_tests:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in ("HybridState", "HybridPipeline", "hybrid"):
            if needle in text:
                violations.append((path, 0, f"pure recurrent test contains {needle}"))

    assert not violations, _format_violations(violations)


def test_recurrent_state_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep recurrent tensor state allocation, binding, reset, and
    advancement behavior in the model families that own recurrent execution.
    Preconditions: recurrent consumers provide family-local state classes.
    Postconditions: the old shared RecurrentState source/header/root test are
    absent, CMake no longer links them into core, and model-owned code does not
    include the retired public header.
    """
    retired_paths = (
        REPO_ROOT / "include" / "trtmc" / "runtime" / "recurrent_state.h",
        REPO_ROOT / "src" / "runtime" / "core" / "recurrent_state.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_recurrent_state.cpp",
    )
    required_owned = {
        "mamba": (
            REPO_ROOT / "src" / "runtime" / "models" / "mamba" / "recurrent_state.h",
            REPO_ROOT / "src" / "runtime" / "models" / "mamba" / "recurrent_state.cpp",
            "MambaRecurrentState",
        ),
        "rwkv": (
            REPO_ROOT / "src" / "runtime" / "models" / "rwkv" / "recurrent_state.h",
            REPO_ROOT / "src" / "runtime" / "models" / "rwkv" / "recurrent_state.cpp",
            "RwkvRecurrentState",
        ),
        "nemotron_h": (
            REPO_ROOT / "src" / "runtime" / "models" / "nemotron_h" / "recurrent_state.h",
            REPO_ROOT / "src" / "runtime" / "models" / "nemotron_h" / "recurrent_state.cpp",
            "NemotronHRecurrentState",
        ),
        "qwen3_5": (
            REPO_ROOT / "src" / "runtime" / "models" / "qwen3_5" / "recurrent_state.h",
            REPO_ROOT / "src" / "runtime" / "models" / "qwen3_5" / "recurrent_state.cpp",
            "Qwen35RecurrentState",
        ),
    }
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared RecurrentState artifact must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in ("src/runtime/core/recurrent_state.cpp", "test_recurrent_state"):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired {term}"))

    for family, (header, source, class_name) in required_owned.items():
        for path in (header, source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned recurrent state file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if class_name not in text:
                violations.append((path, 0, f"missing {class_name}"))
        plugin = REPO_ROOT / "src" / "runtime" / "models" / family / "plugin.cpp"
        plugin_text = plugin.read_text(encoding="utf-8", errors="ignore")
        if class_name not in plugin_text and family not in {"nemotron_h", "qwen3_5"}:
            violations.append((plugin, 0, f"plugin does not instantiate {class_name}"))

    for family, (_, _, class_name) in required_owned.items():
        if family in {"nemotron_h", "qwen3_5"}:
            hybrid = REPO_ROOT / "src" / "runtime" / "models" / family / "hybrid_state.h"
            if class_name not in hybrid.read_text(encoding="utf-8", errors="ignore"):
                violations.append((hybrid, 0, f"hybrid state does not own {class_name}"))

    forbidden_include = "trtmc/runtime/recurrent_state.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            if forbidden_include in path.read_text(encoding="utf-8", errors="ignore"):
                violations.append((path, 0, "includes retired shared recurrent state header"))

    assert not violations, _format_violations(violations)


def test_inference_and_kv_cache_shared_artifacts_are_retired() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep inference state and KV-cache behavior model-owned instead of
    depending on shared runtime cache/state classes.
    Preconditions: every runtime model folder carries local cache/state copies.
    Postconditions: retired shared cache/state artifacts are absent, CMake does
    not link them, and non-model-owned C++ code does not reference their
    public headers or unprefixed API names.
    """
    retired_paths = (
        REPO_ROOT / "include" / "trtmc" / "runtime" / "inference_state.h",
        REPO_ROOT / "include" / "trtmc" / "runtime" / "kv_cache.h",
        REPO_ROOT / "include" / "trtmc" / "runtime" / "triattention_kv_cache.h",
        REPO_ROOT / "src" / "runtime" / "core" / "kv_cache.cpp",
        REPO_ROOT / "src" / "runtime" / "core" / "triattention_kv_cache.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_kv_cache_new.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_triattention_kv_cache.cpp",
    )
    retired_includes = (
        "trtmc/runtime/inference_state.h",
        "trtmc/runtime/kv_cache.h",
        "trtmc/runtime/triattention_kv_cache.h",
    )
    forbidden_shared_symbols = (
        re.compile(r"\bIInferenceState\b"),
        re.compile(r"\bKvCache\b"),
        re.compile(r"\bKvCacheNames\b"),
        re.compile(r"\bTriAttentionKvCache\b"),
        re.compile(r"\bTriAttentionConfig\b"),
        re.compile(r"\bTriAttentionStats\b"),
        re.compile(r"(?<![A-Za-z0-9_])parse_triattention_bundle_config\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])parse_triattention_stats_json\s*\("),
    )
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared cache/state artifact must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in (
        "src/runtime/core/kv_cache.cpp",
        "src/runtime/core/triattention_kv_cache.cpp",
        "test_kv_cache_new",
        "test_triattention_kv_cache",
    ):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired {term}"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", CPP_TESTS):
        for path in _cpp_files_under(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for include in retired_includes:
                if include in text:
                    violations.append((path, 0, f"includes retired {include}"))

            if _is_under(path, RUNTIME_MODELS) or _is_under(path, CPP_TESTS / "models"):
                continue
            stripped = _strip_cpp_comments(text)
            for pattern in forbidden_shared_symbols:
                if pattern.search(stripped):
                    violations.append(
                        (path, 0, f"uses shared cache/state symbol {pattern.pattern}")
                    )

    assert not violations, _format_violations(violations)


def test_model_cache_state_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: ensure each runtime model owns its state interface and dense KV cache.
    Preconditions: runtime model folders are the unit of model implementation.
    Postconditions: every runtime model has local cache/state files with
    family-prefixed types and no include of retired shared cache/state headers.
    """
    retired_includes = (
        "trtmc/runtime/inference_state.h",
        "trtmc/runtime/kv_cache.h",
        "trtmc/runtime/triattention_kv_cache.h",
    )
    forbidden_unowned_symbols = (
        re.compile(r"\bIInferenceState\b"),
        re.compile(r"\bKvCache\b"),
        re.compile(r"\bKvCacheNames\b"),
    )
    violations = []

    for family in sorted(_runtime_model_ids()):
        prefix = _model_prefix(family)
        model_dir = RUNTIME_MODELS / family
        state_header = model_dir / "inference_state.h"
        cache_header = model_dir / "kv_cache.h"
        cache_source = model_dir / "kv_cache.cpp"

        for path in (state_header, cache_header, cache_source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned cache/state file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for include in retired_includes:
                if include in text:
                    violations.append((path, 0, f"includes retired {include}"))
            stripped = _strip_cpp_comments(text)
            for pattern in forbidden_unowned_symbols:
                if pattern.search(stripped):
                    violations.append(
                        (path, 0, f"uses unowned cache/state symbol {pattern.pattern}")
                    )

        if state_header.is_file():
            text = state_header.read_text(encoding="utf-8", errors="ignore")
            if f"class {prefix}InferenceState" not in text:
                violations.append((state_header, 0, f"missing {prefix}InferenceState"))

        if cache_header.is_file():
            text = cache_header.read_text(encoding="utf-8", errors="ignore")
            expected_include = f"runtime/models/{family}/inference_state.h"
            for needle in (
                f"struct {prefix}KvCacheNames",
                f"class {prefix}KvCache : public {prefix}InferenceState",
                expected_include,
            ):
                if needle not in text:
                    violations.append((cache_header, 0, f"missing {needle}"))

        if cache_source.is_file():
            text = cache_source.read_text(encoding="utf-8", errors="ignore")
            for needle in (
                f"runtime/models/{family}/kv_cache.h",
                f"{prefix}KvCache::",
                f"{prefix}KvCacheNames",
            ):
                if needle not in text:
                    violations.append((cache_source, 0, f"missing {needle}"))

        for path in _cpp_files_under(model_dir):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for include in retired_includes:
                if include in text:
                    violations.append((path, 0, f"model includes retired {include}"))

    assert not violations, _format_violations(violations)


def test_text_decoder_triattention_cache_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep TriAttention cache policy local to decoder text model families.
    Preconditions: decoder text runtime folders own pipeline and cache copies.
    Postconditions: every TriAttention-capable text family has family-prefixed
    TriAttention types/functions and family-owned CUDA kernel helpers.
    """
    required_families = (
        "bloom",
        "codegen",
        "deepseek_v2",
        "falcon",
        "gemma",
        "glm",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "gpt_oss",
        "granite",
        "internlm",
        "llama",
        "mistral",
        "mixtral",
        "nemotron",
        "nemotron_labs_diffusion",
        "olmo",
        "olmo2",
        "opt",
        "phi",
        "phi_moe",
        "qwen",
        "qwen_moe",
        "stablelm",
        "starcoder2",
        "xglm",
    )
    forbidden_unowned_symbols = (
        re.compile(r"\bTriAttentionScoreAggregation\b"),
        re.compile(r"\bTriAttentionRopeStyle\b"),
        re.compile(r"\bTriAttentionConfig\b"),
        re.compile(r"\bTriAttentionStats\b"),
        re.compile(r"\bTriAttentionKvCache\b"),
        re.compile(r"(?<![A-Za-z0-9_])parse_triattention_bundle_config\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])parse_triattention_stats_json\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])triattention_score_candidates_gpu\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])triattention_compact_rows_gpu\s*\("),
    )
    violations = []

    retired_shared = (
        REPO_ROOT / "src" / "runtime" / "core" / "triattention_kernels.h",
        REPO_ROOT / "src" / "runtime" / "core" / "triattention_kernels.cu",
    )
    for path in retired_shared:
        if path.exists():
            violations.append((path, 0, "shared TriAttention kernel must be model-owned"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    if "src/runtime/core/triattention_kernels.cu" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "trtmc_core links shared TriAttention kernel"))

    for family in required_families:
        prefix = _model_prefix(family)
        model_dir = RUNTIME_MODELS / family
        header = model_dir / "triattention_kv_cache.h"
        source = model_dir / "triattention_kv_cache.cpp"
        kernel_header = model_dir / "triattention_kernels.h"
        kernel_source = model_dir / "triattention_kernels.cu"
        plugin = model_dir / "plugin.cpp"
        expected_include = f"runtime/models/{family}/triattention_kv_cache.h"
        expected_kernel_include = f"runtime/models/{family}/triattention_kernels.h"
        score_kernel = f"{family}_triattention_score_candidates_gpu"
        compact_kernel = f"{family}_triattention_compact_rows_gpu"

        for path in (header, source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned TriAttention file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "trtmc/runtime/triattention_kv_cache.h" in text:
                violations.append((path, 0, "includes retired shared TriAttention header"))
            if "runtime/core/triattention_kernels.h" in text:
                violations.append((path, 0, "includes retired shared TriAttention kernel"))
            for needle in (
                f"{prefix}TriAttentionScoreAggregation",
                f"{prefix}TriAttentionRopeStyle",
                f"{prefix}TriAttentionConfig",
                f"{prefix}TriAttentionStats",
                f"{prefix}TriAttentionKvCache",
                f"{family}_parse_triattention_bundle_config",
                f"{family}_parse_triattention_stats_json",
            ):
                if needle not in text:
                    violations.append((path, 0, f"missing {needle}"))
            if path == source:
                for needle in (expected_kernel_include, score_kernel, compact_kernel):
                    if needle not in text:
                        violations.append((path, 0, f"missing {needle}"))
            stripped = _strip_cpp_comments(text)
            for pattern in forbidden_unowned_symbols:
                if pattern.search(stripped):
                    violations.append(
                        (path, 0, f"uses unowned TriAttention symbol {pattern.pattern}")
                    )

        for path in (kernel_header, kernel_source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned TriAttention kernel file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "runtime/core/triattention_kernels.h" in text:
                violations.append((path, 0, "family kernel includes retired shared kernel"))
            for needle in (score_kernel, compact_kernel):
                if needle not in text:
                    violations.append((path, 0, f"missing {needle}"))
            stripped = _strip_cpp_comments(text)
            for pattern in forbidden_unowned_symbols:
                if pattern.search(stripped):
                    violations.append(
                        (path, 0, f"uses unowned TriAttention kernel symbol {pattern.pattern}")
                    )

        plugin_text = plugin.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            expected_include,
            f"{prefix}TriAttentionConfig",
            f"{prefix}TriAttentionStats",
            f"{prefix}TriAttentionKvCache",
            f"{family}_parse_triattention_bundle_config",
            f"{family}_parse_triattention_stats_json",
        ):
            if needle not in plugin_text:
                violations.append((plugin, 0, f"plugin missing {needle}"))
        if "trtmc/runtime/triattention_kv_cache.h" in plugin_text:
            violations.append((plugin, 0, "plugin includes retired shared TriAttention header"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", CPP_TESTS):
        for path in _cpp_files_under(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "runtime/core/triattention_kernels.h" in text:
                violations.append((path, 0, "includes retired shared TriAttention kernel"))

    assert not violations, _format_violations(violations)


def test_recurrent_sampler_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep recurrent text sampling policy local to the recurrent model
    family instead of depending on the shared runtime sampler implementation.
    Preconditions: recurrent pipelines are duplicated into model-owned folders.
    Postconditions: each recurrent family owns sampler source/header files and
    its pipeline does not include trtmc/runtime/sampler.h.
    """
    required_owned = {
        "mamba": ("MambaISampler", "MambaSamplingParams", "create_mamba_sampler"),
        "rwkv": ("RwkvISampler", "RwkvSamplingParams", "create_rwkv_sampler"),
        "nemotron_h": (
            "NemotronHISampler",
            "NemotronHSamplingParams",
            "create_nemotron_h_sampler",
        ),
        "qwen3_5": ("Qwen35ISampler", "Qwen35SamplingParams", "create_qwen35_sampler"),
    }
    violations = []

    for family, (sampler_class, params_class, factory_name) in required_owned.items():
        model_dir = RUNTIME_MODELS / family
        sampler_header = model_dir / "sampler.h"
        sampler_source = model_dir / "sampler.cpp"
        pipeline_header = model_dir / "pipeline.h"
        pipeline_source = model_dir / "pipeline.cpp"

        for path in (sampler_header, sampler_source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned sampler file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in (sampler_class, params_class, factory_name):
                if needle not in text:
                    violations.append((path, 0, f"missing {needle}"))
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "model-owned sampler includes shared sampler header"))

        for path in (pipeline_header, pipeline_source):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "recurrent pipeline includes shared sampler header"))
            if sampler_class not in text:
                violations.append((path, 0, f"pipeline does not use {sampler_class}"))
            if params_class not in text:
                violations.append((path, 0, f"pipeline does not use {params_class}"))
            if path == pipeline_source and factory_name not in text:
                violations.append((path, 0, f"pipeline does not call {factory_name}"))

    assert not violations, _format_violations(violations)


def test_shared_sampler_is_retired() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep sampling behavior model-owned once every sampler consumer has
    a family-local implementation.
    Preconditions: decoder, recurrent, and VL runtime folders own sampler files.
    Postconditions: the old shared runtime sampler source/header/root test are
    absent, CMake no longer links them into core, and C++ code does not use the
    unprefixed shared sampler API.
    """
    retired_paths = (
        REPO_ROOT / "include" / "trtmc" / "runtime" / "sampler.h",
        REPO_ROOT / "src" / "runtime" / "core" / "sampler.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_sampler.cpp",
    )
    forbidden_patterns = (
        re.compile(r"\bISampler\b"),
        re.compile(r"\bSamplingParams\b"),
        re.compile(r"\bSamplerFactoryOptions\b"),
        re.compile(r"\bSampleResult\b"),
        re.compile(r"\bLogitsLocation\b"),
        re.compile(r"\bcreate_sampler\s*\("),
        re.compile(r"\bcreate_gpu_greedy_sampler\s*\("),
        re.compile(r"\bsampling_params_from_config\s*\("),
    )
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared sampler artifact must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in ("src/runtime/core/sampler.cpp", "trtmc_add_test(test_sampler)"):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired {term}"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "includes retired shared sampler header"))
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses shared sampler symbol {pattern.pattern}"))

    assert not violations, _format_violations(violations)


def test_shared_decode_runtime_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep CPU decode token selection and mask helpers model-owned.
    Preconditions: seq2seq, speech, and speech-to-speech runtime folders own
    family-local decode_runtime files.
    Postconditions: the old shared decode runtime source/root test are absent,
    CMake does not link them into core, and call sites use family-prefixed APIs.
    """
    required_families = {
        "bark": "BarkMaskedScore",
        "bart": "BartMaskedScore",
        "canary": "CanaryMaskedScore",
        "m2m_100": "M2m100MaskedScore",
        "magpie": "MagpieMaskedScore",
        "personaplex": "PersonaplexMaskedScore",
        "whisper": "WhisperMaskedScore",
    }
    retired_paths = (
        REPO_ROOT / "src" / "runtime" / "core" / "trt_decode_runtime.h",
        REPO_ROOT / "src" / "runtime" / "core" / "trt_decode_runtime.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_decode_runtime.cpp",
    )
    forbidden_unowned_symbols = (
        re.compile(r"(?<![A-Za-z0-9_])select_argmax_token\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])sample_token_topk\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])select_topk_tokens\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])build_attention_mask\s*\("),
    )
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared decode runtime artifact must be model-owned"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in ("src/runtime/core/trt_decode_runtime.cpp", "trtmc_add_test(test_decode_runtime"):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired {term}"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "runtime/core/trt_decode_runtime.h" in text:
                violations.append((path, 0, "includes retired shared decode runtime header"))
            model_owned_path = (
                RUNTIME_MODELS in path.parents
                or (REPO_ROOT / "tests" / "cpp" / "models") in path.parents
            )
            if model_owned_path:
                continue
            for pattern in forbidden_unowned_symbols:
                if pattern.search(text):
                    violations.append(
                        (path, 0, f"uses shared decode runtime symbol {pattern.pattern}")
                    )

    for family, masked_score in required_families.items():
        model_dir = RUNTIME_MODELS / family
        header = model_dir / "decode_runtime.h"
        source = model_dir / "decode_runtime.cpp"
        expected_include = f"runtime/models/{family}/decode_runtime.h"
        if not header.is_file():
            violations.append((header, 0, "missing model-owned decode runtime header"))
            continue
        if not source.is_file():
            violations.append((source, 0, "missing model-owned decode runtime source"))
            continue
        header_text = header.read_text(encoding="utf-8", errors="ignore")
        source_text = source.read_text(encoding="utf-8", errors="ignore")
        if f'#include "{expected_include}"' not in source_text:
            violations.append((source, 0, "model decode runtime does not include local header"))
        if masked_score not in header_text:
            violations.append((header, 0, f"missing local masked score {masked_score}"))
        for symbol in (
            f"{family}_select_argmax_token",
            f"{family}_sample_token_topk",
            f"{family}_select_topk_tokens",
            f"{family}_build_attention_mask",
        ):
            if symbol not in header_text:
                violations.append((header, 0, f"missing declaration for {symbol}"))
            if symbol not in source_text:
                violations.append((source, 0, f"missing definition/use for {symbol}"))

    personaplex_manifest = (RUNTIME_MODELS / "personaplex" / "MODEL.toml").read_text(
        encoding="utf-8", errors="ignore"
    )
    expected_test_entry = (
        "test_personaplex_decode_runtime|test_personaplex_decode_runtime.cpp|_|decode_runtime.cpp|_"
    )
    if expected_test_entry not in personaplex_manifest:
        violations.append(
            (
                RUNTIME_MODELS / "personaplex" / "MODEL.toml",
                0,
                "missing model-owned decode runtime unit test entry",
            )
        )

    personaplex_test = (
        REPO_ROOT
        / "tests"
        / "cpp"
        / "models"
        / "personaplex"
        / ("test_personaplex_decode_runtime.cpp")
    )
    test_text = personaplex_test.read_text(encoding="utf-8", errors="ignore")
    if "runtime/models/personaplex/decode_runtime.h" not in test_text:
        violations.append((personaplex_test, 0, "test does not include personaplex decode runtime"))
    if "trtmc::personaplex_select_argmax_token" not in test_text:
        violations.append(
            (personaplex_test, 0, "test does not exercise personaplex decode symbols")
        )

    assert not violations, _format_violations(violations)


def test_shared_trt_engine_lifecycle_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep decoder tensor-name expansion with the owning model runtime.
    Preconditions: decoder, VL, and Magpie runtime folders own tensor_names.h.
    Postconditions: the old shared TRT engine lifecycle helper and root tests
    are absent, and model call sites use family-prefixed tensor-name helpers.
    """
    required_families = (
        "bloom",
        "codegen",
        "deepseek_ocr",
        "deepseek_v2",
        "falcon",
        "gemma",
        "glm",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "gpt_oss",
        "granite",
        "internlm",
        "internvl",
        "lance",
        "llama",
        "locateanything",
        "magpie",
        "mistral",
        "mixtral",
        "nemotron",
        "nemotron_labs_diffusion",
        "olmo",
        "olmo2",
        "opt",
        "phi",
        "phi4_multimodal",
        "phi_moe",
        "qwen",
        "qwen_moe",
        "qwen_vl",
        "stablelm",
        "starcoder2",
        "xglm",
    )
    retired_paths = (
        REPO_ROOT / "src" / "runtime" / "core" / "trt_engine_lifecycle.h",
        REPO_ROOT / "src" / "runtime" / "core" / "trt_engine_lifecycle.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_trt_engine_lifecycle.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_trt_engine_lifecycle_fake_engine.cpp",
    )
    forbidden_unowned_symbols = (
        re.compile(r"(?<![A-Za-z0-9_])expand_layer_name\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])layer_tensor_name\s*\("),
        re.compile(r"\bDecoderStepEngine\b"),
        re.compile(r"\bhas_all_required_tensors\b"),
        re.compile(r"\bkDefaultMaxCacheLength\b"),
    )
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared TRT engine lifecycle artifact must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in (
        "src/runtime/core/trt_engine_lifecycle.cpp",
        "test_trt_engine_lifecycle",
    ):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired {term}"))

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "runtime/core/trt_engine_lifecycle.h" in text:
                violations.append((path, 0, "includes retired shared TRT engine lifecycle header"))
            model_owned_path = (
                RUNTIME_MODELS in path.parents
                or (REPO_ROOT / "tests" / "cpp" / "models") in path.parents
            )
            if model_owned_path:
                continue
            for pattern in forbidden_unowned_symbols:
                if pattern.search(text):
                    violations.append((path, 0, f"uses shared lifecycle symbol {pattern.pattern}"))

    for family in required_families:
        header = RUNTIME_MODELS / family / "tensor_names.h"
        if not header.is_file():
            violations.append((header, 0, "missing model-owned tensor name helper"))
            continue
        text = header.read_text(encoding="utf-8", errors="ignore")
        for symbol in (f"{family}_expand_layer_name", f"{family}_layer_tensor_name"):
            if symbol not in text:
                violations.append((header, 0, f"missing {symbol}"))

    qwen_manifest = (RUNTIME_MODELS / "qwen" / "MODEL.toml").read_text(
        encoding="utf-8", errors="ignore"
    )
    if "test_qwen_tensor_names|test_qwen_tensor_names.cpp|_|_|_" not in qwen_manifest:
        violations.append((RUNTIME_MODELS / "qwen" / "MODEL.toml", 0, "missing tensor-name test"))
    qwen_test = REPO_ROOT / "tests" / "cpp" / "models" / "qwen" / "test_qwen_tensor_names.cpp"
    if not qwen_test.is_file():
        violations.append((qwen_test, 0, "missing model-owned tensor-name test"))
    else:
        test_text = qwen_test.read_text(encoding="utf-8", errors="ignore")
        if "qwen_expand_layer_name" not in test_text or "qwen_layer_tensor_name" not in test_text:
            violations.append((qwen_test, 0, "tensor-name test does not exercise qwen helpers"))

    assert not violations, _format_violations(violations)


def test_shared_sampler_cuda_kernels_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep GPU sampling kernels with the model-owned sampler copies.
    Preconditions: decoder text and VL runtime folders own sampler.cpp.
    Postconditions: old shared sampler CUDA kernels are absent, trtmc_core does
    not link them, and every GPU-capable sampler family uses local prefixed
    kernel helpers.
    """
    required_families = (
        "bloom",
        "codegen",
        "deepseek_ocr",
        "deepseek_v2",
        "falcon",
        "gemma",
        "glm",
        "gpt2",
        "gpt_neo",
        "gpt_neox",
        "gpt_oss",
        "granite",
        "internlm",
        "internvl",
        "lance",
        "llama",
        "locateanything",
        "mistral",
        "mixtral",
        "nemotron",
        "nemotron_labs_diffusion",
        "olmo",
        "olmo2",
        "opt",
        "phi",
        "phi4_multimodal",
        "phi_moe",
        "qwen",
        "qwen_moe",
        "qwen_vl",
        "stablelm",
        "starcoder2",
        "xglm",
    )
    retired_paths = (
        REPO_ROOT / "src" / "runtime" / "core" / "argmax_kernel.h",
        REPO_ROOT / "src" / "runtime" / "core" / "argmax_kernel.cu",
        REPO_ROOT / "src" / "runtime" / "core" / "sparse_multinomial_kernel.h",
        REPO_ROOT / "src" / "runtime" / "core" / "sparse_multinomial_kernel.cu",
    )
    retired_includes = (
        "runtime/core/argmax_kernel.h",
        "runtime/core/sparse_multinomial_kernel.h",
    )
    forbidden_unowned_symbols = (
        re.compile(r"(?<![A-Za-z0-9_])gpu_argmax\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])gpu_sparse_torch_multinomial_exact\s*\("),
        re.compile(r"(?<![A-Za-z0-9_])compute_torch_multinomial_execution_policy\s*\("),
        re.compile(r"\bTorchMultinomialExecutionPolicy\b"),
    )
    violations = []

    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared sampler CUDA kernel must be model-owned"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in (
        "src/runtime/core/argmax_kernel.cu",
        "src/runtime/core/sparse_multinomial_kernel.cu",
    ):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"trtmc_core links retired {term}"))

    for family in required_families:
        prefix = _model_prefix(family)
        model_dir = RUNTIME_MODELS / family
        sampler = model_dir / "sampler.cpp"
        local_files = (
            model_dir / "argmax_kernel.h",
            model_dir / "argmax_kernel.cu",
            model_dir / "sparse_multinomial_kernel.h",
            model_dir / "sparse_multinomial_kernel.cu",
        )
        expected = (
            f"{family}_gpu_argmax",
            f"{prefix}TorchMultinomialExecutionPolicy",
            f"{family}_compute_torch_multinomial_execution_policy",
            f"{family}_gpu_sparse_torch_multinomial_exact",
        )

        for path in local_files:
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned sampler kernel file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for include in retired_includes:
                if include in text:
                    violations.append((path, 0, f"includes retired {include}"))
            for needle in expected:
                if ("argmax" in path.name and "argmax" not in needle) or (
                    "sparse_multinomial" in path.name and "argmax" in needle
                ):
                    continue
                if needle not in text:
                    violations.append((path, 0, f"missing {needle}"))
            stripped = _strip_cpp_comments(text)
            for pattern in forbidden_unowned_symbols:
                if pattern.search(stripped):
                    violations.append(
                        (path, 0, f"uses unowned sampler kernel symbol {pattern.pattern}")
                    )

        sampler_text = sampler.read_text(encoding="utf-8", errors="ignore")
        for include in retired_includes:
            if include in sampler_text:
                violations.append((sampler, 0, f"includes retired {include}"))
        for include in (
            f"runtime/models/{family}/argmax_kernel.h",
            f"runtime/models/{family}/sparse_multinomial_kernel.h",
        ):
            if include not in sampler_text:
                violations.append((sampler, 0, f"missing local include {include}"))
        for needle in expected:
            if needle not in sampler_text:
                violations.append((sampler, 0, f"missing {needle}"))
        stripped = _strip_cpp_comments(sampler_text)
        for pattern in forbidden_unowned_symbols:
            if pattern.search(stripped):
                violations.append(
                    (sampler, 0, f"uses unowned sampler kernel symbol {pattern.pattern}")
                )

    for root in (REPO_ROOT / "src", REPO_ROOT / "include", CPP_TESTS):
        for path in _cpp_files_under(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for include in retired_includes:
                if include in text:
                    violations.append((path, 0, f"includes retired {include}"))

    assert not violations, _format_violations(violations)


def test_text_decoder_sampler_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep decoder-only text sampling behavior local to each model
    family instead of sharing a single runtime sampler implementation.
    Preconditions: decoder text runtime folders own their pipeline copies.
    Postconditions: each decoder text family owns sampler source/header files,
    its pipeline includes the local sampler, and unprefixed shared sampler
    symbols are not used by decoder text pipelines.
    """
    required_owned = {
        "bloom": ("BloomISampler", "BloomSamplingParams", "create_bloom_sampler"),
        "codegen": ("CodegenISampler", "CodegenSamplingParams", "create_codegen_sampler"),
        "deepseek_v2": (
            "DeepseekV2ISampler",
            "DeepseekV2SamplingParams",
            "create_deepseek_v2_sampler",
        ),
        "falcon": ("FalconISampler", "FalconSamplingParams", "create_falcon_sampler"),
        "gemma": ("GemmaISampler", "GemmaSamplingParams", "create_gemma_sampler"),
        "glm": ("GlmISampler", "GlmSamplingParams", "create_glm_sampler"),
        "gpt2": ("Gpt2ISampler", "Gpt2SamplingParams", "create_gpt2_sampler"),
        "gpt_neo": ("GptNeoISampler", "GptNeoSamplingParams", "create_gpt_neo_sampler"),
        "gpt_neox": (
            "GptNeoxISampler",
            "GptNeoxSamplingParams",
            "create_gpt_neox_sampler",
        ),
        "gpt_oss": ("GptOssISampler", "GptOssSamplingParams", "create_gpt_oss_sampler"),
        "granite": ("GraniteISampler", "GraniteSamplingParams", "create_granite_sampler"),
        "internlm": ("InternlmISampler", "InternlmSamplingParams", "create_internlm_sampler"),
        "llama": ("LlamaISampler", "LlamaSamplingParams", "create_llama_sampler"),
        "mistral": ("MistralISampler", "MistralSamplingParams", "create_mistral_sampler"),
        "mixtral": ("MixtralISampler", "MixtralSamplingParams", "create_mixtral_sampler"),
        "nemotron": (
            "NemotronISampler",
            "NemotronSamplingParams",
            "create_nemotron_sampler",
        ),
        "nemotron_labs_diffusion": (
            "NemotronLabsDiffusionISampler",
            "NemotronLabsDiffusionSamplingParams",
            "create_nemotron_labs_diffusion_sampler",
        ),
        "olmo": ("OlmoISampler", "OlmoSamplingParams", "create_olmo_sampler"),
        "olmo2": ("Olmo2ISampler", "Olmo2SamplingParams", "create_olmo2_sampler"),
        "opt": ("OptISampler", "OptSamplingParams", "create_opt_sampler"),
        "phi": ("PhiISampler", "PhiSamplingParams", "create_phi_sampler"),
        "phi_moe": ("PhiMoeISampler", "PhiMoeSamplingParams", "create_phi_moe_sampler"),
        "qwen": ("QwenISampler", "QwenSamplingParams", "create_qwen_sampler"),
        "qwen_moe": ("QwenMoeISampler", "QwenMoeSamplingParams", "create_qwen_moe_sampler"),
        "stablelm": (
            "StablelmISampler",
            "StablelmSamplingParams",
            "create_stablelm_sampler",
        ),
        "starcoder2": (
            "Starcoder2ISampler",
            "Starcoder2SamplingParams",
            "create_starcoder2_sampler",
        ),
        "xglm": ("XglmISampler", "XglmSamplingParams", "create_xglm_sampler"),
    }
    forbidden_patterns = (
        re.compile(r"\bISampler\b"),
        re.compile(r"\bSamplingParams\b"),
        re.compile(r"\bSampleResult\b"),
        re.compile(r"\bLogitsLocation\b"),
        re.compile(r"\bcreate_sampler\s*\("),
        re.compile(r"\bcreate_gpu_greedy_sampler\s*\("),
        re.compile(r"\bsampling_params_from_config\s*\("),
    )
    violations = []

    for family, (sampler_class, params_class, factory_name) in required_owned.items():
        model_dir = RUNTIME_MODELS / family
        sampler_header = model_dir / "sampler.h"
        sampler_source = model_dir / "sampler.cpp"
        pipeline_header = model_dir / "pipeline.h"
        pipeline_source = model_dir / "pipeline.cpp"
        local_include = f"runtime/models/{family}/sampler.h"
        gpu_factory = factory_name.replace("_sampler", "_gpu_greedy_sampler")

        for path in (sampler_header, sampler_source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned sampler file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in (sampler_class, params_class, factory_name, gpu_factory):
                if needle not in text:
                    violations.append((path, 0, f"missing {needle}"))
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "model-owned sampler includes shared sampler header"))
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses shared sampler symbol {pattern.pattern}"))

        for path in (pipeline_header, pipeline_source):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "text decoder pipeline includes shared sampler header"))
            if path == pipeline_header and local_include not in text:
                violations.append((path, 0, f"pipeline missing local include {local_include}"))
            if sampler_class not in text:
                violations.append((path, 0, f"pipeline does not use {sampler_class}"))
            if params_class not in text:
                violations.append((path, 0, f"pipeline does not use {params_class}"))
            if path == pipeline_source and factory_name not in text:
                violations.append((path, 0, f"pipeline does not call {factory_name}"))
            if path == pipeline_source and gpu_factory not in text:
                violations.append((path, 0, f"pipeline does not call {gpu_factory}"))
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses shared sampler symbol {pattern.pattern}"))

    assert not violations, _format_violations(violations)


def test_text_decoder_pipeline_types_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep decoder-only text pipeline/config C++ types owned by each
    model family instead of sharing unprefixed runtime type names.
    Preconditions: decoder text runtime folders own their pipeline copies.
    Postconditions: each decoder text family exposes a family-prefixed
    pipeline/config type, and unprefixed TextGenerationPipeline/TextGenConfig
    symbols do not appear in decoder text runtime files.
    """
    required_owned = {
        "bloom": ("BloomTextGenerationPipeline", "BloomTextGenConfig"),
        "codegen": ("CodegenTextGenerationPipeline", "CodegenTextGenConfig"),
        "deepseek_v2": ("DeepseekV2TextGenerationPipeline", "DeepseekV2TextGenConfig"),
        "falcon": ("FalconTextGenerationPipeline", "FalconTextGenConfig"),
        "gemma": ("GemmaTextGenerationPipeline", "GemmaTextGenConfig"),
        "glm": ("GlmTextGenerationPipeline", "GlmTextGenConfig"),
        "gpt2": ("Gpt2TextGenerationPipeline", "Gpt2TextGenConfig"),
        "gpt_neo": ("GptNeoTextGenerationPipeline", "GptNeoTextGenConfig"),
        "gpt_neox": ("GptNeoxTextGenerationPipeline", "GptNeoxTextGenConfig"),
        "gpt_oss": ("GptOssTextGenerationPipeline", "GptOssTextGenConfig"),
        "granite": ("GraniteTextGenerationPipeline", "GraniteTextGenConfig"),
        "internlm": ("InternlmTextGenerationPipeline", "InternlmTextGenConfig"),
        "llama": ("LlamaTextGenerationPipeline", "LlamaTextGenConfig"),
        "mistral": ("MistralTextGenerationPipeline", "MistralTextGenConfig"),
        "mixtral": ("MixtralTextGenerationPipeline", "MixtralTextGenConfig"),
        "nemotron": ("NemotronTextGenerationPipeline", "NemotronTextGenConfig"),
        "nemotron_labs_diffusion": (
            "NemotronLabsDiffusionTextGenerationPipeline",
            "NemotronLabsDiffusionTextGenConfig",
        ),
        "olmo": ("OlmoTextGenerationPipeline", "OlmoTextGenConfig"),
        "olmo2": ("Olmo2TextGenerationPipeline", "Olmo2TextGenConfig"),
        "opt": ("OptTextGenerationPipeline", "OptTextGenConfig"),
        "phi": ("PhiTextGenerationPipeline", "PhiTextGenConfig"),
        "phi_moe": ("PhiMoeTextGenerationPipeline", "PhiMoeTextGenConfig"),
        "qwen": ("QwenTextGenerationPipeline", "QwenTextGenConfig"),
        "qwen_moe": ("QwenMoeTextGenerationPipeline", "QwenMoeTextGenConfig"),
        "stablelm": ("StablelmTextGenerationPipeline", "StablelmTextGenConfig"),
        "starcoder2": ("Starcoder2TextGenerationPipeline", "Starcoder2TextGenConfig"),
        "xglm": ("XglmTextGenerationPipeline", "XglmTextGenConfig"),
    }
    forbidden_patterns = (
        re.compile(r"\bTextGenerationPipeline\b"),
        re.compile(r"\bTextGenConfig\b"),
    )
    violations = []

    for family, (pipeline_class, config_class) in required_owned.items():
        model_dir = RUNTIME_MODELS / family
        pipeline_header = model_dir / "pipeline.h"
        pipeline_source = model_dir / "pipeline.cpp"
        plugin_source = model_dir / "plugin.cpp"

        header_text = pipeline_header.read_text(encoding="utf-8", errors="ignore")
        if f"class {pipeline_class}" not in header_text:
            violations.append((pipeline_header, 0, f"missing {pipeline_class}"))
        if f"struct {config_class}" not in header_text:
            violations.append((pipeline_header, 0, f"missing {config_class}"))

        source_text = pipeline_source.read_text(encoding="utf-8", errors="ignore")
        if f"{pipeline_class}::" not in source_text:
            violations.append((pipeline_source, 0, f"missing {pipeline_class} definitions"))
        if config_class not in source_text:
            violations.append((pipeline_source, 0, f"missing {config_class} usage"))

        plugin_text = plugin_source.read_text(encoding="utf-8", errors="ignore")
        if f"std::make_unique<{pipeline_class}>" not in plugin_text:
            violations.append((plugin_source, 0, f"plugin does not instantiate {pipeline_class}"))
        if config_class not in plugin_text:
            violations.append((plugin_source, 0, f"plugin does not use {config_class}"))

        for path, text in (
            (pipeline_header, header_text),
            (pipeline_source, source_text),
            (plugin_source, plugin_text),
        ):
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses unowned text type {pattern.pattern}"))

    llama_test = REPO_ROOT / "tests" / "cpp" / "models" / "llama" / "test_llama_pipeline.cpp"
    if llama_test.exists():
        test_text = llama_test.read_text(encoding="utf-8", errors="ignore")
        if "LlamaTextGenerationPipeline" not in test_text or "LlamaTextGenConfig" not in test_text:
            violations.append((llama_test, 0, "Llama test does not use Llama-owned text types"))
        for pattern in forbidden_patterns:
            if pattern.search(test_text):
                violations.append((llama_test, 0, f"uses unowned text type {pattern.pattern}"))

    assert not violations, _format_violations(violations)


def test_vision_language_sampler_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep vision-language text sampling behavior local to each model
    family instead of sharing the runtime sampler implementation.
    Preconditions: VL runtime folders own their pipeline copies.
    Postconditions: each VL family owns sampler source/header files, its
    pipeline includes the local sampler, and unprefixed shared sampler symbols
    are not used by VL pipelines.
    """
    required_owned = {
        "deepseek_ocr": (
            "DeepseekOcrISampler",
            "DeepseekOcrSamplingParams",
            "create_deepseek_ocr_sampler",
        ),
        "internvl": ("InternVlISampler", "InternVlSamplingParams", "create_internvl_sampler"),
        "lance": ("LanceISampler", "LanceSamplingParams", "create_lance_sampler"),
        "locateanything": (
            "LocateAnythingISampler",
            "LocateAnythingSamplingParams",
            "create_locateanything_sampler",
        ),
        "phi4_multimodal": (
            "Phi4MultimodalISampler",
            "Phi4MultimodalSamplingParams",
            "create_phi4_multimodal_sampler",
        ),
        "qwen_vl": ("QwenVlISampler", "QwenVlSamplingParams", "create_qwen_vl_sampler"),
    }
    forbidden_patterns = (
        re.compile(r"\bISampler\b"),
        re.compile(r"\bSamplingParams\b"),
        re.compile(r"\bSampleResult\b"),
        re.compile(r"\bLogitsLocation\b"),
        re.compile(r"\bcreate_sampler\s*\("),
        re.compile(r"\bcreate_gpu_greedy_sampler\s*\("),
        re.compile(r"\bsampling_params_from_config\s*\("),
    )
    violations = []

    for family, (sampler_class, params_class, factory_name) in required_owned.items():
        model_dir = RUNTIME_MODELS / family
        sampler_header = model_dir / "sampler.h"
        sampler_source = model_dir / "sampler.cpp"
        pipeline_header = model_dir / "pipeline.h"
        pipeline_source = model_dir / "pipeline.cpp"
        local_include = f"runtime/models/{family}/sampler.h"
        gpu_factory = factory_name.replace("_sampler", "_gpu_greedy_sampler")

        for path in (sampler_header, sampler_source):
            if not path.is_file():
                violations.append((path, 0, f"missing {family}-owned sampler file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in (sampler_class, params_class, factory_name, gpu_factory):
                if needle not in text:
                    violations.append((path, 0, f"missing {needle}"))
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "model-owned sampler includes shared sampler header"))
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses shared sampler symbol {pattern.pattern}"))

        for path in (pipeline_header, pipeline_source):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "trtmc/runtime/sampler.h" in text:
                violations.append((path, 0, "VL pipeline includes shared sampler header"))
            if path == pipeline_header and local_include not in text:
                violations.append((path, 0, f"pipeline missing local include {local_include}"))
            if sampler_class not in text:
                violations.append((path, 0, f"pipeline does not use {sampler_class}"))
            if params_class not in text:
                violations.append((path, 0, f"pipeline does not use {params_class}"))
            if path == pipeline_source and factory_name not in text:
                violations.append((path, 0, f"pipeline does not call {factory_name}"))
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses shared sampler symbol {pattern.pattern}"))

    assert not violations, _format_violations(violations)


def test_vision_language_pipeline_types_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep vision-language pipeline/config/preprocessor C++ types owned
    by each model family instead of sharing unprefixed VL type names.
    Preconditions: VL runtime folders own their pipeline and preprocessor
    copies.
    Postconditions: each VL family exposes family-prefixed pipeline,
    config, preprocessor, and image-preprocessor helper symbols.
    """
    required_owned = {
        "deepseek_ocr": ("DeepseekOcr", "deepseek_ocr"),
        "internvl": ("InternVl", "internvl"),
        "lance": ("Lance", "lance"),
        "locateanything": ("LocateAnything", "locateanything"),
        "phi4_multimodal": ("Phi4Multimodal", "phi4_multimodal"),
        "qwen_vl": ("QwenVl", "qwen_vl"),
    }
    forbidden_patterns = (
        re.compile(r"\bVLPipeline\b"),
        re.compile(r"\bVLConfig\b"),
        re.compile(r"\bVLPreprocessConfig\b"),
        re.compile(r"\bPreprocessedImage\b"),
        re.compile(r"\bdecode_image_rgb\b"),
        re.compile(r"\bpreprocess_decoded_image\b"),
        re.compile(r"\bload_and_preprocess_image\b"),
        re.compile(r"\bformat_vl_prompt\b"),
        re.compile(r"\bparse_vl_preprocess_config\b"),
    )
    violations = []

    for family, (prefix, function_prefix) in required_owned.items():
        model_dir = RUNTIME_MODELS / family
        pipeline_header = model_dir / "pipeline.h"
        pipeline_source = model_dir / "pipeline.cpp"
        plugin_source = model_dir / "plugin.cpp"
        preprocessor_header = model_dir / "image_preprocessor.h"
        preprocessor_source = model_dir / "image_preprocessor.cpp"
        pipeline_class = f"{prefix}Pipeline"
        config_class = f"{prefix}Config"
        preprocess_config = f"{prefix}PreprocessConfig"
        preprocessed_image = f"{prefix}PreprocessedImage"
        parse_config = f"{function_prefix}_parse_preprocess_config"
        format_prompt = f"{function_prefix}_format_prompt"
        preprocess_decoded = f"{function_prefix}_preprocess_decoded_image"
        load_preprocess = f"{function_prefix}_load_and_preprocess_image"
        decode_image = f"{function_prefix}_decode_image_rgb"

        header_text = pipeline_header.read_text(encoding="utf-8", errors="ignore")
        if f"class {pipeline_class}" not in header_text:
            violations.append((pipeline_header, 0, f"missing {pipeline_class}"))
        if f"struct {config_class}" not in header_text:
            violations.append((pipeline_header, 0, f"missing {config_class}"))
        if preprocess_config not in header_text:
            violations.append((pipeline_header, 0, f"pipeline missing {preprocess_config}"))

        source_text = pipeline_source.read_text(encoding="utf-8", errors="ignore")
        if f"{pipeline_class}::" not in source_text:
            violations.append((pipeline_source, 0, f"missing {pipeline_class} definitions"))
        for needle in (config_class, preprocess_config, preprocessed_image, format_prompt):
            if needle not in source_text:
                violations.append((pipeline_source, 0, f"missing {needle}"))

        plugin_text = plugin_source.read_text(encoding="utf-8", errors="ignore")
        if f"std::make_unique<{pipeline_class}>" not in plugin_text:
            violations.append((plugin_source, 0, f"plugin does not instantiate {pipeline_class}"))
        for needle in (config_class, parse_config):
            if needle not in plugin_text:
                violations.append((plugin_source, 0, f"missing {needle}"))

        preprocessor_header_text = preprocessor_header.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            preprocess_config,
            preprocessed_image,
            decode_image,
            preprocess_decoded,
            load_preprocess,
            format_prompt,
            parse_config,
        ):
            if needle not in preprocessor_header_text:
                violations.append((preprocessor_header, 0, f"missing {needle}"))

        preprocessor_source_text = preprocessor_source.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            preprocess_config,
            preprocessed_image,
            decode_image,
            preprocess_decoded,
            load_preprocess,
            format_prompt,
            parse_config,
        ):
            if needle not in preprocessor_source_text:
                violations.append((preprocessor_source, 0, f"missing {needle}"))

        test_path = (
            REPO_ROOT / "tests" / "cpp" / "models" / family / f"test_{family}_vl_pipeline.cpp"
        )
        if test_path.exists():
            test_text = test_path.read_text(encoding="utf-8", errors="ignore")
            for needle in (pipeline_class, config_class, preprocess_config):
                if needle not in test_text:
                    violations.append((test_path, 0, f"test missing {needle}"))
        else:
            test_text = ""

        for path, text in (
            (pipeline_header, header_text),
            (pipeline_source, source_text),
            (plugin_source, plugin_text),
            (preprocessor_header, preprocessor_header_text),
            (preprocessor_source, preprocessor_source_text),
            (test_path, test_text),
        ):
            if not path.exists():
                continue
            for pattern in forbidden_patterns:
                if pattern.search(text):
                    violations.append((path, 0, f"uses unowned VL symbol {pattern.pattern}"))

    assert not violations, _format_violations(violations)


def test_shared_device_ops_header_is_retired() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep audio/codebook GPU decode kernels model-owned instead of
    advertising unused shared device operation declarations.
    Preconditions: Magpie owns the concrete CUDA kernels it uses.
    Postconditions: the old public shared device_ops header is absent and no
    source/test file includes it.
    """
    retired_header = REPO_ROOT / "include" / "trtmc" / "runtime" / "device_ops.h"
    violations = []
    if retired_header.exists():
        violations.append((retired_header, 0, "shared device_ops header must be retired"))

    forbidden_include = "trtmc/runtime/device_ops.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            if forbidden_include in path.read_text(encoding="utf-8", errors="ignore"):
                violations.append((path, 0, "includes retired shared device_ops header"))

    assert not violations, _format_violations(violations)


def test_shared_device_kv_cache_runtime_is_retired() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent legacy shared device-side decoder cache behavior from
    becoming the common implementation for model families.
    Preconditions: active decoder pipelines own their runtime execution loops.
    Postconditions: the old DeviceKvCache helper, update plan, and root tests
    are absent, and no source/test file includes the retired header.
    """
    retired_paths = (
        REPO_ROOT / "src" / "runtime" / "core" / "device_kv_cache.h",
        REPO_ROOT / "src" / "runtime" / "core" / "device_kv_cache.cpp",
        REPO_ROOT / "src" / "runtime" / "core" / "device_kv_cache_update_plan.h",
        REPO_ROOT / "tests" / "cpp" / "test_device_kv_cache.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_device_resources.cpp",
    )
    violations = []
    for path in retired_paths:
        if path.exists():
            violations.append((path, 0, "shared device KV cache artifact must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for term in (
        "device_kv_cache.cpp",
        "device_kv_cache_update_plan.h",
        "test_device_kv_cache",
        "test_device_resources",
    ):
        if term in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired {term}"))

    forbidden_terms = (
        "runtime/core/device_kv_cache.h",
        "runtime/core/device_kv_cache_update_plan.h",
        '"DeviceKvCache"',
        "run_decoder_step_device",
        "DeviceResources",
    )
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests" / "cpp"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden_terms:
                if term in text:
                    violations.append((path, 0, f"references retired shared cache term {term}"))

    assert not violations, _format_violations(violations)


def test_recurrent_step_contracts_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep recurrent validation/output initialization contracts and TRT
    tensor binding behavior with the owning recurrent model family.
    Preconditions: recurrent model families carry local contract copies.
    Postconditions: retired shared recurrent domain headers and root unit test
    are absent, CMake no longer names them, and family code includes only
    model-owned recurrent contract headers.
    """
    retired_shared = RUNTIME_DOMAINS / "recurrent" / "recurrent_step_contracts.h"
    retired_tensor_bindings = RUNTIME_DOMAINS / "recurrent" / "recurrent_tensor_bindings.h"
    retired_test = REPO_ROOT / "tests" / "cpp" / "test_recurrent_step_contracts.cpp"
    required_owned = {
        "mamba": (
            RUNTIME_MODELS / "mamba" / "mamba_recurrent_step_contracts.h",
            "mamba_recurrent",
            REPO_ROOT
            / "tests"
            / "cpp"
            / "models"
            / "mamba"
            / "test_mamba_recurrent_output_initializers.cpp",
        ),
        "rwkv": (
            RUNTIME_MODELS / "rwkv" / "rwkv_recurrent_step_contracts.h",
            "rwkv_recurrent",
            REPO_ROOT
            / "tests"
            / "cpp"
            / "models"
            / "rwkv"
            / "test_rwkv_recurrent_output_initializers.cpp",
        ),
        "nemotron_h": (
            RUNTIME_MODELS / "nemotron_h" / "nemotron_h_recurrent_step_contracts.h",
            "nemotron_h_recurrent",
            REPO_ROOT
            / "tests"
            / "cpp"
            / "models"
            / "nemotron_h"
            / "test_nemotron_h_recurrent_output_initializers.cpp",
        ),
        "qwen3_5": (
            RUNTIME_MODELS / "qwen3_5" / "qwen3_5_recurrent_step_contracts.h",
            "qwen3_5_recurrent",
            REPO_ROOT
            / "tests"
            / "cpp"
            / "models"
            / "qwen3_5"
            / "test_qwen3_5_recurrent_output_initializers.cpp",
        ),
    }
    violations = []
    if retired_shared.exists():
        violations.append((retired_shared, 0, "shared recurrent step contract must be retired"))
    if retired_tensor_bindings.exists():
        violations.append(
            (retired_tensor_bindings, 0, "shared recurrent tensor bindings must be retired")
        )
    if retired_test.exists():
        violations.append((retired_test, 0, "root recurrent contract test must be model-owned"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    if "test_recurrent_step_contracts" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "CMake references retired recurrent contract test"))

    required_terms = (
        "validate_state_layer_count",
        "struct StateTensorView",
        "validate_state_tensor_sizes",
        "initialize_layer_outputs",
    )
    required_test_terms = (
        "validate_state_layer_count",
        "StateTensorView",
        "validate_state_tensor_sizes",
        "initialize_layer_outputs",
    )
    for family, (header, namespace, test_path) in required_owned.items():
        if not header.is_file():
            violations.append((header, 0, f"missing {family}-owned recurrent contract"))
            continue
        text = header.read_text(encoding="utf-8", errors="ignore")
        if f"namespace {namespace}" not in text:
            violations.append((header, 0, f"missing {namespace} namespace"))
        for term in required_terms:
            if term not in text:
                violations.append((header, 0, f"missing recurrent contract term {term}"))

        if not test_path.is_file():
            violations.append((test_path, 0, f"missing {family}-owned recurrent contract test"))
            continue
        test_text = test_path.read_text(encoding="utf-8", errors="ignore")
        if f"trtmc::{namespace}" not in test_text:
            violations.append((test_path, 0, f"test does not exercise trtmc::{namespace}"))
        for term in required_test_terms:
            if term not in test_text:
                violations.append((test_path, 0, f"test missing recurrent contract term {term}"))

    forbidden_includes = (
        "runtime/domains/recurrent/recurrent_step_contracts.h",
        "runtime/domains/recurrent/recurrent_tensor_bindings.h",
    )
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for forbidden_include in forbidden_includes:
                if forbidden_include in text:
                    violations.append((path, 0, "includes retired shared recurrent domain header"))

    assert not violations, _format_violations(violations)


def test_diffusion_config_and_weight_types_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep diffusion config/weight/result structs out of shared runtime domains.
    Preconditions: diffusion runtime families own their type headers.
    Postconditions: the retired shared diffusion type header is absent, model
    headers carry copied family-owned type names, and no source/test file
    includes the retired shared header.
    """
    retired_shared = RUNTIME_DIFFUSION_DOMAINS / "diffusion_types.h"
    required_owned = {
        "flux": (
            RUNTIME_MODELS / "flux" / "flux_diffusion_types.h",
            ("FluxDiffusionConfig", "FluxPreprocessorWeights"),
        ),
        "wan": (
            RUNTIME_MODELS / "wan" / "wan_diffusion_types.h",
            ("WanDiffusionConfig", "WanPreprocessorWeights", "WanVideoResult"),
        ),
        "pixart": (
            RUNTIME_MODELS / "pixart" / "pixart_diffusion_types.h",
            ("PixArtDiffusionConfig", "PixArtPreprocessorWeights", "PixArtVideoResult"),
        ),
        "z_image": (
            RUNTIME_MODELS / "z_image" / "z_image_diffusion_types.h",
            ("ZImageDiffusionConfig", "ZImageCommonPreprocessorWeights"),
        ),
        "ltx_video": (
            RUNTIME_MODELS / "ltx_video" / "ltx_video_diffusion_types.h",
            ("LTXVideoDiffusionConfig", "LTXVideoPreprocessorWeights", "LTXVideoResult"),
        ),
        "qwen_image": (
            RUNTIME_MODELS / "qwen_image" / "qwen_image_diffusion_types.h",
            ("QwenImageCommonDiffusionConfig", "QwenImageCommonPreprocessorWeights"),
        ),
    }
    violations = []
    if retired_shared.exists():
        violations.append((retired_shared, 0, "shared diffusion types header must be retired"))

    for family, (path, required_types) in required_owned.items():
        if not path.is_file():
            violations.append((path, 0, f"missing {family}-owned diffusion type header"))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for type_name in required_types:
            if f"struct {type_name}" not in text:
                violations.append((path, 0, f"missing family-owned type {type_name}"))
        for retired_type in (
            "struct DiffusionConfig",
            "struct PreprocessorWeights",
            "struct VideoResult",
        ):
            if retired_type in text:
                violations.append((path, 0, f"keeps generic retired type {retired_type}"))

    forbidden_include = "runtime/domains/diffusion/diffusion_types.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if forbidden_include in text:
                violations.append((path, 0, "includes retired shared diffusion type header"))

    assert not violations, _format_violations(violations)


def test_qwen_image_scheduler_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Qwen Image scheduler behavior out of shared runtime core.
    Preconditions: Qwen Image owns the C++ FlowMatchEulerScheduler it uses.
    Postconditions: retired shared scheduler files are absent, build files do
    not reference them, and only Qwen Image runtime/test files include the
    model-owned scheduler.
    """
    retired_shared_paths = (
        REPO_ROOT / "include" / "trtmc" / "runtime" / "scheduler.h",
        REPO_ROOT / "src" / "runtime" / "core" / "flow_match_euler_scheduler.cpp",
        REPO_ROOT / "tests" / "cpp" / "test_flow_match_scheduler.cpp",
    )
    required_owned_paths = (
        RUNTIME_MODELS / "qwen_image" / "qwen_image_scheduler.h",
        RUNTIME_MODELS / "qwen_image" / "qwen_image_scheduler.cpp",
        REPO_ROOT
        / "tests"
        / "cpp"
        / "models"
        / "qwen_image"
        / "test_qwen_image_flow_match_scheduler.cpp",
    )
    violations = []
    for path in retired_shared_paths:
        if path.exists():
            violations.append((path, 0, "shared scheduler file must be retired"))
    for path in required_owned_paths:
        if not path.is_file():
            violations.append((path, 0, "missing Qwen Image-owned scheduler file"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    for needle in (
        "src/runtime/core/flow_match_euler_scheduler.cpp",
        "test_flow_match_scheduler",
    ):
        if needle in cmake_text:
            violations.append((CMAKE_ROOT, 0, f"CMake references retired scheduler {needle}"))

    manifest = RUNTIME_MODELS / "qwen_image" / "MODEL.toml"
    manifest_text = manifest.read_text(encoding="utf-8", errors="ignore")
    if (
        "test_qwen_image_flow_match_scheduler|test_qwen_image_flow_match_scheduler.cpp"
        not in manifest_text
    ):
        violations.append((manifest, 0, "missing Qwen Image scheduler runtime test entry"))
    if "qwen_image_scheduler.cpp" not in manifest_text:
        violations.append((manifest, 0, "missing Qwen Image scheduler test source entry"))

    include_needle = "trtmc/runtime/scheduler.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if include_needle in text:
                violations.append((path, 0, "includes retired shared scheduler header"))

    owned_include = "runtime/models/qwen_image/qwen_image_scheduler.h"
    allowed_owned_include_roots = (
        RUNTIME_MODELS / "qwen_image",
        REPO_ROOT / "tests" / "cpp" / "models" / "qwen_image",
    )
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if owned_include not in text:
                continue
            if not any(path.is_relative_to(allowed) for allowed in allowed_owned_include_roots):
                violations.append((path, 0, "non-Qwen Image file includes Qwen Image scheduler"))

    assert not violations, _format_violations(violations)


def test_runtime_strategy_default_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep runtime_strategy dispatch explicit and model-owned.
    Preconditions: generated plugin index exposes an optional fallback hook.
    Postconditions: shared runtime files do not hardcode text-generation defaults.
    """
    violations = []
    for path in RUNTIME_STRATEGY_DEFAULT_FILES:
        text = path.read_text(encoding="utf-8")
        if "decoder_kv_cache" in text:
            violations.append(
                (path, 0, "shared runtime file hardcodes text-generation strategy default")
            )

    manifests_with_defaults = [
        path
        for path in RUNTIME_MODELS.glob("*/MODEL.toml")
        if "default_runtime_strategy" in path.read_text(encoding="utf-8")
    ]
    if manifests_with_defaults:
        violations.append(
            (manifests_with_defaults[0], 0, "runtime strategy fallback must not be model-specific")
        )

    loader_header = REPO_ROOT / "include" / "trtmc" / "runtime" / "pipeline_plugin_loader.h"
    if "default_runtime_strategy()" not in loader_header.read_text(encoding="utf-8"):
        violations.append((loader_header, 0, "missing generated default strategy API"))

    index_template = REPO_ROOT / "cmake" / "model_plugin_index.cpp.in"
    if "@TRTMC_DEFAULT_RUNTIME_STRATEGY_RETURN@" not in index_template.read_text(encoding="utf-8"):
        violations.append((index_template, 0, "missing generated default strategy template hook"))

    assert not violations, _format_violations(violations)


def test_shared_diff_vl_tool_has_no_family_reference_implementations() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep VL diff reference behavior in model-owned family modules.
    Preconditions: tools/diff_vl.py provides only generic test orchestration.
    Postconditions: named family reference loaders live under family folders.
    """
    text = SHARED_DIFF_VL_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "Qwen",
        "qwen",
        "LocateAnything",
        "locateanything",
        "qwen_merge_group",
        "MoonViT",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "_get_hf_vision_features_qwen",
        "_get_hf_vision_features_locateanything",
    )
    violations = [
        (SHARED_DIFF_VL_TOOL, 0, f"shared VL diff tool contains family term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned VL diff handler")
        for path in MODEL_OWNED_DIFF_VL_HANDLERS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_diff_vl_tool_uses_family_owned_debug_runners() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep VL engine/debug execution in the owning family.
    Preconditions: VL runtime families are declared in plugin metadata or E2E
    manifests.
    Postconditions: tools/diff_vl.py loads a family-owned vl_debug_runner.py
    and does not import shared debug runner classes.
    """
    text = SHARED_DIFF_VL_TOOL.read_text(encoding="utf-8")
    forbidden_calls = (
        "runner_from_bundle",
        "VisionTrtRunner",
        "VLTrtRunner",
        "load_vision_engine_from_bundle",
        "load_section_from_bundle",
    )
    violations = [
        (SHARED_DIFF_VL_TOOL, 0, "shared VL diff tool imports the shared debug runner surface")
        for needle in ("from tensorrt_model_connect.debug_runner import",)
        if needle in text
    ]
    tree = ast.parse(text, filename=str(SHARED_DIFF_VL_TOOL))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in forbidden_calls
        ):
            violations.append(
                (
                    SHARED_DIFF_VL_TOOL,
                    node.lineno,
                    f"shared VL diff tool directly calls {node.func.id}",
                )
            )
    if "_load_family_vl_debug_runner" not in text:
        violations.append((SHARED_DIFF_VL_TOOL, 0, "missing family-owned VL debug runner loader"))

    vl_families: set[str] = set()
    for plugin_path in sorted(FAMILIES.glob("*/plugin.py")):
        plugin_text = plugin_path.read_text(encoding="utf-8")
        if re.search(
            r"runtime_strategy\s*=\s*['\"][A-Za-z0-9_]+_vision_language['\"]", plugin_text
        ):
            vl_families.add(plugin_path.parent.name)
    for manifest_path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(manifest.get("runtime_strategy") or "").endswith("_vision_language"):
            vl_families.add(manifest_path.parents[1].name)

    violations.extend(
        (
            FAMILIES / family / "vl_debug_runner.py",
            0,
            "vision-language family missing model-owned VL debug runner",
        )
        for family in sorted(vl_families)
        if not (FAMILIES / family / "vl_debug_runner.py").is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_diff_logits_tool_has_no_family_reference_implementations() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep logit-diff family behavior in model-owned family modules.
    Preconditions: tools/diff_logits.py provides generic logit orchestration.
    Postconditions: named family runners/HF paths live under family folders.
    """
    text = SHARED_DIFF_LOGITS_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "Whisper",
        "whisper",
        "WhisperForConditionalGeneration",
        "WhisperTrtRunner",
        "_run_hf_whisper",
        "openai/whisper",
        "MambaTrtRunner",
        "RwkvTrtRunner",
    )
    violations = [
        (SHARED_DIFF_LOGITS_TOOL, 0, f"shared logit diff tool contains family term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned logit diff handler")
        for path in MODEL_OWNED_DIFF_LOGITS_HANDLERS
        + (
            FAMILIES / "mamba" / "diff_logits.py",
            FAMILIES / "rwkv" / "diff_logits.py",
        )
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_diff_tools_do_not_branch_on_model_type_for_hf_loading() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific HF reference loading in model-owned handlers.
    Preconditions: diff tools dispatch optional family diff handlers.
    Postconditions: shared diff helpers do not choose HF classes by model name.
    """
    forbidden = (
        "AutoModelForImageTextToText",
        '"vl" in model_type',
        "'vl' in model_type",
        '"vision" in model_type',
        "'vision' in model_type",
        "is_vl_model",
    )
    violations = []
    for path in (SHARED_DIFF_LOGITS_TOOL, SHARED_TOOL_HELPERS):
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"shared diff helper owns model-specific HF loading: {needle}")
            for needle in forbidden
            if needle in text
        )

    diff_logits_text = SHARED_DIFF_LOGITS_TOOL.read_text(encoding="utf-8", errors="ignore")
    for snippet in (
        "_find_family_diff_logits_handler(model_type)",
        'load_hf_model = getattr(handler, "load_hf_model", None)',
    ):
        if snippet not in diff_logits_text:
            violations.append(
                (
                    SHARED_DIFF_LOGITS_TOOL,
                    0,
                    f"missing family-owned diff-logits loading dispatch: {snippet}",
                )
            )

    assert not violations, _format_violations(violations)


def test_shared_runner_parity_tool_uses_family_registry_dispatch() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep runner parity generic and route model-owned runners through metadata.
    Preconditions: family metadata declares owned debug runners.
    Postconditions: shared runner parity tool names no model-owned debug runners
    and does not call the shared debug_runner factory surface.
    """
    text = SHARED_RUNNER_PARITY_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "MambaTrtRunner",
        "RwkvTrtRunner",
        "HybridTrtRunner",
        "from tensorrt_model_connect.debug_runner import",
        "runner_from_bundle",
        'runtime_strategy == "mamba_ssm_recurrent"',
        'runtime_strategy == "rwkv_recurrent"',
        'runtime_strategy == "nemotron_h_hybrid_mamba_attention"',
    )
    violations = [
        (
            SHARED_RUNNER_PARITY_TOOL,
            0,
            f"shared runner parity tool contains model-owned dispatch {needle}",
        )
        for needle in forbidden
        if needle in text
    ]
    if "resolve_debug_runner" not in text:
        violations.append(
            (
                SHARED_RUNNER_PARITY_TOOL,
                0,
                "shared runner parity tool should dispatch through family metadata",
            )
        )

    assert not violations, _format_violations(violations)


def test_shared_diff_layers_tool_uses_family_capability_dispatch() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep per-layer debug engine construction model-owned.
    Preconditions: family plugins declare debug_layer_outputs when supported.
    Postconditions: shared diff layer tool dispatches through plugin metadata
    and does not import a concrete family builder.
    """
    text = SHARED_DIFF_LAYERS_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "families.llama",
        "families.qwen",
        "standard_decoder_builder",
        "build_standard_decoder_engine",
        "Qwen",
        "qwen",
        "LLaMA",
        "llama",
    )
    violations = [
        (
            SHARED_DIFF_LAYERS_TOOL,
            0,
            f"shared layer diff tool contains family-owned builder term {needle}",
        )
        for needle in forbidden
        if needle in text
    ]
    for needle in ("family_has_capability", "debug_layer_outputs", "plugin.build_engine"):
        if needle not in text:
            violations.append(
                (
                    SHARED_DIFF_LAYERS_TOOL,
                    0,
                    f"shared layer diff tool missing dispatch marker {needle}",
                )
            )

    assert not violations, _format_violations(violations)


def test_runtime_cli_python_requirement_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific C++ CLI helper-python needs out of RunContext.
    Preconditions: model manifests may opt into passing runtime Python to the CLI.
    Postconditions: shared contracts read metadata, and PersonaPlex owns its opt-in.
    """
    contracts_text = E2E_CONTRACTS.read_text(encoding="utf-8")
    violations = []
    if 'runtime_strategy or "") not in {"speech_to_speech"}' in contracts_text:
        violations.append(
            (
                E2E_CONTRACTS,
                0,
                "RunContext hardcodes speech_to_speech for --hf-python",
            )
        )
    if "runtime_cli_requires_hf_python" not in contracts_text:
        violations.append(
            (
                E2E_CONTRACTS,
                0,
                "RunContext should read runtime_cli_requires_hf_python metadata",
            )
        )

    for manifest in PERSONAPLEX_E2E_MANIFESTS:
        text = manifest.read_text(encoding="utf-8")
        if '"runtime_cli_requires_hf_python": true' not in text:
            violations.append((manifest, 0, "missing model-owned runtime CLI Python opt-in"))

    assert not violations, _format_violations(violations)


def test_shared_perf_tools_use_family_perf_hooks() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep runtime-specific perf/profiling behavior in family hooks.
    Preconditions: families may expose perf_hooks.py for non-generic runners.
    Postconditions: shared perf tools do not name model-owned recurrent runners.
    """
    forbidden = (
        "MambaTrtRunner",
        "RwkvTrtRunner",
        "HybridTrtRunner",
        "bench_trt_mamba",
        "is_mamba",
        "TRT-Mamba",
        "Mamba model",
        'runtime_strategy == "mamba_ssm_recurrent"',
        'runtime_strategy == "rwkv_recurrent"',
        'runtime_strategy == "nemotron_h_hybrid_mamba_attention"',
    )
    violations = []
    for path in (
        SHARED_PERF_COMPARE_TOOL,
        SHARED_CPU_PROFILE_TOOL,
        SHARED_CPU_PROFILE_MATRIX_TOOL,
        SHARED_TRTMC_PROFILE_TOOL,
    ):
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared perf/profile tool contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )

    expected_hooks = (FAMILIES / "mamba" / "perf_hooks.py",)
    violations.extend(
        (path, 0, "missing model-owned performance hook")
        for path in expected_hooks
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_text_debug_tools_use_family_debug_runner_dispatch() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep standard decoder debug execution in family-owned adapters.
    Preconditions: shared tools may load engines and know runtime_strategy.
    Postconditions: shared tools do not instantiate the shared text TrtRunner.
    """
    paths = (
        SHARED_DIFF_LOGITS_TOOL,
        SHARED_DIFF_LAYERS_TOOL,
        SHARED_PERF_COMPARE_TOOL,
        SHARED_CPU_PROFILE_TOOL,
        SHARED_CPU_PROFILE_MATRIX_TOOL,
        SHARED_TRTMC_PROFILE_TOOL,
        REPO_ROOT / "tools" / "diff_framework" / "checks" / "layer_profile.py",
    )
    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "from tensorrt_model_connect.debug_runner import TrtRunner" in text:
            violations.append((path, 0, "shared text debug tool imports shared TrtRunner"))

    assert not violations, _format_violations(violations)


def test_family_tools_do_not_import_shared_runner_classes() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep runner behavior in the importing family rather than debug_runner.py.
    Preconditions: family-owned files may import low-level debug_runner helpers.
    Postconditions: family-owned tools do not import shared runner classes.
    """
    forbidden = {
        "TrtRunner",
        "VisionTrtRunner",
        "VLTrtRunner",
        "MagpieTrtRunner",
    }
    violations = []
    for path in sorted(FAMILIES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "tensorrt_model_connect.debug_runner":
                continue
            for alias in node.names:
                if alias.name in forbidden:
                    violations.append((path, node.lineno, f"imports shared {alias.name}"))

    assert not violations, _format_violations(violations)


def test_model_e2e_vl_runners_use_model_local_debug_runner() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep VL debug execution in each model-owned E2E plugin folder.
    Preconditions: each model E2E plugin may provide a vision_language runner.
    Postconditions: VL subprocess scripts import their model-local runner copy.
    """
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/vision_language.py")):
        family = path.parts[-4]
        text = path.read_text(encoding="utf-8")
        if family not in VL_RUNTIME_FAMILIES:
            if "plugin = None" not in text:
                violations.append((path, 0, "non-VL runner is not inert"))
            if "VLTrtRunner" in text or "class VisionLanguageRunner" in text:
                violations.append((path, 0, "non-VL runner contains VL behavior"))
            continue

        helper = path.parent / "vl_debug_runner.py"
        expected = (
            f"from tests.e2e.models.{family}.e2e_plugins.runners.vl_debug_runner import VLTrtRunner"
        )
        if "from tensorrt_model_connect.debug_runner import VLTrtRunner" in text:
            violations.append((path, 0, "imports shared VLTrtRunner"))
        if expected not in text:
            violations.append((path, 0, "does not import model-local VLTrtRunner"))
        if not helper.is_file():
            violations.append((helper, 0, "missing model-local VL debug runner"))

    assert not violations, _format_violations(violations)


def test_shared_perf_profile_tools_and_tests_use_neutral_fixtures() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared perf/profile docs and unit tests model-neutral.
    Preconditions: model-specific perf defaults live in family-owned sidecars.
    Postconditions: shared perf/profile surfaces do not use concrete Qwen fixtures.
    """
    forbidden = (
        "Qwen/",
        "qwen3-0.6b",
        "qwen3.trtfb",
        "qwen_profile",
        "MAMBA_PHASES",
    )
    paths = (
        SHARED_PERF_COMPARE_TOOL,
        SHARED_CPU_PROFILE_TOOL,
        SHARED_TRTMC_PROFILE_TOOL,
        *SHARED_PERF_PROFILE_TEST_FILES,
    )
    violations = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared perf/profile surface contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_report_impact_and_profile_surfaces_use_neutral_fixtures() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared report, impact, and profiling fixtures model-neutral.
    Preconditions: model-specific report/perf tests live under tests/e2e/models.
    Postconditions: shared tests/tools do not carry concrete Qwen fixtures.
    """
    forbidden = (
        "Qwen/",
        "Qwen2",
        "Qwen3",
        "qwen3",
        "qwen_",
        "qwen-",
        "qwen_profile",
        "qwen3.trtfb",
    )
    violations = []
    for path in (*SHARED_REPORT_AND_IMPACT_TEST_FILES, *SHARED_REPORT_AND_PROFILE_TOOLS):
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared report/impact/profile surface contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_qwen_perf_parity_test_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Qwen bundle/tokenizer parity validation in Qwen-owned tests.
    Preconditions: shared tools tests should be model-neutral.
    Postconditions: the Qwen parity test lives under tests/e2e/models/qwen only.
    """
    shared_path = REPO_ROOT / "tests" / "tools" / "test_perf_parity.py"
    owned_path = E2E_MODELS / "qwen" / "test_perf_parity.py"
    violations = []
    if shared_path.exists():
        violations.append(
            (shared_path, 0, "Qwen parity test should not live in shared tools tests")
        )
    if not owned_path.is_file():
        violations.append((owned_path, 0, "missing Qwen-owned parity test"))
    else:
        text = owned_path.read_text(encoding="utf-8")
        for needle in ("Qwen/Qwen3-0.6B", "qwen3-0.6b.trtfb"):
            if needle not in text:
                violations.append((owned_path, 0, f"missing Qwen parity fixture {needle}"))

    assert not violations, _format_violations(violations)


def test_shared_perf_analysis_tools_use_runtime_metadata_and_model_owned_batches() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep perf-analysis tools from owning model/runtime-specific route tables.
    Preconditions: runtime strategy metadata and model perf_validation sidecars exist.
    Postconditions: shared tools derive modes from metadata and load model-owned batches.
    """
    tool_paths = (
        SHARED_SOL_ESTIMATE_TOOL,
        SHARED_AUTO_PERF_TUNE_TOOL,
        SHARED_CLASSIFY_BOTTLENECK_TOOL,
    )
    forbidden = (
        "PIPELINE_MODES = {",
        "PIPELINE_MODES: dict",
        "VALIDATION_MODELS = [",
        "Qwen/",
        "black-forest-labs/",
        "Wan-AI/",
        "Tongyi-MAI/",
        "openai/whisper",
        "suno/bark",
        "nvidia/magpie",
        "facebook/sam",
        "state-spaces/mamba",
        "RWKV/",
        "text_to_audio_bark",
        "text_to_audio_magpie",
    )
    violations = []
    for path in tool_paths:
        text = path.read_text(encoding="utf-8")
        if "runtime_strategy_performance_mode" not in text:
            violations.append((path, 0, "perf tool should use runtime strategy metadata"))
        violations.extend(
            (path, 0, f"shared perf tool contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )

    auto_perf_text = SHARED_AUTO_PERF_TUNE_TOOL.read_text(encoding="utf-8")
    for needle in (
        "DEFAULT_BENCHMARK",
        "_expand_command_template",
        "load_default_validation_models",
        "perf_validation.json",
        'benchmark=entry.get("benchmark")',
        '"repo_root"',
    ):
        if needle not in auto_perf_text:
            violations.append((SHARED_AUTO_PERF_TUNE_TOOL, 0, f"missing batch loader {needle}"))

    forbidden_auto_perf_command_terms = (
        "generate-audio",
        "generate-video",
        "transcribe",
        '"segment"',
        '"encode"',
        "generated_wav",
        "generated_image",
        "_generate_test_wav",
        "_generate_test_image",
    )
    violations.extend(
        (SHARED_AUTO_PERF_TUNE_TOOL, 0, f"shared auto perf owns benchmark command/detail {needle}")
        for needle in forbidden_auto_perf_command_terms
        if needle in auto_perf_text
    )

    owned_sidecars = sorted(E2E_MODELS.glob("*/perf_validation.json"))
    if not owned_sidecars:
        violations.append((E2E_MODELS, 0, "missing model-owned perf_validation sidecars"))
    for path in owned_sidecars:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("models", raw) if isinstance(raw, dict) else raw
        for index, entry in enumerate(entries, 1):
            benchmark = entry.get("benchmark") if isinstance(entry, dict) else None
            command = benchmark.get("command") if isinstance(benchmark, dict) else None
            if not isinstance(command, list) or not all(
                isinstance(token, str) for token in command
            ):
                violations.append(
                    (
                        path,
                        0,
                        f"entry {index} missing model-owned benchmark.command template",
                    )
                )

    assert not violations, _format_violations(violations)


def test_shared_diff_audio_tool_has_no_family_reference_implementations() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep audio-diff family behavior in model-owned family modules.
    Preconditions: tools/diff_audio.py provides generic audio diff dispatch.
    Postconditions: named family staged diff logic lives under family folders.
    """
    text = SHARED_DIFF_AUDIO_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "Bark",
        "bark",
        "audio_bark",
        "suno/",
        "BarkModel",
        "run_cpp_bark",
        "stage4_greedy_parity",
    )
    violations = [
        (SHARED_DIFF_AUDIO_TOOL, 0, f"shared audio diff tool contains family term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned audio diff handler")
        for path in MODEL_OWNED_DIFF_AUDIO_HANDLERS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_personaplex_diff_tool_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep PersonaPlex reference diff behavior in the PersonaPlex family.
    Preconditions: the PersonaPlex diff implementation is family-owned.
    Postconditions: no root compatibility tool carries model-specific dispatch.
    """
    violations = []
    owned_tool = FAMILIES / "personaplex" / "diff_personaplex.py"
    if not owned_tool.is_file():
        violations.append((owned_tool, 0, "missing PersonaPlex-owned diff tool"))
    root_tool = REPO_ROOT / "tools" / "diff_personaplex.py"
    if root_tool.exists():
        violations.append(
            (
                root_tool,
                0,
                "root PersonaPlex diff wrapper must not exist",
            )
        )

    assert not violations, _format_violations(violations)


def test_qwen_aime_benchmark_tool_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Qwen-specific benchmark behavior in the Qwen family.
    Preconditions: the Qwen benchmark implementation is family-owned.
    Postconditions: no root compatibility tool carries model-specific dispatch.
    """
    violations = []
    owned_tool = FAMILIES / "qwen" / "benchmark_qwen3_8b_aime25_vs_hf.py"
    if not owned_tool.is_file():
        violations.append((owned_tool, 0, "missing Qwen-owned AIME benchmark"))
    if SHARED_QWEN_AIME_BENCHMARK_TOOL.exists():
        violations.append(
            (
                SHARED_QWEN_AIME_BENCHMARK_TOOL,
                0,
                "root Qwen benchmark wrapper must not exist",
            )
        )

    assert not violations, _format_violations(violations)


def test_timm_vit_trt_path_benchmark_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep timm_vit validation behavior in the timm_vit family.
    Preconditions: the TRT path benchmark implementation is family-owned.
    Postconditions: no root compatibility tool carries model-specific dispatch.
    """
    violations = []
    owned_tool = E2E_MODELS / "timm_vit" / "e2e_plugins" / "benchmark_trt_paths.py"
    if not owned_tool.is_file():
        violations.append((owned_tool, 0, "missing timm_vit-owned TRT path benchmark"))
    if SHARED_TIMM_VIT_TRT_PATH_TOOL.exists():
        violations.append(
            (
                SHARED_TIMM_VIT_TRT_PATH_TOOL,
                0,
                "root timm_vit benchmark wrapper must not exist",
            )
        )

    assert not violations, _format_violations(violations)


def test_qwen_flashinfer_benchmark_tool_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Qwen-specific FlashInfer benchmark behavior in the Qwen family.
    Preconditions: the Qwen FlashInfer benchmark implementation is family-owned.
    Postconditions: no root compatibility tool carries model-specific dispatch.
    """
    violations = []
    owned_tool = FAMILIES / "qwen" / "bench_flashinfer_e2e.py"
    if not owned_tool.is_file():
        violations.append((owned_tool, 0, "missing Qwen-owned FlashInfer benchmark"))
    if SHARED_QWEN_FLASHINFER_BENCHMARK_TOOL.exists():
        violations.append(
            (
                SHARED_QWEN_FLASHINFER_BENCHMARK_TOOL,
                0,
                "root Qwen FlashInfer benchmark wrapper must not exist",
            )
        )

    assert not violations, _format_violations(violations)


def test_shared_diffusion_validation_tools_have_no_family_implementations() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep diffusion validation internals in model-owned family modules.
    Preconditions: tools/validate_*.py provide only generic validation dispatch.
    Postconditions: named family HF setup, dimensions, and builders live under
    family folders.
    """
    forbidden_by_file = {
        SHARED_VALIDATE_T5_TOOL: (
            "Wan",
            "wan",
            "UMT5EncoderModel",
            "AutoTokenizer",
            "build_t5_encoder_engine",
            "load_t5_weights",
        ),
        SHARED_VALIDATE_DIT_TOOL: (
            "Wan",
            "wan",
            "WanTransformer3DModel",
            "DIM = 1536",
            "NUM_LAYERS = 30",
            "FFN_DIM = 8960",
            "TEXT_SEQ = 16",
            "build_standard_dit_engine",
            "load_dit_weights",
        ),
    }
    violations = []
    for path, forbidden in forbidden_by_file.items():
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared diffusion validation tool contains family term {needle}")
            for needle in forbidden
            if needle in text
        )
    violations.extend(
        (path, 0, "missing model-owned diffusion validation handler")
        for path in MODEL_OWNED_DIFFUSION_VALIDATE_HANDLERS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_diff_t5_tool_has_no_family_reference_implementations() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep encoder diff internals in model-owned family modules.
    Preconditions: tools/diff_t5.py provides only generic diff dispatch.
    Postconditions: named family examples, HF setup, and builders live under
    family folders.
    """
    text = SHARED_DIFF_T5_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "Wan",
        "wan",
        "UMT5EncoderModel",
        "T5EncoderModel",
        "AutoTokenizer",
        "build_t5_encoder_engine",
        "load_t5_weights",
    )
    violations = [
        (SHARED_DIFF_T5_TOOL, 0, f"shared encoder diff tool contains family term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned encoder diff handler")
        for path in MODEL_OWNED_DIFF_T5_HANDLERS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_debug_diffusion_pipeline_has_no_family_implementation() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep component-by-component diffusion debug behavior model-owned.
    Preconditions: tools/debug_diffusion_pipeline.py provides only generic
    debug-handler dispatch.
    Postconditions: named family HF pipeline logic and dimension assumptions
    live under family folders.
    """
    text = SHARED_DEBUG_DIFFUSION_PIPELINE_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "Wan",
        "wan",
        "WanPipeline",
        "WanTransformer3DModel",
        "UMT5",
        "dit_dim",
        "flow_shift",
        "text_encoder_dim",
    )
    violations = [
        (
            SHARED_DEBUG_DIFFUSION_PIPELINE_TOOL,
            0,
            f"shared diffusion debug tool contains family term {needle}",
        )
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned diffusion debug handler")
        for path in MODEL_OWNED_DEBUG_DIFFUSION_PIPELINE_HANDLERS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_root_decoder_builder_package_is_removed() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep decoder builder behavior out of root shared Python modules.
    Preconditions: model-specific decoder builders live in family packages.
    Postconditions: retired root builder package files are absent.
    """
    violations = [
        (path, 0, "root shared builder package file must not exist")
        for path in REMOVED_SHARED_BUILDER_FILES
        if path.exists()
    ]

    assert not violations, _format_violations(violations)


def test_root_graph_helpers_are_removed() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep graph-building behavior in family-owned helper modules.
    Preconditions: family graph helper copies exist.
    Postconditions: root graph_ops/graph_blocks modules are absent.
    """
    violations = [
        (path, 0, "root shared graph helper must not exist")
        for path in REMOVED_ROOT_GRAPH_HELPERS
        if path.exists()
    ]

    assert not violations, _format_violations(violations)


def test_root_checkpoint_mapper_is_type_only() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep checkpoint loading and tensor mapping in family-owned modules.
    Preconditions: family checkpoint_mapper.py copies own concrete loaders.
    Postconditions: root checkpoint_mapper.py exposes only the stable WeightDict type.
    """
    tree = ast.parse(CHECKPOINT_MAPPER.read_text(encoding="utf-8"), filename=str(CHECKPOINT_MAPPER))
    violations = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            violations.append(
                (
                    CHECKPOINT_MAPPER,
                    node.lineno,
                    f"root checkpoint mapper defines helper function {node.name}",
                )
            )
        if isinstance(node, ast.ClassDef) and node.name != "WeightDict":
            violations.append(
                (
                    CHECKPOINT_MAPPER,
                    node.lineno,
                    f"root checkpoint mapper defines non-protocol class {node.name}",
                )
            )

    text = CHECKPOINT_MAPPER.read_text(encoding="utf-8")
    forbidden = (
        "load_standard_weights",
        "_open_safetensors",
        "_load_tensor",
        "_has_tensor",
        "_transpose_2d",
        "safetensors",
        "numpy",
    )
    violations.extend(
        (CHECKPOINT_MAPPER, 0, f"root checkpoint mapper contains {needle}")
        for needle in forbidden
        if needle in text
    )

    assert not violations, _format_violations(violations)


def test_quantization_uses_family_graph_ops_plumbing() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep quantization shared only as format plumbing, not graph ownership.
    Preconditions: QuantContext is the graph-builder quantization boundary.
    Postconditions: quantization takes family graph helpers by injection.
    """
    forbidden_imports = (
        "from .. import graph_ops",
        "from tensorrt_model_connect import graph_ops",
        "from tensorrt_model_connect.graph_ops import",
        "import tensorrt_model_connect.graph_ops",
    )
    violations = []
    for path in sorted(QUANTIZATION.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden_imports:
            if needle in text:
                violations.append(
                    (
                        path,
                        0,
                        f"quantization imports root graph helper via {needle}",
                    )
                )

    context_text = (QUANTIZATION / "context.py").read_text(encoding="utf-8", errors="ignore")
    formats_text = (QUANTIZATION / "formats.py").read_text(encoding="utf-8", errors="ignore")
    init_text = (QUANTIZATION / "__init__.py").read_text(encoding="utf-8", errors="ignore")
    engine_text = ENGINE_BUILDER.read_text(encoding="utf-8", errors="ignore")

    required_snippets = (
        (QUANTIZATION / "context.py", context_text, "graph_ops: Any | None = None"),
        (QUANTIZATION / "context.py", context_text, "self.profile.format.wrap_matmul("),
        (QUANTIZATION / "context.py", context_text, "graph_ops=graph_ops"),
        (QUANTIZATION / "formats.py", formats_text, "graph_ops: Any"),
        (QUANTIZATION / "__init__.py", init_text, "graph_ops: Any | None = None"),
        (ENGINE_BUILDER, engine_text, "def _plugin_graph_ops_module(plugin)"),
        (ENGINE_BUILDER, engine_text, "graph_ops=_plugin_graph_ops_module(plugin)"),
    )
    for path, text, snippet in required_snippets:
        if snippet not in text:
            violations.append((path, 0, f"missing quant graph plumbing: {snippet}"))

    assert not violations, _format_violations(violations)


def test_time_series_trt_helpers_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep time-series TRT builder utility behavior in each family.
    Preconditions: time-series families provide local time_series_trt.py files.
    Postconditions: no model plugin imports the retired shared helper.
    """
    owners = ("chronos_bolt", "patchtst", "patchtsmixer", "timesfm")
    violations = []
    retired_helper = FAMILIES / "_time_series_trt.py"
    retired_text = retired_helper.read_text(encoding="utf-8", errors="ignore")
    if "RetiredSharedFamilyHelperError" not in retired_text:
        violations.append((retired_helper, 0, "shared time-series helper is not retired"))
    for family in owners:
        helper = FAMILIES / family / "time_series_trt.py"
        plugin = FAMILIES / family / "plugin.py"
        if not helper.is_file():
            violations.append((helper, 0, "missing family-owned time-series helper"))
            continue
        helper_text = helper.read_text(encoding="utf-8", errors="ignore")
        if "from . import graph_ops" not in helper_text:
            violations.append((helper, 0, "time-series helper must import local graph_ops"))
        if "from .checkpoint_mapper import" not in helper_text:
            violations.append((helper, 0, "time-series helper must import local checkpoint mapper"))
        plugin_text = plugin.read_text(encoding="utf-8", errors="ignore")
        if "from .._time_series_trt import" in plugin_text:
            violations.append((plugin, 0, "imports retired shared time-series helper"))
        if "from .time_series_trt import" not in plugin_text:
            violations.append((plugin, 0, "missing family-owned time-series helper import"))

    assert not violations, _format_violations(violations)


def test_shared_generic_helpers_describe_capabilities_not_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared helper docs/API names family-neutral.
    Preconditions: family-specific math/backend variants live in family modules.
    Postconditions: generic helpers and their shared tests do not name families.
    """
    forbidden = (
        "Qwen",
        "qwen",
        "LLaMA",
        "Llama",
        "llama",
        "Nemotron",
        "InternVL",
        "Phi",
        "Wan",
        "FLUX",
        "Flux",
        "Bark",
        "Whisper",
        "Magpie",
        "CodeGen",
        "GPT-J",
        "GPT-NeoX",
    )
    violations = []
    for path in SHARED_GENERIC_HELPER_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared generic helper contains family term {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_test_impact_has_no_model_specific_runtime_routes() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep CI impact dispatch generic and model-owned exceptions local.
    Preconditions: model-owned impact rules live under tests/e2e/models/<model>.
    Postconditions: tools/test_impact.py names no model-specific C++ route tables
    or SAM3 diff-refinement rules.
    """
    text = TEST_IMPACT.read_text(encoding="utf-8")
    forbidden = (
        "CPP_PLUGIN_STRATEGIES",
        "CPP_PIPELINE_STRATEGIES",
        "RUNNER_TASK_STRATEGIES",
        "COMPARATOR_TASK_STRATEGIES",
        "THRESHOLD_PROFILE_TASK_STRATEGIES",
        "REFERENCE_TASK_STRATEGIES",
        "PLUGIN_TASK_STRATEGIES",
        "qwen_image_plugin",
        "magpie_plugin",
        "whisper_plugin",
        "bark_plugin",
        "_sam3_models",
        "sam3_public_prompted_segmentation_api",
        "sam3_engine_builder_metadata",
        "sam3_segment_prompted_cli_usage",
        "sam3_segment_prompted_cli_runtime",
        "sam3_segment_sam_cli_usage",
        "sam3_segment_sam_cli_runtime",
        "sam3_perception_config",
        "sam3_bpe_end_of_word_suffix",
        "sam3_harness_contract",
    )
    violations = [
        (TEST_IMPACT, 0, f"shared test impact contains model-owned route {needle}")
        for needle in forbidden
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_shared_runtime_domains_do_not_carry_single_owner_model_helpers() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep single-owner runtime helper APIs inside the owning model folder.
    Preconditions: Bark, Magpie, SAM, and Sam3 runtime helpers have model-local homes.
    Postconditions: shared runtime domains do not expose those model-owned helpers.
    """
    forbidden_paths = (
        RUNTIME_DOMAINS / "audio" / "bark_config.h",
        RUNTIME_DOMAINS / "audio" / "bark_generation_plan.h",
        RUNTIME_DOMAINS / "audio" / "magpie_codec_plan.h",
        RUNTIME_DOMAINS / "audio" / "magpie_decode_policy.h",
        RUNTIME_DOMAINS / "audio" / "magpie_decoder_plan.h",
        RUNTIME_DOMAINS / "audio" / "magpie_text_completion_policy.h",
        RUNTIME_DOMAINS / "audio" / "omni_audio_plan.h",
        RUNTIME_DOMAINS / "audio" / "speech_delay_cache.h",
        RUNTIME_DOMAINS / "audio" / "speech_depth_plan.h",
        RUNTIME_DOMAINS / "audio" / "speech_generation_policy.h",
        RUNTIME_DOMAINS / "audio" / "speech_mimi_decode_plan.h",
        RUNTIME_DOMAINS / "audio" / "speech_runtime_plan.h",
        RUNTIME_DOMAINS / "audio" / "speech_temporal_embed_plan.h",
        RUNTIME_DOMAINS / "audio" / "speech_waveform_postprocess.h",
        RUNTIME_DOMAIN_INCLUDES / "audio" / "speech_decode_stop_policy.h",
        RUNTIME_DOMAIN_INCLUDES / "audio" / "subprocess_runner.h",
        RUNTIME_DOMAINS / "audio" / "rnnt_config.h",
        RUNTIME_DOMAINS / "audio" / "whisper_config.h",
        RUNTIME_DOMAINS / "audio" / "whisper_cross_kv_apply.h",
        RUNTIME_DOMAINS / "audio" / "whisper_cross_kv_plan.h",
        RUNTIME_DOMAINS / "audio" / "whisper_decode_policy.h",
        RUNTIME_DOMAINS / "audio" / "whisper_host_plan.h",
        RUNTIME_DOMAINS / "perception" / "sam_image_preprocess_seam.h",
        RUNTIME_DOMAINS / "perception" / "sam_output_selection.h",
        RUNTIME_DOMAINS / "perception" / "sam_postprocess_seam.h",
        RUNTIME_DOMAINS / "perception" / "sam_prompt_seam.h",
        RUNTIME_DOMAINS / "perception" / "segmentation_preprocess_seam.h",
        RUNTIME_DOMAINS / "perception" / "segmentation_postprocess_seam.h",
        RUNTIME_DOMAINS / "perception" / "perception_types.h",
    )
    violations = [
        (path, 0, "single-owner runtime helper should live under src/runtime/models")
        for path in forbidden_paths
        if path.exists()
    ]

    for path in sorted((RUNTIME_DOMAINS / "perception").glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            "SAM",
            "Sam",
            "sam_",
            "SegmentationConfig",
            "SegmentationLogitsShape",
            "preprocess_segmentation_image",
            "compute_segmentation_class_map_from_logits",
        ):
            if needle in text:
                violations.append(
                    (
                        path,
                        0,
                        f"shared perception domain contains single-family term {needle}",
                    )
                )

    expected_segformer_owned = (
        RUNTIME_MODELS / "segformer" / "segformer_preprocess_seam.h",
        RUNTIME_MODELS / "segformer" / "segformer_postprocess_seam.h",
        CPP_TESTS / "models" / "segformer" / "test_segformer_preprocess_seam.cpp",
        CPP_TESTS / "models" / "segformer" / "test_segformer_postprocess_seam.cpp",
    )
    violations.extend(
        (path, 0, "missing SegFormer-owned perception helper/test")
        for path in expected_segformer_owned
        if not path.is_file()
    )

    audio_configs = RUNTIME_DOMAINS / "audio" / "audio_configs.h"
    if audio_configs.exists():
        text = audio_configs.read_text(encoding="utf-8")
        for needle in ("MagpieTTSConfig", "SpeechConfig", "OmniConfig"):
            if needle in text:
                violations.append(
                    (
                        audio_configs,
                        0,
                        f"{needle} should live under src/runtime/models",
                    )
                )

    assert not violations, _format_violations(violations)


def test_audio_runtime_model_helpers_do_not_carry_sibling_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep copied audio helper implementations from leaking across model DSOs.
    Preconditions: Magpie and speech runtime helpers have model-local homes.
    Postconditions: Magpie-only helpers are absent from sibling audio models and vice versa.
    """
    violations = []
    magpie_forbidden_outside_magpie = (
        "MagpieTTSConfig",
        "build_magpie_config",
        "magpie_ipa",
        "make_ipa_tok",
    )
    for model in ("bark", "qwen3_omni", "nemotron_speech_streaming", "personaplex", "whisper"):
        model_dir = RUNTIME_MODELS / model
        for path in sorted(model_dir.rglob("*")):
            if path.suffix not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            violations.extend(
                (path, 0, f"{model} runtime helper contains Magpie behavior {needle}")
                for needle in magpie_forbidden_outside_magpie
                if needle in text
            )

    speech_forbidden_in_magpie = (
        "SpeechConfig",
        "build_speech_config_from_bundle",
        "speech_text_prompt_ids",
        "speech_system_prompt",
    )
    for path in sorted((RUNTIME_MODELS / "magpie").rglob("*")):
        if path.suffix not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"magpie runtime helper contains speech behavior {needle}")
            for needle in speech_forbidden_in_magpie
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_legacy_runtime_strategy_aliases_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep backward-compat strategy rewrites as model metadata.
    Preconditions: legacy aliases, if any, are declared in runtime model manifests.
    Postconditions: pipeline_factory evaluates aliases generically and names no model branches.
    """
    text = PIPELINE_FACTORY.read_text(encoding="utf-8")
    forbidden = (
        "magpie_tts",
        "text_to_audio_bark",
        "text_to_audio_magpie",
        "diffusion_backend_type",
        "diffusion_flux",
        "diffusion_wan",
        "diffusion_zimage",
        "diffusion_pixart",
        "diffusion_qwen_image",
    )
    violations = [
        (PIPELINE_FACTORY, 0, f"pipeline_factory contains model-owned alias detail {needle}")
        for needle in forbidden
        if needle in text
    ]

    expected_alias_owners = {
        "bark": "text_to_audio|default|_|_|text_to_audio_bark",
        "magpie": "text_to_audio|truthy|magpie_tts|_|text_to_audio_magpie",
        "flux": "diffusion|contains|diffusion_backend_type|flux|diffusion_flux",
        "wan": "diffusion|default|_|_|diffusion_wan",
        "z_image": "diffusion|contains|diffusion_backend_type|z_image|diffusion_zimage",
    }
    for model, alias in expected_alias_owners.items():
        manifest = RUNTIME_MODELS / model / "MODEL.toml"
        if alias not in manifest.read_text(encoding="utf-8"):
            violations.append((manifest, 0, f"missing model-owned legacy alias {alias}"))

    assert not violations, _format_violations(violations)


def test_shared_github_ci_package_smoke_is_model_neutral() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep package-smoke model choices in model-owned test metadata.
    Preconditions: package smoke defaults live under tests/e2e/models/<family>.
    Postconditions: shared GitHub CI scripts expose only generic smoke controls.
    """
    forbidden = (
        "TRTMC_WHEEL_" + "QWEN",
        "wheel-" + "qwen" + "-smoke",
        "Qwen " + "smoke test from trtmc pip package",
        "trtmc-wheel-" + "qwen" + "-smoke",
    )
    violations = []
    for path in SHARED_GITHUB_CI_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared GitHub CI package smoke contains {needle}")
            for needle in forbidden
            if needle in text
        )

    smoke_configs = sorted(E2E_MODELS.glob("*/package_smoke.json"))
    defaults = []
    for path in smoke_configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("default") is True:
            defaults.append(path)
    if len(defaults) != 1:
        violations.append(
            (
                E2E_MODELS,
                0,
                "expected exactly one model-owned default package_smoke.json",
            )
        )

    assert not violations, _format_violations(violations)


def test_shared_diffusion_vlm_assessment_default_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep VLM judge model defaults in model-owned test metadata.
    Preconditions: shared CI/tooling may discover one model-owned VLM config.
    Postconditions: shared CI/tooling does not hardcode a concrete VLM model.
    """
    forbidden = ("Qwen/" + "Qwen2.5-VL-3B-Instruct",)
    shared_paths = (*SHARED_GITHUB_CI_FILES, DIFFUSION_VLM_SIMILARITY_TOOL)
    violations = []
    for path in shared_paths:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared diffusion VLM assessment contains {needle}")
            for needle in forbidden
            if needle in text
        )

    configs = sorted(E2E_MODELS.glob("*/diffusion_vlm_assessment.json"))
    defaults = []
    for path in configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("default") is True:
            defaults.append(path)
    if len(defaults) != 1:
        violations.append(
            (
                E2E_MODELS,
                0,
                "expected exactly one model-owned default diffusion_vlm_assessment.json",
            )
        )

    assert not violations, _format_violations(violations)


def test_shared_diffusion_domains_do_not_carry_single_owner_generation_plans() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep single-owner diffusion generation helpers inside owning model folders.
    Preconditions: Flux, Wan, and PixArt generation helpers have model-local homes.
    Postconditions: shared diffusion domains contain only generic diffusion helpers.
    """
    forbidden_paths = (
        RUNTIME_DIFFUSION_DOMAINS / "diffusion_generation_plan.h",
        RUNTIME_DIFFUSION_DOMAINS / "wan_generation_conditioning.h",
        RUNTIME_DIFFUSION_DOMAINS / "diffusion_denoising_step_seam.h",
        RUNTIME_DIFFUSION_DOMAINS / "batch_utils.h",
        RUNTIME_DIFFUSION_DOMAINS / "batch_utils.cpp",
    )
    required_owned_denoising = {
        "flux": (
            RUNTIME_MODELS / "flux" / "flux_denoising_step_seam.h",
            "run_flux_denoising_steps",
        ),
        "wan": (
            RUNTIME_MODELS / "wan" / "wan_denoising_step_seam.h",
            "run_wan_video_denoising_steps",
        ),
        "pixart": (
            RUNTIME_MODELS / "pixart" / "pixart_denoising_step_seam.h",
            "run_pixart_video_denoising_steps",
        ),
    }
    violations = [
        (path, 0, "single-owner diffusion generation helper should live under src/runtime/models")
        for path in forbidden_paths
        if path.exists()
    ]

    forbidden_terms = (
        "FluxGenerationPlan",
        "WanGenerationPlan",
        "PixArtGenerationPlan",
        "run_video_denoising_steps",
        "run_flux_denoising_steps",
        "run_wan_video_denoising_steps",
        "run_pixart_video_denoising_steps",
    )
    for path in sorted(RUNTIME_DIFFUSION_DOMAINS.glob("*.h")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"shared diffusion domain contains model generation plan {needle}")
            for needle in forbidden_terms
            if needle in text
        )

    for family, (path, symbol) in required_owned_denoising.items():
        if not path.is_file():
            violations.append((path, 0, f"missing {family}-owned denoising seam"))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if symbol not in text:
            violations.append((path, 0, f"missing {family}-owned denoising symbol {symbol}"))

    required_owned_batch_helpers = {
        "flux": (
            RUNTIME_MODELS / "flux" / "flux_batch_utils.h",
            CPP_TESTS / "models" / "flux" / "test_flux_batch_utils.cpp",
            "plan_chunks",
        ),
        "z_image": (
            RUNTIME_MODELS / "z_image" / "z_image_batch_utils.h",
            CPP_TESTS / "models" / "z_image" / "test_z_image_batch_utils.cpp",
            "derive_per_sample_seeds",
        ),
        "qwen_image": (
            RUNTIME_MODELS / "qwen_image" / "qwen_image_batch_utils.h",
            CPP_TESTS / "models" / "qwen_image" / "test_qwen_image_batch_utils.cpp",
            "plan_chunks",
        ),
    }
    for family, (helper_path, test_path, symbol) in required_owned_batch_helpers.items():
        if not helper_path.is_file():
            violations.append((helper_path, 0, f"missing {family}-owned batch helper"))
        elif symbol not in helper_path.read_text(encoding="utf-8", errors="ignore"):
            violations.append((helper_path, 0, f"missing {family}-owned batch symbol {symbol}"))
        if not test_path.is_file():
            violations.append((test_path, 0, f"missing {family}-owned batch helper test"))

    if "src/runtime/domains/diffusion/batch_utils.cpp" in CMAKE_ROOT.read_text(
        encoding="utf-8", errors="ignore"
    ):
        violations.append((CMAKE_ROOT, 0, "trtmc_core links retired diffusion batch helper"))
    if "test_diffusion_batch_utils" in CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore"):
        violations.append((CMAKE_ROOT, 0, "root CMake registers retired diffusion batch test"))

    retired_include = "runtime/domains/diffusion/diffusion_denoising_step_seam.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            if retired_include in path.read_text(encoding="utf-8", errors="ignore"):
                violations.append((path, 0, "includes retired shared diffusion denoising seam"))

    if "test_diffusion_denoising_step_seam" in CMAKE_ROOT.read_text(encoding="utf-8"):
        violations.append((CMAKE_ROOT, 0, "shared diffusion denoising seam test target remains"))

    assert not violations, _format_violations(violations)


def test_shared_bundle_writer_has_no_model_specific_section_names() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep bundle serialization generic; model section schemas live with models.
    Preconditions: bundle_writer.py owns only generic .trtfb serialization.
    Postconditions: model-specific section constants are absent from bundle_writer.py.
    """
    text = BUNDLE_WRITER.read_text(encoding="utf-8")
    violations = [
        (BUNDLE_WRITER, 0, "bundle_writer.py defines Qwen Image section names")
        for needle in ("QWEN_IMAGE_", "qwen_image_")
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_engine_builder_uses_declared_family_capabilities() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep builder dispatch generic and move family capability rules to metadata.
    Preconditions: engine_builder.py orchestrates all family builds.
    Postconditions: no source-text inspection or family-specific NeMo routing remains.
    """
    shared_texts = {
        ENGINE_BUILDER: ENGINE_BUILDER.read_text(encoding="utf-8"),
        BUILD_CLI: BUILD_CLI.read_text(encoding="utf-8"),
    }
    forbidden = [
        "inspect.getsource",
        "_find_sentencepiece_model",
        "source.spm",
        "SentencePiece",
        "Unigram",
        "Marian",
        "NLLB",
        "MagpieTTS",
        "magpietts",
        "EncDecRNNT",
        "Transducer",
        "EncDecMultiTaskModel",
        "model_config.yaml",
        "model_weights.ckpt",
        "_nemo_archive_path",
        "resolve_nemo_model_type",
        "audio_magpie",
        "text_to_audio_magpie",
    ]
    violations = []
    for path, text in shared_texts.items():
        violations.extend(
            (path, 0, f"{path.name} contains shared family leak {needle}")
            for needle in forbidden
            if needle in text
        )

    engine_tree = ast.parse(
        shared_texts[ENGINE_BUILDER],
        filename=str(ENGINE_BUILDER),
    )
    for node in ast.walk(engine_tree):
        if not isinstance(node, ast.Compare):
            continue
        names = [node.left, *node.comparators]
        compares_runtime_strategy = any(
            isinstance(expr, ast.Name) and expr.id == "runtime_strategy" for expr in names
        )
        compares_literal_strategy = any(
            isinstance(expr, ast.Constant) and isinstance(expr.value, str) for expr in names
        )
        if compares_runtime_strategy and compares_literal_strategy:
            violations.append(
                (
                    ENGINE_BUILDER,
                    getattr(node, "lineno", 0),
                    "engine_builder branches on a concrete runtime_strategy",
                )
            )

    expected_tokenizer_adapters = (
        FAMILIES / "marian" / "tokenizer_json.py",
        FAMILIES / "m2m_100" / "tokenizer_json.py",
        FAMILIES / "t5" / "tokenizer_json.py",
    )
    violations.extend(
        (path, 0, "missing family-owned tokenizer conversion adapter")
        for path in expected_tokenizer_adapters
        if not path.is_file()
    )

    expected_nemo_adapters = (
        FAMILIES / "magpie_tts" / "nemo_archive.py",
        FAMILIES / "canary" / "nemo_archive.py",
        FAMILIES / "nemotron_speech_streaming" / "nemo_archive.py",
    )
    violations.extend(
        (path, 0, "missing family-owned NeMo archive adapter")
        for path in expected_nemo_adapters
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_discovery_and_cache_warm_use_family_metadata() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared discovery/cache utilities metadata-driven.
    Preconditions: family MODEL.toml files own model aliases and extra HF assets.
    Postconditions: shared scripts import registry helpers instead of naming families.
    """
    discover_text = AUTOPILOT_DISCOVER.read_text(encoding="utf-8")
    warm_text = WARM_HF_CACHE.read_text(encoding="utf-8")
    violations = []

    if "family_probe_model_types" not in discover_text:
        violations.append(
            (
                AUTOPILOT_DISCOVER,
                0,
                "autopilot discovery does not use family_probe_model_types",
            )
        )
    for needle in (
        '"qwen"',
        '"qwen2"',
        '"qwen3"',
        '"magpie_tts"',
        '"canary"',
        '"lance"',
    ):
        if needle in discover_text:
            violations.append(
                (
                    AUTOPILOT_DISCOVER,
                    0,
                    f"autopilot discovery hardcodes model type {needle}",
                )
            )

    for helper in (
        "family_hf_required_files_by_id",
        "family_hf_warm_dependencies",
        "family_hf_warm_files",
    ):
        if helper not in warm_text:
            violations.append((WARM_HF_CACHE, 0, f"warm cache missing {helper}"))
    for needle in (
        "nemo-nano-codec",
        "byt5-small",
        "wavlm-base-plus",
        "Nemotron-Labs-Diffusion-8B",
        "CLIP-ViT-B-32-laion2B-s34B-b79K",
        "open_clip_pytorch_model.bin",
        "adapter_config.json",
        "adapter_model.safetensors",
    ):
        if needle in warm_text:
            violations.append(
                (
                    WARM_HF_CACHE,
                    0,
                    f"warm cache hardcodes family-owned asset {needle}",
                )
            )

    assert not violations, _format_violations(violations)


def test_shared_autopilot_prompt_has_no_single_family_examples() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep generic onboarding automation from naming existing model plugins.
    Preconditions: family-specific onboarding examples live in model-owned tests/docs.
    Postconditions: autopilot prompt and filters are family-neutral.
    """
    text = AUTOPILOT_AUTORUN.read_text(encoding="utf-8")
    forbidden = (
        "sam3_video",
        "whisper_plugin",
    )
    violations = [
        (AUTOPILOT_AUTORUN, 0, f"autopilot autorun contains family example {needle}")
        for needle in forbidden
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_root_model_script_wrappers_are_absent() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific utility behavior out of shared root scripts.
    Preconditions: family-owned Python modules implement each utility.
    Postconditions: root compatibility wrappers are absent.
    """
    violations = []
    for wrapper, owned_path in ROOT_MODEL_SCRIPT_WRAPPERS.items():
        if not owned_path.is_file():
            violations.append((owned_path, 0, "missing family-owned script module"))
        if wrapper.exists():
            violations.append((wrapper, 0, "root model-specific wrapper must not exist"))

    assert not violations, _format_violations(violations)


def test_model_specific_builder_tests_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep family-specific builder tests with their owning model family.
    Preconditions: tests/e2e/models/<family> owns model-specific test files.
    Postconditions: shared tests/builder does not carry these family test files.
    """
    violations = []
    for shared_path, owned_paths in MODEL_OWNED_BUILDER_TESTS.items():
        if shared_path.exists():
            violations.append(
                (
                    shared_path,
                    0,
                    "model-specific builder test should not live in shared tests/builder",
                )
            )
        if isinstance(owned_paths, Path):
            owned_paths = (owned_paths,)
        for owned_path in owned_paths:
            if not owned_path.is_file():
                violations.append((owned_path, 0, "missing family-owned builder test"))

    assert not violations, _format_violations(violations)


def test_model_specific_plugin_weight_tests_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep family-specific load_weights expectations with the owning family.
    Preconditions: tests/e2e/models/<family> owns model-specific weight tests.
    Postconditions: shared tests/builder only carries generic test support/dispatch.
    """
    shared_texts = {
        path: path.read_text(encoding="utf-8") for path in SHARED_PLUGIN_WEIGHT_TEST_FILES
    }
    support_path = REPO_ROOT / "tests" / "builder" / "family_plugin_test_support.py"
    support_text = support_path.read_text(encoding="utf-8")

    violations = []
    for class_name, owned_path in MODEL_OWNED_PLUGIN_WEIGHT_TESTS.items():
        for shared_path, shared_text in shared_texts.items():
            if class_name in shared_text:
                violations.append(
                    (
                        shared_path,
                        0,
                        f"model-specific plugin weight test {class_name} is shared",
                    )
                )
        if class_name in support_text:
            violations.append(
                (
                    support_path,
                    0,
                    f"generic support imports model-specific test {class_name}",
                )
            )
        if not owned_path.is_file():
            violations.append((owned_path, 0, "missing family-owned weight test"))

    for needle in (
        "model.embed_tokens.weight",
        "encoder.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "vision_model.",
        "block_sparse_moe",
        "backbone.layers",
        "transf_decoder.",
    ):
        for shared_path, shared_text in shared_texts.items():
            if needle in shared_text:
                violations.append(
                    (
                        shared_path,
                        0,
                        f"shared plugin tests contain checkpoint-specific key {needle}",
                    )
                )
        if needle in support_text:
            violations.append(
                (
                    support_path,
                    0,
                    f"shared plugin support contains checkpoint-specific key {needle}",
                )
            )

    assert not violations, _format_violations(violations)


def test_model_specific_registry_contracts_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep exact family registry aliases/attributes in family-owned tests.
    Preconditions: tests/e2e/models/<family> owns family-specific registry contracts.
    Postconditions: shared tests/builder/test_families.py stays generic.
    """
    text = SHARED_FAMILY_REGISTRY_TEST.read_text(encoding="utf-8")

    forbidden = (
        "phimoe",
        "phi3",
        "qwen2_vl",
        "internvl_chat",
        "locateanything",
        "qwen3_omni",
        "personaplex",
        "nemotron_h",
        "nemotron_speech_streaming",
        "patchtst_trt",
        "patchtsmixer_trt",
        "timesfm_trt",
        "chronos_bolt",
        "vision_language",
        "omni_multimodal",
        QWEN3_OMNI_RUNTIME_STRATEGY,
        "speech_to_speech",
        PERSONAPLEX_RUNTIME_STRATEGY,
        "speech_to_text_rnnt",
        NEMOTRON_SPEECH_STREAMING_RUNTIME_STRATEGY,
        "nemotron_h_hybrid_mamba_attention",
        "mamba_ssm_recurrent",
    )

    violations = [
        (
            SHARED_FAMILY_REGISTRY_TEST,
            0,
            f"shared family registry test contains model-specific contract {needle}",
        )
        for needle in forbidden
        if needle in text
    ]
    for family, owned_path in MODEL_OWNED_REGISTRY_CONTRACT_TESTS.items():
        if not owned_path.is_file():
            violations.append((owned_path, 0, f"missing {family} registry contract"))

    assert not violations, _format_violations(violations)


def test_model_specific_manifest_contracts_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep exact manifest contracts with the owning model family.
    Preconditions: model-owned manifest contract tests live in tests/e2e/models.
    Postconditions: shared manifest validation tests cover schema/layout only.
    """
    text = SHARED_MANIFEST_VALIDATION_TEST.read_text(encoding="utf-8")
    forbidden = (
        "nemotron-labs-diffusion-8b",
        "nemotron_labs_diffusion_model_card",
        "model_card_generation_parity",
        "flux-2-dev-fp8",
        "Wan-specific",
        "internlm-case",
        "Qwen/Qwen3",
        "qwen3-test",
        "eagle_vlm",
    )
    expected_model_tests = (
        E2E_MODELS / "flux" / "test_flux_manifest_contract.py",
        E2E_MODELS
        / "nemotron_labs_diffusion"
        / "test_nemotron_labs_diffusion_manifest_contract.py",
        E2E_MODELS / "internlm" / "test_internlm_manifest_profiles.py",
    )

    violations = [
        (
            SHARED_MANIFEST_VALIDATION_TEST,
            0,
            f"shared manifest validation test contains model-owned term {needle}",
        )
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing family-owned manifest contract test")
        for path in expected_model_tests
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_python_profile_details_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete Python dependency profiles in family-owned metadata.
    Preconditions: specialized families may declare Python profile requirements.
    Postconditions: shared python_profiles.toml is generic and family assets are local.
    """
    shared_text = PYTHON_PROFILES.read_text(encoding="utf-8")
    forbidden = (
        "[family_defaults",
        "[profiles.internlm",
        "[profiles.chronos",
        "internlm.lock.txt",
        "chronos.lock.txt",
    )
    violations = [
        (PYTHON_PROFILES, 0, f"shared Python profile config contains {needle}")
        for needle in forbidden
        if needle in shared_text
    ]

    shared_requirements_dir = (
        REPO_ROOT / "python" / "tensorrt_model_connect" / "python_profile_requirements"
    )
    for filename in ("internlm.lock.txt", "chronos.lock.txt"):
        path = shared_requirements_dir / filename
        if path.exists():
            violations.append((path, 0, "profile requirements must be family-owned"))

    expected = {
        "internlm": (
            "internlm|families/internlm/python_profile_requirements/internlm.lock.txt|families/internlm/python_profile_verify.py|true",
            "build|internlm",
            "runtime|internlm",
            "reference|internlm",
            FAMILIES / "internlm" / "python_profile_requirements" / "internlm.lock.txt",
            FAMILIES / "internlm" / "python_profile_verify.py",
        ),
        "chronos_bolt": (
            "chronos|families/chronos_bolt/python_profile_requirements/chronos.lock.txt|families/chronos_bolt/python_profile_verify.py|true",
            "build|chronos",
            "reference|chronos",
            FAMILIES / "chronos_bolt" / "python_profile_requirements" / "chronos.lock.txt",
            FAMILIES / "chronos_bolt" / "python_profile_verify.py",
        ),
    }
    for family, entries in expected.items():
        manifest = FAMILIES / family / "MODEL.toml"
        manifest_text = manifest.read_text(encoding="utf-8")
        for entry in entries:
            if isinstance(entry, Path):
                if not entry.is_file():
                    violations.append((entry, 0, "missing family-owned Python profile asset"))
            elif entry not in manifest_text:
                violations.append((manifest, 0, f"missing family-owned profile metadata {entry}"))

    assert not violations, _format_violations(violations)


def test_shared_generic_tests_use_neutral_fixture_names() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep generic subsystem tests from owning concrete family fixtures.
    Preconditions: family-specific behavior lives under tests/e2e/models/<family>.
    Postconditions: shared tests use synthetic fixture names for generic behavior.
    """
    forbidden = (
        "Qwen",
        "qwen",
        "Llama",
        "llama",
        "llama3",
        "internlm",
        "eagle_vlm",
        "gpt2",
        "bloom",
        "falcon",
        "qwen3",
        "qwen2",
        "deepseek",
        "DeepSeek",
        "Nemotron",
        "nemotron",
        "Omni",
        "omni",
        "distilbert",
        "flux-2",
        "diffusion_flux",
    )

    violations = []
    for path in SHARED_GENERIC_FIXTURE_TEST_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared generic test contains concrete family fixture {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_builder_config_uses_family_metadata_for_model_specific_formats() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific config formats and download allowlists in families.
    Preconditions: family manifests may declare config adapters and HF allow patterns.
    Postconditions: shared config/builder code does not name ELF files or variants.
    """
    forbidden = (
        "ELF",
        "elf_variant",
        "elf_flow",
        "checkpoint_*",
        "elf_params",
        "model.npz",
        "_is_elf_model_dir",
        "_ELF_VARIANTS",
        "_elf_yaml_to_config",
    )
    violations = []
    for path in (CONFIG_PY, ENGINE_BUILDER):
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared builder/config contains model-owned format term {needle}")
            for needle in forbidden
            if needle in text
        )

    for family_dir in sorted(FAMILIES.iterdir()):
        if not family_dir.is_dir() or family_dir.name == "elf_flow":
            continue
        for filename in ("config.py", "model_config.py"):
            path = family_dir / filename
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            violations.extend(
                (path, 0, f"{family_dir.name} contains ELF-owned config term {needle}")
                for needle in forbidden
                if needle in text
            )

    elf_manifest = FAMILIES / "elf_flow" / "MODEL.toml"
    elf_text = elf_manifest.read_text(encoding="utf-8")
    for needle in (
        'config_adapter = "model_config.py|config_from_dir"',
        "hf_allow_patterns",
        "checkpoint_*",
        "model.npz",
        "elf_params.npz",
    ):
        if needle not in elf_text:
            violations.append((elf_manifest, 0, f"missing ELF-owned metadata {needle}"))

    assert not violations, _format_violations(violations)


def test_hf_snapshot_allow_patterns_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-family snapshot layouts out of the shared builder.
    Preconditions: family MODEL.toml files can declare hf_allow_patterns.
    Postconditions: shared engine_builder.py keeps only generic HF patterns;
    Diffusers and LoRA-specific snapshot patterns live in owning family manifests.
    """
    diffusion_patterns = (
        "model_index.json",
        "scheduler/**",
        "text_encoder/**",
        "text_encoder_2/**",
        "transformer/**",
        "vae/**",
        "tokenizer/**",
        "tokenizer_2/**",
        "*/diffusion_pytorch_model.safetensors",
        "*/diffusion_pytorch_model-*.safetensors",
        "*/diffusion_pytorch_model.safetensors.index.json",
    )
    lora_patterns = ("linear_spec_lora/**",)
    shared_text = ENGINE_BUILDER.read_text(encoding="utf-8", errors="ignore")
    allow_match = re.search(
        r"_HF_ALLOW_PATTERNS\s*=\s*\[(.*?)\]\n",
        shared_text,
        flags=re.DOTALL,
    )
    shared_allow_patterns = allow_match.group(1) if allow_match else shared_text
    violations = [
        (ENGINE_BUILDER, 0, f"shared builder owns family HF pattern {needle}")
        for needle in diffusion_patterns + lora_patterns
        if needle in shared_allow_patterns
    ]

    for family in ("flux", "ltx_video", "pixart", "qwen_image", "wan_t2v", "z_image"):
        manifest = FAMILIES / family / "MODEL.toml"
        text = manifest.read_text(encoding="utf-8", errors="ignore")
        if "hf_allow_patterns" not in text:
            violations.append((manifest, 0, "missing hf_allow_patterns"))
            continue
        for needle in diffusion_patterns:
            if needle not in text:
                violations.append((manifest, 0, f"missing diffusion HF pattern {needle}"))

    nemotron_manifest = FAMILIES / "nemotron_labs_diffusion" / "MODEL.toml"
    nemotron_text = nemotron_manifest.read_text(encoding="utf-8", errors="ignore")
    for needle in lora_patterns:
        if needle not in nemotron_text:
            violations.append((nemotron_manifest, 0, f"missing LoRA HF pattern {needle}"))

    assert not violations, _format_violations(violations)


def test_shared_fp8_calibration_has_no_model_specific_hooks() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep ModelOpt compatibility hooks in the owning family plugin.
    Preconditions: fp8_calibrate.py exposes generic calibration utilities.
    Postconditions: shared FP8 calibration code names no model-specific modules.
    """
    forbidden = (
        "Flux2Attention",
        "Flux2ParallelSelfAttention",
        "FLUX",
        "Flux",
        "_register_diffusers_flux2_attention_quantizers",
    )
    text = FP8_CALIBRATE.read_text(encoding="utf-8")
    violations = [
        (FP8_CALIBRATE, 0, f"shared FP8 calibration contains {needle}")
        for needle in forbidden
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_shared_multimodal_preprocessor_uses_generic_strategy_names() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep image preprocessing behavior owned by VL runtime families.
    Preconditions: VL model runtimes carry local preprocessor copies.
    Postconditions: the old shared preprocessor implementation is absent and
    each VL runtime owns its preprocessor source.
    """
    retired_shared_files = (
        RUNTIME_DOMAIN_INCLUDES / "multimodal" / "image_transform_helper.h",
        RUNTIME_DOMAINS / "multimodal" / "image_preprocessor.h",
        RUNTIME_DOMAINS / "multimodal" / "image_preprocessor.cpp",
        RUNTIME_DOMAINS / "multimodal" / "vision_engine.h",
        RUNTIME_DOMAINS / "multimodal" / "vision_engine.cpp",
        RUNTIME_DOMAINS / "multimodal" / "vision_execution_plan.h",
        RUNTIME_DOMAINS / "multimodal" / "vl_decode_policy.h",
    )
    violations = []
    for path in retired_shared_files:
        if path.exists():
            violations.append((path, 0, "shared VL runtime helper must be retired"))
    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8")
    for needle in (
        "src/runtime/domains/multimodal/image_preprocessor.cpp",
        "src/runtime/domains/multimodal/vision_engine.cpp",
        "test_image_preprocessor",
        "test_vision_execution_plan",
        "test_vl_decode_policy",
    ):
        if needle in cmake_text:
            violations.append(
                (CMAKE_ROOT, 0, f"CMake still references retired shared VL helper {needle}")
            )
    if (CPP_TESTS / "test_vision_execution_plan.cpp").exists():
        violations.append(
            (
                CPP_TESTS / "test_vision_execution_plan.cpp",
                0,
                "shared vision execution plan test must be retired",
            )
        )
    if (CPP_TESTS / "test_vl_decode_policy.cpp").exists():
        violations.append(
            (
                CPP_TESTS / "test_vl_decode_policy.cpp",
                0,
                "shared VL decode policy test must be retired",
            )
        )

    for family in VL_RUNTIME_FAMILIES:
        runtime_dir = RUNTIME_MODELS / family
        for filename in (
            "image_preprocessor.h",
            "image_preprocessor.cpp",
            "image_transform_helper.h",
        ):
            path = runtime_dir / filename
            if not path.is_file():
                violations.append((path, 0, "missing family-owned VL image preprocessor"))
        image_preprocessor = runtime_dir / "image_preprocessor.cpp"
        if image_preprocessor.is_file():
            text = image_preprocessor.read_text(encoding="utf-8", errors="ignore")
            if "trtmc/runtime/domains/multimodal/image_transform_helper.h" in text:
                violations.append(
                    (
                        image_preprocessor,
                        0,
                        "VL image preprocessor includes shared transform helper",
                    )
                )
            if '"image_transform_helper.h"' not in text:
                violations.append(
                    (
                        image_preprocessor,
                        0,
                        "VL image preprocessor does not include family-owned transform helper",
                    )
                )
        image_helper = runtime_dir / "image_transform_helper.h"
        if image_helper.is_file():
            text = image_helper.read_text(encoding="utf-8", errors="ignore")
            for needle in (
                "ImageNormalizationParams",
                "ImageTransformParams",
                "normalize_hwc_u8_to_chw",
                "transform_chw_layout",
            ):
                if needle not in text:
                    violations.append(
                        (image_helper, 0, f"family-owned transform helper missing {needle}")
                    )
        for path in (
            runtime_dir / "plugin.cpp",
            runtime_dir / "pipeline.h",
            runtime_dir / "pipeline.cpp",
        ):
            if not path.is_file():
                violations.append((path, 0, "missing family-owned VL runtime file"))
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "runtime/domains/multimodal/image_preprocessor.h" in text:
                violations.append((path, 0, "VL runtime includes shared image preprocessor"))
            for needle in (
                "runtime/domains/multimodal/vision_engine.h",
                "runtime/domains/multimodal/vision_execution_plan.h",
                "runtime/domains/multimodal/vl_decode_policy.h",
            ):
                if needle in text:
                    violations.append((path, 0, f"VL runtime includes shared helper {needle}"))

    assert not violations, _format_violations(violations)


def test_vision_language_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent a shared VL runtime plugin from coupling VL model families.
    Preconditions: each active VL family declares a model-owned runtime strategy.
    Postconditions: no shared vision_language runtime model remains and every
    active VL family owns a runtime DSO manifest.
    """
    violations = []
    shared_runtime_dir = RUNTIME_MODELS / "vision_language"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared vision_language runtime must be removed"))

    for family in VL_RUNTIME_FAMILIES:
        strategy = f"{family}_vision_language"
        runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
        family_plugin = FAMILIES / family / "plugin.py"
        if not runtime_manifest.is_file():
            violations.append((runtime_manifest, 0, "missing family-owned VL runtime manifest"))
            continue
        manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            f'id = "{family}"',
            f'runtime_library = "libtrtmc_model_{family}.so"',
            f'runtime_strategies = ["{strategy}"]',
        ):
            if needle not in manifest_text:
                violations.append((runtime_manifest, 0, f"missing {needle}"))
        if family_plugin.is_file():
            plugin_text = family_plugin.read_text(encoding="utf-8", errors="ignore")
            if f'runtime_strategy = "{strategy}"' not in plugin_text:
                violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))

    for manifest_path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("runtime_strategy") == "vision_language":
            violations.append((manifest_path, 0, "E2E manifest uses shared VL runtime strategy"))

    assert not violations, _format_violations(violations)


def test_segmentation_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent a shared segmentation runtime plugin from coupling
    SegFormer, SAM, and SAM3.
    Preconditions: each active segmentation family declares a model-owned
    runtime strategy.
    Postconditions: no shared segmentation runtime model remains and every
    active segmentation family owns a runtime DSO manifest.
    """
    violations = []
    shared_runtime_dir = RUNTIME_MODELS / "segmentation"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared segmentation runtime must be removed"))
    shared_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / "segmentation"
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared segmentation C++ tests must be removed"))

    for family, strategy in SEGMENTATION_RUNTIME_STRATEGIES.items():
        runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
        family_plugin = FAMILIES / family / "plugin.py"
        if not runtime_manifest.is_file():
            violations.append(
                (runtime_manifest, 0, "missing family-owned segmentation runtime manifest")
            )
            continue
        manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            f'id = "{family}"',
            f'runtime_library = "libtrtmc_model_{family}.so"',
            f'runtime_strategies = ["{strategy}"]',
        ):
            if needle not in manifest_text:
                violations.append((runtime_manifest, 0, f"missing {needle}"))
        if family_plugin.is_file():
            plugin_text = family_plugin.read_text(encoding="utf-8", errors="ignore")
            if f'runtime_strategy = "{strategy}"' not in plugin_text:
                violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))

    shared_runtime_strategies = {"segmentation", "prompted_segmentation"}
    for manifest_path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("runtime_strategy") in shared_runtime_strategies:
            violations.append(
                (manifest_path, 0, "E2E manifest uses shared segmentation runtime strategy")
            )

    assert not violations, _format_violations(violations)


def test_encoder_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent a shared encoder runtime plugin from coupling encoder,
    embedding, reranking, object-detection, and neural-operator families.
    Preconditions: active encoder-style families declare model-owned runtime
    strategies.
    Postconditions: no shared encoder runtime model remains, generic encoder
    runtime strategies are retired, and every active encoder family owns a DSO
    manifest plus C++ test registration.
    """
    violations = []
    shared_runtime_dir = RUNTIME_MODELS / "encoder"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared encoder runtime must be removed"))
    shared_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / "encoder"
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared encoder C++ tests must be removed"))

    retired_runtime_strategies = {
        "encoder_only",
        "embedding",
        "reranking",
        "object_detection",
        "neural_operator",
    }
    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    for strategy in sorted(retired_runtime_strategies & guard_strategies):
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"guard lists retired {strategy}"))
    for strategy in sorted(retired_runtime_strategies & matrix_strategies):
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"matrix defines retired {strategy}"))

    for family, strategies in ENCODER_RUNTIME_STRATEGIES.items():
        expected = (strategies,) if isinstance(strategies, str) else strategies
        runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
        family_plugin = FAMILIES / family / "plugin.py"
        cpp_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / family
        if not runtime_manifest.is_file():
            violations.append(
                (runtime_manifest, 0, "missing family-owned encoder runtime manifest")
            )
            continue

        manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            f'id = "{family}"',
            f'runtime_library = "libtrtmc_model_{family}.so"',
            f"trtmc_model_{family}",
        ):
            if needle not in manifest_text:
                violations.append((runtime_manifest, 0, f"missing {needle}"))
        for strategy in expected:
            if strategy not in matrix_strategies:
                violations.append(
                    (RUNTIME_STRATEGY_MATRIX, 0, f"missing matrix strategy {strategy}")
                )
            if strategy not in manifest_text:
                violations.append((runtime_manifest, 0, f"missing runtime strategy {strategy}"))

        if not family_plugin.is_file():
            violations.append((family_plugin, 0, "missing family-owned Python plugin"))
        else:
            plugin_text = family_plugin.read_text(encoding="utf-8", errors="ignore")
            for strategy in expected:
                if strategy not in plugin_text:
                    violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))

        if not cpp_test_dir.is_dir():
            violations.append((cpp_test_dir, 0, "missing family-owned encoder C++ test dir"))

    for manifest_path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("runtime_strategy") in retired_runtime_strategies:
            violations.append(
                (manifest_path, 0, "E2E manifest uses retired shared encoder runtime strategy")
            )

    assert not violations, _format_violations(violations)


def test_recurrent_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent one recurrent runtime DSO from coupling Mamba, RWKV,
    Nemotron-H, and Qwen3.5.
    Preconditions: active recurrent families declare model-owned runtime
    strategies.
    Postconditions: no shared recurrent runtime model remains, retired generic
    recurrent strategies are absent, and every active recurrent family owns a
    runtime DSO manifest plus C++ tests.
    """
    violations = []
    shared_runtime_dir = RUNTIME_MODELS / "recurrent"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared recurrent runtime must be removed"))
    shared_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / "recurrent"
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared recurrent C++ tests must be removed"))

    retired_runtime_strategies = {"ssm_recurrent", "hybrid_mamba_attention"}
    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    for strategy in sorted(retired_runtime_strategies & guard_strategies):
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"guard lists retired {strategy}"))
    for strategy in sorted(retired_runtime_strategies & matrix_strategies):
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"matrix defines retired {strategy}"))

    for family, strategy in RECURRENT_RUNTIME_STRATEGIES.items():
        runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
        family_plugin = FAMILIES / family / "plugin.py"
        cpp_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / family
        if not runtime_manifest.is_file():
            violations.append(
                (runtime_manifest, 0, "missing family-owned recurrent runtime manifest")
            )
            continue

        manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            f'id = "{family}"',
            f'runtime_library = "libtrtmc_model_{family}.so"',
            f'runtime_strategies = ["{strategy}"]',
            f"trtmc_model_{family}",
        ):
            if needle not in manifest_text:
                violations.append((runtime_manifest, 0, f"missing {needle}"))
        if strategy not in matrix_strategies:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"missing matrix strategy {strategy}"))

        if not family_plugin.is_file():
            violations.append((family_plugin, 0, "missing family-owned Python plugin"))
        else:
            plugin_text = family_plugin.read_text(encoding="utf-8", errors="ignore")
            if strategy not in plugin_text:
                violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))

        if not cpp_test_dir.is_dir():
            violations.append((cpp_test_dir, 0, "missing family-owned recurrent C++ test dir"))

    for manifest_path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("runtime_strategy") in retired_runtime_strategies:
            violations.append(
                (manifest_path, 0, "E2E manifest uses retired shared recurrent runtime strategy")
            )

    assert not violations, _format_violations(violations)


def test_timm_vit_runtime_strategy_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep timm_vit runtime dispatch model-owned while preserving the
    generic image-classification task strategy.
    Preconditions: timm_vit declares a runtime DSO and E2E manifests.
    Postconditions: runtime_strategy uses timm_vit_image_classification and
    the old image_classification runtime key is retired.
    """
    violations = []
    owned_strategy = "timm_vit_image_classification"
    retired_strategy = "image_classification"

    runtime_manifest = RUNTIME_MODELS / "timm_vit" / "MODEL.toml"
    family_plugin = FAMILIES / "timm_vit" / "plugin.py"
    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))

    if owned_strategy not in runtime_manifest.read_text(encoding="utf-8"):
        violations.append((runtime_manifest, 0, f"missing {owned_strategy}"))
    if owned_strategy not in family_plugin.read_text(encoding="utf-8"):
        violations.append((family_plugin, 0, f"missing {owned_strategy}"))
    if owned_strategy not in matrix_strategies:
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"missing {owned_strategy}"))
    for strategy_set, label in (
        (guard_strategies, "guard"),
        (matrix_strategies, "matrix"),
    ):
        if retired_strategy in strategy_set:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"{label} lists retired runtime key"))

    for manifest_path in sorted((E2E_MODELS / "timm_vit" / "manifests").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_strategy") != owned_strategy:
            violations.append((manifest_path, 0, "timm_vit manifest uses non-owned runtime key"))

    assert not violations, _format_violations(violations)


def test_speech_to_text_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent one ASR runtime DSO from coupling Whisper and Canary.
    Preconditions: speech-to-text families declare model-owned runtime
    strategies while keeping task_strategy=speech_to_text.
    Postconditions: the retired speech_to_text runtime key is absent and each
    active ASR family owns its runtime DSO, C++ tests, and E2E manifests.
    """
    violations = []
    retired_strategy = "speech_to_text"
    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    for strategy_set, label in (
        (guard_strategies, "guard"),
        (matrix_strategies, "matrix"),
    ):
        if retired_strategy in strategy_set:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"{label} lists retired runtime key"))

    for family, strategy in SPEECH_TO_TEXT_RUNTIME_STRATEGIES.items():
        runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
        family_plugin = FAMILIES / family / "plugin.py"
        cpp_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / family
        if not runtime_manifest.is_file():
            violations.append((runtime_manifest, 0, "missing family-owned ASR runtime manifest"))
            continue

        manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            f'id = "{family}"',
            f'runtime_library = "libtrtmc_model_{family}.so"',
            f'runtime_strategies = ["{strategy}"]',
            f"trtmc_model_{family}",
        ):
            if needle not in manifest_text:
                violations.append((runtime_manifest, 0, f"missing {needle}"))
        if strategy not in matrix_strategies:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"missing matrix strategy {strategy}"))

        if not family_plugin.is_file():
            violations.append((family_plugin, 0, "missing family-owned Python plugin"))
        else:
            plugin_text = family_plugin.read_text(encoding="utf-8", errors="ignore")
            if f'runtime_strategy = "{strategy}"' not in plugin_text:
                violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))

        if not cpp_test_dir.is_dir():
            violations.append((cpp_test_dir, 0, "missing family-owned ASR C++ test dir"))

        for manifest_path in sorted((E2E_MODELS / family / "manifests").glob("*.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("runtime_strategy") != strategy:
                violations.append(
                    (manifest_path, 0, f"{family} manifest uses non-owned runtime key")
                )
            if manifest.get("task_strategy") != retired_strategy:
                violations.append((manifest_path, 0, f"{family} manifest changed ASR task key"))

        runner = E2E_MODELS / family / "e2e_plugins" / "runners" / "audio_speech.py"
        runner_text = runner.read_text(encoding="utf-8", errors="ignore")
        if f'return "{strategy}"' not in runner_text:
            violations.append((runner, 0, f"runner does not register {strategy}"))

    assert not violations, _format_violations(violations)


def test_nemotron_speech_streaming_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent a generic RNNT runtime DSO from owning Nemotron speech
    streaming behavior.
    Preconditions: Nemotron Speech Streaming declares a model-owned runtime
    strategy while preserving RNNT bundle config fields.
    Postconditions: the retired rnnt runtime directory and
    speech_to_text_rnnt runtime key are absent, and the owning family has local
    runtime tests and E2E manifests.
    """
    violations = []
    retired_strategy = "speech_to_text_rnnt"
    family = "nemotron_speech_streaming"
    strategy = NEMOTRON_SPEECH_STREAMING_RUNTIME_STRATEGY
    shared_runtime_dir = RUNTIME_MODELS / "rnnt"
    shared_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / "rnnt"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared rnnt runtime must be removed"))
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared rnnt C++ tests must be removed"))

    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    for strategy_set, label in (
        (guard_strategies, "guard"),
        (matrix_strategies, "matrix"),
    ):
        if retired_strategy in strategy_set:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"{label} lists retired runtime key"))
    if strategy not in matrix_strategies:
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"missing matrix strategy {strategy}"))

    runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
    family_plugin = FAMILIES / family / "plugin.py"
    runner = E2E_MODELS / family / "e2e_plugins" / "runners" / "audio_speech.py"
    manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
    for needle in (
        f'id = "{family}"',
        f'runtime_library = "libtrtmc_model_{family}.so"',
        f'runtime_strategies = ["{strategy}"]',
        f"trtmc_model_{family}",
        "test_nemotron_speech_streaming_audio_helpers|test_nemotron_speech_streaming_audio_helpers.cpp",
        "test_nemotron_speech_streaming_decode_policy|test_nemotron_speech_streaming_decode_policy.cpp",
        "test_nemotron_speech_streaming_streaming_contract|test_nemotron_speech_streaming_streaming_contract.cpp",
    ):
        if needle not in manifest_text:
            violations.append((runtime_manifest, 0, f"missing {needle}"))
    if f'runtime_strategy = "{strategy}"' not in family_plugin.read_text(encoding="utf-8"):
        violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))
    if f'return "{strategy}"' not in runner.read_text(encoding="utf-8", errors="ignore"):
        violations.append((runner, 0, f"runner does not register {strategy}"))

    for manifest_path in sorted((E2E_MODELS / family / "manifests").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_strategy") != strategy:
            violations.append((manifest_path, 0, f"{family} manifest uses non-owned runtime key"))
        if manifest.get("task_strategy") != "speech_to_text":
            violations.append((manifest_path, 0, f"{family} manifest changed ASR task key"))

    assert not violations, _format_violations(violations)


def test_personaplex_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent a generic speech runtime DSO from owning PersonaPlex.
    Preconditions: PersonaPlex declares a model-owned runtime strategy while
    preserving task_strategy=speech_to_speech.
    Postconditions: the retired speech runtime directory and speech_to_speech
    runtime key are absent, and PersonaPlex owns its runtime tests.
    """
    violations = []
    retired_strategy = "speech_to_speech"
    family = "personaplex"
    strategy = PERSONAPLEX_RUNTIME_STRATEGY
    shared_runtime_dir = RUNTIME_MODELS / "speech"
    shared_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / "speech"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared speech runtime must be removed"))
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared speech C++ tests must be removed"))

    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    for strategy_set, label in (
        (guard_strategies, "guard"),
        (matrix_strategies, "matrix"),
    ):
        if retired_strategy in strategy_set:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"{label} lists retired runtime key"))
    if strategy not in matrix_strategies:
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"missing matrix strategy {strategy}"))

    runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
    family_plugin = FAMILIES / family / "plugin.py"
    runner = E2E_MODELS / family / "e2e_plugins" / "runners" / "audio_speech.py"
    manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
    for needle in (
        f'id = "{family}"',
        f'runtime_library = "libtrtmc_model_{family}.so"',
        f'runtime_strategies = ["{strategy}"]',
        f"trtmc_model_{family}",
        "test_personaplex_speech_subprocess_seam|test_personaplex_speech_subprocess_seam.cpp",
        "test_personaplex_speech_decode_stop_policy|test_personaplex_speech_decode_stop_policy.cpp",
        "test_personaplex_speech_generation_helpers|test_personaplex_speech_generation_helpers.cpp",
        "test_personaplex_speech_depth_plan|test_personaplex_speech_depth_plan.cpp",
        "test_personaplex_speech_runtime_plan|test_personaplex_speech_runtime_plan.cpp",
        "test_personaplex_speech_temporal_embed_plan|test_personaplex_speech_temporal_embed_plan.cpp",
        "test_personaplex_speech_mimi_decode_plan|test_personaplex_speech_mimi_decode_plan.cpp",
        "test_personaplex_speech_pipeline|test_personaplex_speech_pipeline.cpp",
    ):
        if needle not in manifest_text:
            violations.append((runtime_manifest, 0, f"missing {needle}"))
    if f'runtime_strategy = "{strategy}"' not in family_plugin.read_text(encoding="utf-8"):
        violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))
    if f'return "{strategy}"' not in runner.read_text(encoding="utf-8", errors="ignore"):
        violations.append((runner, 0, f"runner does not register {strategy}"))

    for manifest_path in sorted((E2E_MODELS / family / "manifests").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_strategy") != strategy:
            violations.append((manifest_path, 0, f"{family} manifest uses non-owned runtime key"))
        if manifest.get("task_strategy") != retired_strategy:
            violations.append((manifest_path, 0, f"{family} manifest changed speech task key"))

    assert not violations, _format_violations(violations)


def test_qwen3_omni_runtime_is_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent a generic omni runtime DSO from owning Qwen3 Omni.
    Preconditions: Qwen3 Omni declares a model-owned runtime strategy while
    preserving task_strategy=omni_multimodal.
    Postconditions: the retired omni runtime directory and omni_multimodal
    runtime key are absent, and Qwen3 Omni owns its runtime tests.
    """
    violations = []
    retired_strategy = "omni_multimodal"
    family = "qwen3_omni"
    strategy = QWEN3_OMNI_RUNTIME_STRATEGY
    shared_runtime_dir = RUNTIME_MODELS / "omni"
    shared_test_dir = REPO_ROOT / "tests" / "cpp" / "models" / "omni"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared omni runtime must be removed"))
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared omni C++ tests must be removed"))

    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    for strategy_set, label in (
        (guard_strategies, "guard"),
        (matrix_strategies, "matrix"),
    ):
        if retired_strategy in strategy_set:
            violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"{label} lists retired runtime key"))
    if strategy not in matrix_strategies:
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"missing matrix strategy {strategy}"))

    runtime_manifest = RUNTIME_MODELS / family / "MODEL.toml"
    family_plugin = FAMILIES / family / "plugin.py"
    runner = E2E_MODELS / family / "e2e_plugins" / "runners" / "omni.py"
    manifest_text = runtime_manifest.read_text(encoding="utf-8", errors="ignore")
    for needle in (
        f'id = "{family}"',
        f'runtime_library = "libtrtmc_model_{family}.so"',
        f'runtime_strategies = ["{strategy}"]',
        f"trtmc_model_{family}",
        "test_qwen3_omni_audio_plan|test_qwen3_omni_audio_plan.cpp",
        "test_qwen3_omni_pipeline|test_qwen3_omni_pipeline.cpp",
    ):
        if needle not in manifest_text:
            violations.append((runtime_manifest, 0, f"missing {needle}"))
    if f'runtime_strategy = "{strategy}"' not in family_plugin.read_text(encoding="utf-8"):
        violations.append((family_plugin, 0, f"missing runtime_strategy {strategy}"))
    if f'return "{strategy}"' not in runner.read_text(encoding="utf-8", errors="ignore"):
        violations.append((runner, 0, f"runner does not register {strategy}"))

    for manifest_path in sorted((E2E_MODELS / family / "manifests").glob("*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("runtime_strategy") != strategy:
            violations.append((manifest_path, 0, f"{family} manifest uses non-owned runtime key"))
        if manifest.get("task_strategy") != retired_strategy:
            violations.append((manifest_path, 0, f"{family} manifest changed omni task key"))

    for source_test in (
        E2E_MODELS / family / "test_hidden_state_flow.py",
        E2E_MODELS / family / "test_qwen3_omni_runner.py",
    ):
        text = source_test.read_text(encoding="utf-8", errors="ignore")
        if "src/runtime/models/omni" in text:
            violations.append((source_test, 0, "test references retired omni runtime path"))

    assert not violations, _format_violations(violations)


def test_retired_generic_audio_and_diffusion_runtime_keys_are_not_matrix_entries() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep legacy audio/diffusion strategy names out of active runtime
    strategy selection.
    Preconditions: model-owned runtime strategies and legacy aliases are
    declared by their owning runtime model manifests.
    Postconditions: tests/runtime_strategy_matrix.yaml does not list
    text_to_audio or diffusion as active runtime strategies.
    """
    retired = {"text_to_audio", "diffusion"}
    matrix = json.loads(RUNTIME_STRATEGY_MATRIX.read_text(encoding="utf-8"))
    guard_strategies = set(matrix.get("new_runtime_guard_strategies", ()))
    matrix_strategies = set(matrix.get("runtime_strategies", {}))
    violations = []
    for strategy in sorted(retired & guard_strategies):
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"guard lists retired {strategy}"))
    for strategy in sorted(retired & matrix_strategies):
        violations.append((RUNTIME_STRATEGY_MATRIX, 0, f"matrix defines retired {strategy}"))

    assert not violations, _format_violations(violations)


def test_shared_recurrent_contracts_are_model_neutral() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep recurrent-family output layouts in the recurrent model owner.
    Preconditions: recurrent families carry their own state contract copies.
    Postconditions: shared recurrent contracts are absent, and model-owned
    output initializer coverage is registered by the recurrent manifest.
    """
    retired_shared = (
        RUNTIME_DOMAINS / "recurrent" / "recurrent_step_contracts.h",
        RUNTIME_DOMAINS / "recurrent" / "recurrent_tensor_bindings.h",
    )
    violations = [
        (path, 0, "shared recurrent contract/helper must be model-owned")
        for path in retired_shared
        if path.exists()
    ]

    shared_runtime_dir = RUNTIME_MODELS / "recurrent"
    if shared_runtime_dir.exists():
        violations.append((shared_runtime_dir, 0, "shared recurrent runtime must be removed"))
    shared_test_dir = CPP_TESTS / "models" / "recurrent"
    if shared_test_dir.exists():
        violations.append((shared_test_dir, 0, "shared recurrent C++ tests must be removed"))

    for family in RECURRENT_RUNTIME_STRATEGIES:
        owned_header = RUNTIME_MODELS / family / "recurrent_output_initializers.h"
        owned_contract = RUNTIME_MODELS / family / f"{family}_recurrent_step_contracts.h"
        owned_test = (
            CPP_TESTS / "models" / family / f"test_{family}_recurrent_output_initializers.cpp"
        )
        manifest = RUNTIME_MODELS / family / "MODEL.toml"
        expected_manifest_entry = (
            f"test_{family}_recurrent_output_initializers|"
            f"test_{family}_recurrent_output_initializers.cpp"
        )
        for path in (owned_header, owned_contract, owned_test):
            if not path.is_file():
                violations.append((path, 0, "missing family-owned recurrent output initializer"))
        if not manifest.is_file():
            violations.append((manifest, 0, "missing family-owned recurrent runtime manifest"))
        elif expected_manifest_entry not in manifest.read_text(encoding="utf-8"):
            violations.append((manifest, 0, "missing family-owned output initializer test"))

    assert not violations, _format_violations(violations)


def test_shared_audio_domain_has_no_model_owned_feature_extractors() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific audio feature-extractor policy in model owners.
    Preconditions: shared audio domain helper implementations are retired.
    Postconditions: generic WAV I/O uses trtmc_io, and Whisper, Canary, and
    RNNT mel feature extraction lives under the owning runtime models.
    """
    forbidden = (
        "NeMo",
        "nemo",
        "RNNT",
        "rnnt",
        "extract_nemo_mel_spectrogram",
    )
    violations = []
    retired_shared_audio = (
        RUNTIME_AUDIO_DOMAIN_DIR / "audio_types.h",
        RUNTIME_AUDIO_DOMAIN_DIR / "audio_types.cpp",
        CPP_TESTS / "test_audio_types.cpp",
    )
    violations.extend(
        (path, 0, "shared audio value/WAV helper must be retired")
        for path in retired_shared_audio
        if path.exists()
    )
    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8")
    if "src/runtime/domains/audio/audio_types.cpp" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "trtmc_core links retired shared audio_types.cpp"))
    if "test_audio_types" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "CMake registers retired shared audio_types test"))

    retired_shared_mel = (
        RUNTIME_AUDIO_DOMAIN_DIR / "mel_spectrogram.h",
        RUNTIME_AUDIO_DOMAIN_DIR / "mel_spectrogram.cpp",
    )
    violations.extend(
        (path, 0, "shared mel spectrogram implementation must be model-owned")
        for path in retired_shared_mel
        if path.exists()
    )
    if "src/runtime/domains/audio/mel_spectrogram.cpp" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "trtmc_core links shared mel_spectrogram.cpp"))
    if (CPP_TESTS / "test_mel_spectrogram.cpp").exists():
        violations.append(
            (
                CPP_TESTS / "test_mel_spectrogram.cpp",
                0,
                "shared mel spectrogram test must be model-owned",
            )
        )

    forbidden_includes = (
        "runtime/domains/audio/audio_types.h",
        "runtime/domains/audio/mel_spectrogram.h",
    )
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for forbidden_include in forbidden_includes:
                if forbidden_include in text:
                    violations.append((path, 0, "includes retired shared audio domain header"))

    for path in sorted(RUNTIME_AUDIO_DOMAIN_DIR.glob("*.[ch]*")):
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared audio domain contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )

    owned_files = (
        RUNTIME_MODELS / "whisper" / "whisper_mel_spectrogram.h",
        RUNTIME_MODELS / "whisper" / "whisper_mel_spectrogram.cpp",
        CPP_TESTS / "models" / "whisper" / "test_whisper_mel_spectrogram.cpp",
        RUNTIME_MODELS / "canary" / "canary_mel_spectrogram.h",
        RUNTIME_MODELS / "canary" / "canary_mel_spectrogram.cpp",
        CPP_TESTS / "models" / "canary" / "test_canary_mel_spectrogram.cpp",
        RUNTIME_MODELS / "nemotron_speech_streaming" / "audio_helpers.h",
        RUNTIME_MODELS / "nemotron_speech_streaming" / "audio_helpers.cpp",
        CPP_TESTS
        / "models"
        / "nemotron_speech_streaming"
        / "test_nemotron_speech_streaming_audio_helpers.cpp",
    )
    violations.extend(
        (path, 0, "missing model-owned audio feature-extraction surface")
        for path in owned_files
        if not path.is_file()
    )
    manifest_expectations = {
        RUNTIME_MODELS
        / "whisper"
        / "MODEL.toml": "test_whisper_mel_spectrogram|test_whisper_mel_spectrogram.cpp",
        RUNTIME_MODELS
        / "canary"
        / "MODEL.toml": "test_canary_mel_spectrogram|test_canary_mel_spectrogram.cpp",
        RUNTIME_MODELS
        / "nemotron_speech_streaming"
        / "MODEL.toml": "test_nemotron_speech_streaming_audio_helpers|"
        "test_nemotron_speech_streaming_audio_helpers.cpp",
    }
    for manifest, expected in manifest_expectations.items():
        if expected not in manifest.read_text(encoding="utf-8"):
            violations.append((manifest, 0, "missing family-owned audio helper test"))

    assert not violations, _format_violations(violations)


def test_shared_debug_runner_has_no_model_owned_runners() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific TRT debug runners in owning family modules.
    Preconditions: debug_runner.py exposes only shared distributed/debug infrastructure.
    Postconditions: shared debug runner does not define model-owned runner classes,
    bundle readers, or bundle-to-runner factories.
    """
    forbidden = (
        "def runner_from_bundle",
        "def load_engine_from_bundle",
        "def load_vision_engine_from_bundle",
        "def load_section_from_bundle",
        "def load_config_from_bundle",
        "def load_preprocessor_config_from_bundle",
        "class TrtRunner",
        "class VisionTrtRunner",
        "class SegmentationTrtRunner",
        "class VLTrtRunner",
        "def preprocess_image_inputs_for_trt",
        "def preprocess_image_for_trt",
        "def _preprocess_",
        "def _resolve_pil_interpolation",
        "class MambaTrtRunner",
        "class RwkvTrtRunner",
        "class HybridTrtRunner",
        "class Seq2SeqTrtRunner",
        "class TriAttentionTrtRunner",
        "MambaTrtRunner(",
        "RwkvTrtRunner(",
        "HybridTrtRunner(",
        "Seq2SeqTrtRunner(",
        "TriAttentionTrtRunner(",
        "mamba_ssm_recurrent",
        "rwkv_recurrent",
        "nemotron_h_hybrid_mamba_attention",
        "marian_translation",
        "seq2seq_encoder_decoder",
        "text_to_text",
        "class WhisperTrtRunner",
        "class QwenImageDebugRunner",
        "_QWEN_IMAGE_T2I_PROMPT_TEMPLATE",
        "_QWEN_IMAGE_T2I_DROP_IDX",
        "qwen_image_tokenizer_",
    )
    text = DEBUG_RUNNER.read_text(encoding="utf-8")
    violations = [
        (DEBUG_RUNNER, 0, f"shared debug runner contains {needle}")
        for needle in forbidden
        if needle in text
    ]
    expected_owned_files = (
        FAMILIES / "mamba" / "debug_runner.py",
        FAMILIES / "rwkv" / "debug_runner.py",
        FAMILIES / "nemotron_h" / "debug_runner.py",
        FAMILIES / "nemotron_labs_diffusion" / "debug_runner.py",
        FAMILIES / "qwen3_5" / "debug_runner.py",
        FAMILIES / "whisper" / "debug_runner.py",
        FAMILIES / "qwen_image" / "debug_runner.py",
        FAMILIES / "bark" / "debug_runner.py",
        FAMILIES / "segformer" / "debug_runner.py",
        FAMILIES / "elf_flow" / "debug_runner.py",
        FAMILIES / "qwen" / "debug_runner.py",
        FAMILIES / "marian" / "debug_runner.py",
        FAMILIES / "bart" / "debug_runner.py",
        FAMILIES / "m2m_100" / "debug_runner.py",
        FAMILIES / "t5" / "debug_runner.py",
    )
    violations.extend(
        (path, 0, "missing model-owned debug runner")
        for path in expected_owned_files
        if not path.is_file()
    )
    for plugin_path in sorted(FAMILIES.glob("*/plugin.py")):
        plugin_text = plugin_path.read_text(encoding="utf-8")
        match = re.search(r"runtime_strategy\s*=\s*['\"]([^'\"]+)['\"]", plugin_text)
        if match is None:
            continue
        strategy = match.group(1)
        if not (strategy.endswith("_decoder_kv_cache") or strategy.endswith("_decoder_moe")):
            continue
        family_dir = plugin_path.parent
        manifest = family_dir / "MODEL.toml"
        debug_runner = family_dir / "debug_runner.py"
        manifest_text = manifest.read_text(encoding="utf-8")
        if 'debug_runner = "debug_runner.py|runner_from_bundle"' not in manifest_text:
            violations.append(
                (manifest, 0, "decoder family missing model-owned debug_runner metadata")
            )
        if strategy not in manifest_text:
            violations.append((manifest, 0, f"decoder family missing debug strategy {strategy}"))
        if not debug_runner.is_file():
            violations.append((debug_runner, 0, "decoder family missing model-owned debug runner"))
    root_test_text = DEBUG_RUNNER_TEST.read_text(encoding="utf-8")
    root_extended_test_text = DEBUG_RUNNER_EXTENDED_TEST.read_text(encoding="utf-8")
    shared_test_forbidden = (
        "marian_translation",
        "seq2seq_encoder_decoder",
        "text_to_text",
        "rwkv_recurrent",
        "mamba_ssm_recurrent",
        "nemotron_h_hybrid_mamba_attention",
        "MambaTrtRunner",
        "RwkvTrtRunner",
        "WhisperTrtRunner",
        "HybridTrtRunner",
        "Seq2SeqTrtRunner",
        "TriAttentionTrtRunner",
    )
    for path, test_text in (
        (DEBUG_RUNNER_TEST, root_test_text),
        (DEBUG_RUNNER_EXTENDED_TEST, root_extended_test_text),
    ):
        violations.extend(
            (path, 0, f"shared debug runner test contains model-owned {needle}")
            for needle in shared_test_forbidden
            if needle in test_text
        )

    expected_owned_tests = (
        E2E_MODELS / "mamba" / "test_mamba_debug_runner.py",
        E2E_MODELS / "rwkv" / "test_rwkv_debug_runner.py",
        E2E_MODELS / "nemotron_h" / "test_nemotron_h_debug_runner.py",
        E2E_MODELS / "nemotron_labs_diffusion" / "test_nemotron_labs_diffusion_debug_runner.py",
        E2E_MODELS / "qwen3_5" / "test_qwen3_5_debug_runner.py",
        E2E_MODELS / "whisper" / "test_whisper_debug_runner.py",
        E2E_MODELS / "qwen" / "test_qwen_debug_runner.py",
        E2E_MODELS / "marian" / "test_marian_debug_runner.py",
        E2E_MODELS / "bart" / "test_bart_debug_runner.py",
        E2E_MODELS / "m2m_100" / "test_m2m_100_debug_runner.py",
        E2E_MODELS / "t5" / "test_t5_debug_runner.py",
    )
    violations.extend(
        (path, 0, "missing model-owned debug runner test")
        for path in expected_owned_tests
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_cli_uses_capabilities_not_model_names() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep CLI dispatch generic and model behavior in pipeline overrides.
    Preconditions: model pipelines expose generic capability methods.
    Postconditions: shared CLI files do not branch on model or family names.
    """
    forbidden = (
        "ptype.find",
        "pipeline_type() ==",
        "Flux",
        "Wan",
        "ZImage",
        "QwenImage",
        "Qwen-Image",
        "Z-Image",
        "ElfFlowPipeline",
        "Whisper",
        "Magpie",
        "Bark",
        "PersonaPlex",
        "Canary",
        "Nemotron",
        "InternVL",
        "LocateAnything",
        "SAM",
        "Sam",
        "segment-sam",
        "cmd_segment_sam",
        "write_sam_overlay",
    )
    violations = []
    for path in SHARED_CLI_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared CLI contains model dispatch term {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_e2e_contracts_do_not_assign_model_contracts() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model/user contract ownership in model manifests.
    Preconditions: contracts.py defines only generic E2E dataclasses/protocols.
    Postconditions: shared harness code has no model-name contract lookup tables.
    """
    text = E2E_CONTRACTS.read_text(encoding="utf-8")
    forbidden = [
        "MODEL_REFERENCE_FAMILY",
        "REFERENCE_FAMILY_TO_USER_CONTRACT",
        "REFERENCE_FAMILY_TO_COMPARISON_MODE",
        "RUNTIME_TO_TASK_STRATEGY",
        "NEMOTRON_LABS_DIFFUSION_MODEL_CARD",
        "ASR_WHISPER",
        "ASR_CANARY",
        "TTS_BARK",
        "TTS_MAGPIE",
        "S2S_PERSONAPLEX",
    ]
    violations = [
        (E2E_CONTRACTS, 0, f"shared E2E contracts define {needle}")
        for needle in forbidden
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_shared_e2e_docs_use_generic_examples() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared harness documentation from advertising one family.
    Preconditions: shared E2E contracts and runners document generic protocols.
    Postconditions: examples/docstrings in shared harness files use generic names.
    """
    forbidden = (
        "qwen",
        "Qwen",
        "llama",
        "LLaMA",
        "InternVL",
        "Nemotron",
        "Magpie",
        "Bark",
        "Whisper",
        "Canary",
        "Flux",
        "Wan",
        "PixArt",
        "Z-Image",
    )
    violations = []
    for path in E2E_SHARED_DOC_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared E2E documentation contains model term {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_e2e_preflight_and_repro_do_not_name_single_model_details() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep dependency checks and repro commands declared by model assets.
    Preconditions: manifests declare preflight_requirements and model plugins may
    declare repro command providers.
    Postconditions: shared manifest/orchestrator code has no single-model checks
    for timm, Chronos, or ELF/image-classification command construction.
    """
    violations = []
    manifest_text = E2E_MANIFEST_LOADER.read_text(encoding="utf-8")
    for needle in ("timm", "chronos", "chronos_bolt_trt"):
        if needle in manifest_text:
            violations.append((E2E_MANIFEST_LOADER, 0, f"shared preflight names {needle}"))

    orchestrator_text = E2E_ORCHESTRATOR.read_text(encoding="utf-8")
    for needle in (
        'elif task_strategy == "diffusion_text_generation"',
        'elif task_strategy == "image_classification"',
        "/tmp/trtmc_elf_samples.jsonl",
        "--condition-latents-raw",
        "--condition-mask-raw",
    ):
        if needle in orchestrator_text:
            violations.append(
                (
                    E2E_ORCHESTRATOR,
                    0,
                    f"shared repro command contains model-owned detail {needle}",
                )
            )

    assert not violations, _format_violations(violations)


def test_shared_e2e_harness_has_no_sam3_behavior_branches() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Sam3 test behavior in tests/e2e/models/sam3/e2e_plugins.
    Preconditions: shared E2E harness files provide only generic orchestration.
    Postconditions: Sam3 runner/reference/contract behavior is absent from shared files.
    """
    forbidden = ("sam3", "Sam3", "prompted_segmentation_sam3", "_run_sam3")
    violations = []
    for path in E2E_SHARED_HARNESS_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared E2E harness contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_segmentation_contracts_have_no_model_owned_prompted_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep prompted-segmentation family contracts model-owned.
    Preconditions: SAM-family contract plugins live under tests/e2e/models.
    Postconditions: shared contracts/plugins do not name SAM-owned reference
    families or reference configuration.
    """
    forbidden = (
        "PROMPTED_SEGMENTATION_SAM",
        "prompted_segmentation_sam",
        "sam_mode",
    )
    violations = []
    for path in E2E_SHARED_SEGMENTATION_CONTRACT_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared segmentation contract contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )
    violations.extend(
        (path, 0, "missing model-owned E2E contract plugin")
        for path in MODEL_OWNED_E2E_CONTRACT_PLUGINS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_references_have_no_model_owned_prompted_segmentation_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep prompted-segmentation HF reference behavior model-owned.
    Preconditions: SAM-family reference plugins live under tests/e2e/models.
    Postconditions: shared reference backends do not implement prompted
    segmentation model calls.
    """
    forbidden = (
        "_run_prompted_segmentation_ref",
        "SamModel",
        "SamProcessor",
        "hf_sam",
        "hf_prompted_segmentation",
    )
    violations = []
    for path in E2E_SHARED_PROMPTED_SEGMENTATION_REFERENCE_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared reference contains prompted segmentation term {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_segmentation_runtime_has_no_model_owned_prompted_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep prompted-segmentation runner/comparator behavior model-owned.
    Preconditions: SAM-family runner/comparator plugins live under tests/e2e/models.
    Postconditions: shared segmentation runtime files do not implement or
    register prompted-segmentation strategy behavior.
    """
    forbidden = (
        "PromptedSegmentationRunner",
        "PromptedSegmentationComparator",
        "prompted_segmentation",
        "segment-sam",
        "_load_mask_outputs",
        "_write_segmented_overlay",
        "point_prompt",
    )
    violations = []
    for path in E2E_SHARED_PROMPTED_SEGMENTATION_RUNTIME_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared segmentation runtime contains prompted term {needle}")
            for needle in forbidden
            if needle in text
        )
    violations.extend(
        (path, 0, "missing model-owned E2E prompted segmentation runtime plugin")
        for path in MODEL_OWNED_E2E_PROMPTED_SEGMENTATION_RUNTIME_PLUGINS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_prompted_segmentation_e2e_behavior_lives_only_in_sam_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep prompted-segmentation E2E runners owned by SAM-family tests.
    Preconditions: generated model-local sidecars may exist for many families.
    Postconditions: only sam and sam3 sidecars carry prompted segmentation;
    segformer carries semantic segmentation only; all other sidecars are inert.
    """
    prompted_owners = {"sam", "sam3"}
    semantic_owners = {"segformer"}
    prompted_terms = (
        "PromptedSegmentationRunner",
        "segment-prompted",
        "segment-sam",
        "SAM mask",
        "_write_segmented_overlay",
    )
    semantic_terms = (
        "class SegmentationRunner",
        "trtmc segment",
    )
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/segmentation.py")):
        family = path.parts[-4]
        text = path.read_text(encoding="utf-8", errors="ignore")
        if family in prompted_owners:
            violations.extend(
                (path, 0, f"prompted segmentation sidecar contains semantic behavior {needle}")
                for needle in semantic_terms
                if needle in text
            )
            continue
        if family in semantic_owners:
            violations.extend(
                (path, 0, f"semantic segmentation sidecar contains prompted behavior {needle}")
                for needle in prompted_terms
                if needle in text
            )
            continue
        if "plugin = None" not in text:
            violations.append((path, 0, "non-segmentation sidecar is not inert"))
        for needle in prompted_terms + semantic_terms:
            if needle in text:
                violations.append(
                    (
                        path,
                        0,
                        f"non-segmentation sidecar contains segmentation behavior {needle}",
                    )
                )

    assert not violations, _format_violations(violations)


def test_shared_diffusion_harness_has_no_named_family_behavior_branches() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep diffusion-family test behavior in tests/e2e/models/<family>/e2e_plugins.
    Preconditions: shared diffusion runner/reference provide only generic strategies.
    Postconditions: named diffusion family runner/reference behavior is absent from shared files.
    """
    forbidden = (
        "class DiffusionMediaRunner",
        "class HfDiffusersReference",
        "class DiffusionComparator",
        "class DiffusionPlugin",
        "plugin = Diffusion",
        "crossover_ref_t5_trt_dit",
        "crossover_trt_t5_ref_dit",
        "t5_encode",
        "dit_step",
        "qwen_image",
        "Qwen-Image",
        "QwenImage",
        "diffusion_qwen_image",
        "qwen_image_initial",
        "flux",
        "FluxPipeline",
        "Flux2Pipeline",
        "diffusion_flux",
        "z_image",
        "Z-Image",
        "diffusion_zimage",
        "pixart",
        "PixArt",
        "PixArtSigmaPipeline",
        "diffusion_pixart",
        "ltx_video",
        "LTXPipeline",
        "diffusion_ltx",
        "WanPipeline",
        "diffusion_wan",
        "wan_guidance",
        "case.family ==",
        "family in (",
    )
    violations = []
    for path in E2E_SHARED_DIFFUSION_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared diffusion harness contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_diffusion_runner_and_scheduler_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent shared Python diffusion runtime/scheduler behavior from
    coupling diffusion model families.
    Preconditions: diffusion families that use the Python runner carry local
    runner and scheduler copies.
    Postconditions: root diffusion_runner/schedulers implementations are absent
    and model-owned callers import their owning family package.
    """
    owners = ("flux", "pixart", "wan_t2v", "z_image")
    retired_shared_paths = (
        REPO_ROOT / "python" / "tensorrt_model_connect" / "diffusion_runner.py",
        REPO_ROOT / "python" / "tensorrt_model_connect" / "schedulers" / "__init__.py",
        REPO_ROOT / "python" / "tensorrt_model_connect" / "schedulers" / "base.py",
        REPO_ROOT / "python" / "tensorrt_model_connect" / "schedulers" / "flow_match_euler.py",
    )
    violations = []
    for path in retired_shared_paths:
        if path.exists():
            violations.append((path, 0, "shared diffusion runner/scheduler must be retired"))

    for owner in owners:
        family_dir = FAMILIES / owner
        required = (
            family_dir / "diffusion_runner.py",
            family_dir / "schedulers" / "__init__.py",
            family_dir / "schedulers" / "base.py",
            family_dir / "schedulers" / "flow_match_euler.py",
        )
        for path in required:
            if not path.is_file():
                violations.append((path, 0, "missing family-owned diffusion runner/scheduler"))

        runner_path = E2E_MODELS / owner / "e2e_plugins" / "runners" / "diffusion.py"
        if runner_path.is_file():
            text = runner_path.read_text(encoding="utf-8", errors="ignore")
            shared_imports = (
                "tensorrt_model_connect.diffusion_runner",
                "tensorrt_model_connect.schedulers",
            )
            for needle in shared_imports:
                if needle in text:
                    violations.append((runner_path, 0, f"imports retired shared {needle}"))
            expected = f"tensorrt_model_connect.families.{owner}.diffusion_runner"
            if expected not in text:
                violations.append((runner_path, 0, "missing family-owned diffusion runner import"))

    for path in sorted(FAMILIES.glob("*/debug_diffusion_pipeline.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in (
            "tensorrt_model_connect.diffusion_runner",
            "tensorrt_model_connect.schedulers",
        ):
            if needle in text:
                violations.append((path, 0, f"debug pipeline imports retired shared {needle}"))

    assert not violations, _format_violations(violations)


def test_cpp_diffusion_scheduler_helpers_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep branchy diffusion scheduler policy out of shared runtime domains.
    Preconditions: diffusion runtime families carry their own scheduler helper
    copies when they need C++ scheduler state or request fallback policy.
    Postconditions: the retired shared scheduler helper is absent and all
    remaining scheduler helper includes point at model-owned files.
    """
    retired_shared = RUNTIME_DIFFUSION_DOMAINS / "diffusion_scheduler_helpers.h"
    required_owned = {
        "flux": RUNTIME_MODELS / "flux" / "flux_scheduler_helpers.h",
        "wan": RUNTIME_MODELS / "wan" / "wan_scheduler_helpers.h",
        "pixart": RUNTIME_MODELS / "pixart" / "pixart_scheduler_helpers.h",
        "z_image": RUNTIME_MODELS / "z_image" / "z_image_scheduler_helpers.h",
        "ltx_video": RUNTIME_MODELS / "ltx_video" / "ltx_video_scheduler_helpers.h",
    }
    violations = []
    if retired_shared.exists():
        violations.append((retired_shared, 0, "shared C++ scheduler helper must be retired"))
    for family, path in required_owned.items():
        if not path.is_file():
            violations.append((path, 0, f"missing {family}-owned scheduler helper"))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        expected_namespace = f"namespace {family}_scheduler"
        if expected_namespace not in text:
            violations.append((path, 0, "scheduler helper lacks family-owned namespace"))
        if "FlowMatchEulerState" not in text:
            violations.append((path, 0, "scheduler helper lacks owned scheduler state"))

    forbidden_include = "runtime/domains/diffusion/diffusion_scheduler_helpers.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if forbidden_include in text:
                violations.append((path, 0, "includes retired shared scheduler helper"))

    allowed_owned_roots = {
        family: (
            RUNTIME_MODELS / family,
            REPO_ROOT / "tests" / "cpp" / "models" / family,
        )
        for family in required_owned
    }
    for family, helper_path in required_owned.items():
        include_needle = f"runtime/models/{family}/{helper_path.name}"
        for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if include_needle not in text:
                    continue
                if not any(path.is_relative_to(allowed) for allowed in allowed_owned_roots[family]):
                    violations.append(
                        (path, 0, f"non-{family} file includes {family} scheduler helper")
                    )

    assert not violations, _format_violations(violations)


def test_cpp_diffusion_preprocessor_parser_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep diffusion model weight-key parsing out of shared runtime domains.
    Preconditions: diffusion runtime families carry their own preprocessor parser
    copies in model-owned helper sources.
    Postconditions: the retired shared parser source is absent, trtmc_core does
    not link it, and model key strings live only under model runtime folders.
    """
    retired_shared = (
        RUNTIME_DIFFUSION_DOMAINS / "diffusion_preprocessor.cpp",
        RUNTIME_DIFFUSION_DOMAINS / "diffusion_preprocessor_weights_helpers.h",
    )
    required_owned = {
        "flux": (
            RUNTIME_MODELS / "flux" / "diffusion_helpers.cpp",
            RUNTIME_MODELS / "flux" / "preprocessor_weights_helpers.h",
            "flux_preprocessor_weights",
        ),
        "wan": (
            RUNTIME_MODELS / "wan" / "diffusion_helpers.cpp",
            RUNTIME_MODELS / "wan" / "preprocessor_weights_helpers.h",
            "wan_preprocessor_weights",
        ),
        "pixart": (
            RUNTIME_MODELS / "pixart" / "diffusion_helpers.cpp",
            RUNTIME_MODELS / "pixart" / "preprocessor_weights_helpers.h",
            "pixart_preprocessor_weights",
        ),
        "z_image": (
            RUNTIME_MODELS / "z_image" / "diffusion_helpers.cpp",
            RUNTIME_MODELS / "z_image" / "preprocessor_weights_helpers.h",
            "z_image_preprocessor_weights",
        ),
        "ltx_video": (
            RUNTIME_MODELS / "ltx_video" / "diffusion_helpers.cpp",
            RUNTIME_MODELS / "ltx_video" / "preprocessor_weights_helpers.h",
            "ltx_video_preprocessor_weights",
        ),
        "qwen_image": (
            RUNTIME_MODELS / "qwen_image" / "diffusion_helpers.cpp",
            RUNTIME_MODELS / "qwen_image" / "preprocessor_weights_helpers.h",
            "qwen_image_preprocessor_weights",
        ),
    }
    violations = []
    for path in retired_shared:
        if path.exists():
            violations.append((path, 0, "shared diffusion preprocessor helper must be retired"))

    cmake_text = CMAKE_ROOT.read_text(encoding="utf-8", errors="ignore")
    if "src/runtime/domains/diffusion/diffusion_preprocessor.cpp" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "trtmc_core links retired diffusion preprocessor parser"))
    if "src/runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h" in cmake_text:
        violations.append((CMAKE_ROOT, 0, "trtmc_core references retired preprocessor helper"))

    required_parser_terms = (
        "parse_preprocessor_weights",
        "condition_embedder.time_embedding.0.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "x_embedder.weight",
        "vae_bn.running_mean",
    )
    required_wire_terms = (
        "inline bool extract_preprocessor_index",
        "inline bool parse_shape_csv",
        "inline bool find_preprocessor_entry",
        "inline bool load_preprocessor_floats",
        "inline bool load_with_fallback",
    )
    for family, (parser_path, helper_path, namespace) in required_owned.items():
        if not parser_path.is_file():
            violations.append((parser_path, 0, f"missing {family}-owned preprocessor parser"))
            continue
        text = parser_path.read_text(encoding="utf-8", errors="ignore")
        for term in required_parser_terms:
            if term not in text:
                violations.append((parser_path, 0, f"missing model-owned parser term {term}"))
        if f"{namespace}::" not in text:
            violations.append((parser_path, 0, f"parser does not call {namespace}"))

        if not helper_path.is_file():
            violations.append((helper_path, 0, f"missing {family}-owned preprocessor helper"))
            continue
        helper_text = helper_path.read_text(encoding="utf-8", errors="ignore")
        if f"namespace {namespace}" not in helper_text:
            violations.append((helper_path, 0, "preprocessor helper lacks family namespace"))
        for term in required_wire_terms:
            if term not in helper_text:
                violations.append((helper_path, 0, f"missing preprocessor helper term {term}"))

    forbidden_model_terms = (
        "condition_embedder.",
        "time_text_embed.",
        "x_embedder.",
        "context_embedder.",
        "vae_bn.running_",
        "extract_preprocessor_index",
        "load_preprocessor_floats",
        "load_with_fallback",
    )
    for path in sorted(RUNTIME_DIFFUSION_DOMAINS.rglob("*")):
        if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden_model_terms:
            if term in text:
                violations.append((path, 0, f"shared diffusion domain contains model key {term}"))

    forbidden_include = "runtime/domains/diffusion/diffusion_preprocessor_weights_helpers.h"
    for root in (REPO_ROOT / "src", REPO_ROOT / "include", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".h", ".hpp", ".cpp", ".cu", ".cc"}:
                continue
            if forbidden_include in path.read_text(encoding="utf-8", errors="ignore"):
                violations.append((path, 0, "includes retired shared preprocessor helper"))

    assert not violations, _format_violations(violations)


def test_diffusion_model_plugins_do_not_name_sibling_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep each diffusion E2E plugin folder owned by exactly one family.
    Preconditions: diffusion model folders provide local runner/reference files.
    Postconditions: no diffusion model-owned runner/reference carries sibling family behavior.
    """
    family_terms = {
        "flux": (
            "flux",
            "FluxPipeline",
            "Flux2Pipeline",
            "diffusion_flux",
        ),
        "z_image": (
            "z_image",
            "Z-Image",
            "diffusion_zimage",
        ),
        "pixart": (
            "pixart",
            "PixArt",
            "PixArtSigmaPipeline",
            "diffusion_pixart",
        ),
        "ltx_video": (
            "ltx_video",
            "LTXPipeline",
            "diffusion_ltx",
        ),
        "qwen_image": (
            "qwen_image",
            "Qwen-Image",
            "QwenImage",
            "diffusion_qwen_image",
            "qwen_image_initial",
        ),
        "wan_t2v": (
            "wan_t2v",
            "WanPipeline",
            "diffusion_wan",
            "wan_guidance",
        ),
    }
    relative_files = (
        "e2e_plugins/runner.py",
        "e2e_plugins/reference.py",
        "e2e_plugins/comparator.py",
        "e2e_plugins/runners/diffusion.py",
        "e2e_plugins/references/hf_diffusers.py",
        "e2e_plugins/comparators/diffusion.py",
    )
    violations = []
    for owner, owner_terms in family_terms.items():
        owner_dir = E2E_MODELS / owner
        sibling_terms = tuple(
            term for family, terms in family_terms.items() if family != owner for term in terms
        )
        for rel in relative_files:
            path = owner_dir / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in sibling_terms:
                if needle in text:
                    violations.append(
                        (
                            path,
                            0,
                            f"{owner} diffusion E2E plugin contains sibling behavior {needle}",
                        )
                    )

    assert not violations, _format_violations(violations)


def test_non_diffusion_model_plugins_do_not_carry_diffusion_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep generated non-diffusion model folders free of diffusion-family code.
    Preconditions: only diffusion model families own diffusion runner/reference behavior.
    Postconditions: unused diffusion sidecars in other model folders are inert placeholders.
    """
    diffusion_families = {"flux", "z_image", "pixart", "ltx_video", "qwen_image", "wan_t2v"}
    forbidden = (
        "class DiffusionMediaRunner",
        "class HfDiffusersReference",
        "class DiffusionComparator",
        "plugin = Diffusion",
        "t5_encode",
        "dit_step",
        "FluxPipeline",
        "Flux2Pipeline",
        "diffusion_flux",
        "z_image",
        "Z-Image",
        "diffusion_zimage",
        "PixArt",
        "PixArtSigmaPipeline",
        "diffusion_pixart",
        "LTXPipeline",
        "diffusion_ltx",
        "Qwen-Image",
        "QwenImage",
        "diffusion_qwen_image",
        "qwen_image_initial",
        "WanPipeline",
        "diffusion_wan",
        "wan_guidance",
        "family in (",
    )
    relative_files = (
        "e2e_plugins/runners/diffusion.py",
        "e2e_plugins/references/hf_diffusers.py",
        "e2e_plugins/comparators/diffusion.py",
    )
    violations = []
    for model_dir in sorted(E2E_MODELS.iterdir()):
        if not model_dir.is_dir() or model_dir.name in diffusion_families:
            continue
        for rel in relative_files:
            path = model_dir / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "plugin = None" not in text:
                violations.append((path, 0, "non-diffusion E2E plugin is not inert"))
            violations.extend(
                (path, 0, f"non-diffusion E2E plugin contains diffusion behavior {needle}")
                for needle in forbidden
                if needle in text
            )

    assert not violations, _format_violations(violations)


def test_shared_hf_transformers_reference_has_no_named_family_branches() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep named HF Transformers reference variants in model-owned plugins.
    Preconditions: model folders provide local hf_transformers reference copies.
    Postconditions: shared HF Transformers reference contains no concrete behavior.
    """
    forbidden = (
        "class HfTransformersReference",
        "run_reference_subprocess",
        "_run_full_generation",
        "_run_full_inference",
        "_decode_vl_generated_text",
        "_torch_dtype_for_case",
        "AutoModel",
        "AutoTokenizer",
        "subprocess",
        "textwrap",
        "qwen",
        "Qwen",
        "internvl",
        "InternVL",
        "locateanything",
        "LocateAnything",
        "canary",
        "Canary",
        "nemotron_labs_diffusion",
        "Nemotron Labs",
        "linear_spec",
        "nemo_canary",
        "bark",
        "Bark",
        "_run_text_to_audio_ref",
        "DPRContextEncoder",
        "dpr_context_embed",
        'model_type == "dpr"',
        "model_type == 'dpr'",
        "import timm",
        "timm.create_model",
        "_run_image_classification_ref",
        "_run_prompted_segmentation_ref",
        "SamModel",
        "SamProcessor",
        "hf_prompted_segmentation",
        "_vl_fallback_prompt",
        "_run_canary_ref",
        "_run_locateanything",
    )
    text = E2E_SHARED_HF_TRANSFORMERS.read_text(encoding="utf-8")
    violations = [
        (E2E_SHARED_HF_TRANSFORMERS, 0, f"shared HF Transformers reference contains {needle}")
        for needle in forbidden
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_shared_e2e_placeholder_sidecars_have_no_runtime_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete E2E task behavior in model-owned sidecars.
    Preconditions: shared sidecars exist only for import/discovery compatibility.
    Postconditions: placeholder sidecars contain no classes, functions, or branches.
    """
    forbidden_nodes = (
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Match,
    )
    violations = []
    for path in E2E_SHARED_PLACEHOLDER_SIDECARS:
        if not path.is_file():
            violations.append((path, 0, "missing shared placeholder sidecar"))
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "plugin = None" not in text:
            violations.append((path, 0, "shared placeholder sidecar is not inert"))
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, forbidden_nodes):
                violations.append(
                    (
                        path,
                        getattr(node, "lineno", 0),
                        f"shared placeholder contains {type(node).__name__}",
                    )
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                    violations.append(
                        (
                            path,
                            getattr(node, "lineno", 0),
                            "shared placeholder imports runtime code",
                        )
                    )

    assert not violations, _format_violations(violations)


def test_diffusion_build_cli_args_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep diffusion build-shape CLI policy out of the shared E2E harness.
    Preconditions: family MODEL.toml files can declare build_cli_args mappings.
    Postconditions: orchestrator appends generic declared CLI args only; concrete
    diffusion build flags live in owning model test families.
    """
    diffusion_build_flags = (
        "--image-height",
        "--image-width",
        "--video-height",
        "--video-width",
        "--video-num-frames",
        "--num-inference-steps",
    )
    shared_text = E2E_ORCHESTRATOR.read_text(encoding="utf-8", errors="ignore")
    violations = [
        (E2E_ORCHESTRATOR, 0, f"shared orchestrator owns diffusion build flag {flag}")
        for flag in diffusion_build_flags
        if flag in shared_text
    ]
    if "diffusion_build_args" in shared_text:
        violations.append(
            (
                E2E_ORCHESTRATOR,
                0,
                "shared orchestrator contains diffusion-specific build arg mapping",
            )
        )

    for family in ("flux", "ltx_video", "pixart", "qwen_image", "wan_t2v", "z_image"):
        manifest = E2E_MODELS / family / "MODEL.toml"
        text = manifest.read_text(encoding="utf-8", errors="ignore")
        if "build_cli_args" not in text:
            violations.append((manifest, 0, "missing family-owned build_cli_args"))
            continue
        for flag in diffusion_build_flags:
            if flag not in text:
                violations.append((manifest, 0, f"missing family-owned {flag} mapping"))

    assert not violations, _format_violations(violations)


def test_diffusion_bundle_sections_config_and_tokenizers_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep diffusion component-to-bundle-section and component-derived
    config/tokenizer policy with each model family.
    Preconditions: diffusion family plugins expose diffusion_bundle_sections()
    diffusion_bundle_config(), and diffusion tokenizer hooks.
    Postconditions: shared engine_builder only asks the plugin for component
    sections/config/tokenizers and does not hard-code component names or
    optional features.
    """
    text = ENGINE_BUILDER.read_text(encoding="utf-8", errors="ignore")
    start = text.index("def _build_diffusion_bundle(")
    end = text.index("\ndef build(", start)
    diffusion_body = text[start:end]
    violations = []
    for snippet in (
        'BundleSection("denoiser_plan"',
        'BundleSection("vae_decoder_plan"',
        'BundleSection("vision_engine_plan"',
        'BundleSection("vae_encoder_plan"',
        'BundleSection("preprocessor_weights"',
        'components["denoiser"]',
        'components["vae_decoder"]',
        'components["preprocessor_weights"]',
        'components["text_encoders"]',
        'components.get("denoiser_ranks")',
        'components["denoiser_ranks"]',
        '"num_text_encoders":',
        '"tokenizer_2"',
        '"clip_tokenizer.json"',
        '"clip_vocab.json"',
        '"clip_merges.txt"',
        "clip_file_map",
        "for tok_subdir in",
        '"vision_engine" in components',
        '"vae_encoder" in components',
    ):
        if snippet in diffusion_body:
            violations.append(
                (
                    ENGINE_BUILDER,
                    0,
                    f"shared diffusion builder owns bundle section policy {snippet}",
                )
            )
    if "_diffusion_bundle_sections_from_plugin(plugin, components, parallel)" not in diffusion_body:
        violations.append(
            (
                ENGINE_BUILDER,
                0,
                "shared diffusion builder does not delegate section assembly to plugin",
            )
        )
    if 'getattr(plugin, "diffusion_bundle_config", None)' not in diffusion_body:
        violations.append(
            (
                ENGINE_BUILDER,
                0,
                "shared diffusion builder does not delegate component-derived config to plugin",
            )
        )
    if "_diffusion_tokenizer_add_special_tokens_from_plugin(" not in diffusion_body:
        violations.append(
            (
                ENGINE_BUILDER,
                0,
                "shared diffusion builder does not delegate tokenizer add-special policy to plugin",
            )
        )
    if "_diffusion_tokenizer_bundle_sections_from_plugin(" not in diffusion_body:
        violations.append(
            (
                ENGINE_BUILDER,
                0,
                "shared diffusion builder does not delegate tokenizer sections to plugin",
            )
        )

    for family in ("flux", "ltx_video", "pixart", "qwen_image", "wan_t2v", "z_image"):
        plugin_path = FAMILIES / family / "plugin.py"
        plugin_text = plugin_path.read_text(encoding="utf-8", errors="ignore")
        if "def diffusion_bundle_sections(" not in plugin_text:
            violations.append(
                (
                    plugin_path,
                    0,
                    "missing family-owned diffusion_bundle_sections()",
                )
            )
        if "def diffusion_bundle_config(" not in plugin_text:
            violations.append(
                (
                    plugin_path,
                    0,
                    "missing family-owned diffusion_bundle_config()",
                )
            )
        if "def diffusion_tokenizer_add_special_tokens(" not in plugin_text:
            violations.append(
                (
                    plugin_path,
                    0,
                    "missing family-owned diffusion_tokenizer_add_special_tokens()",
                )
            )
        if "def diffusion_tokenizer_bundle_sections(" not in plugin_text:
            violations.append(
                (
                    plugin_path,
                    0,
                    "missing family-owned diffusion_tokenizer_bundle_sections()",
                )
            )

    assert not violations, _format_violations(violations)


def test_image_repro_cli_args_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep image input CLI policy out of shared repro command rendering.
    Preconditions: image-owning model families can expose e2e_plugins/repro.py.
    Postconditions: shared orchestrator does not append --image; every manifest
    family with image inputs has a model-owned repro provider.
    """
    shared_text = E2E_ORCHESTRATOR.read_text(encoding="utf-8", errors="ignore")
    loader_text = E2E_MANIFEST_LOADER.read_text(encoding="utf-8", errors="ignore")
    violations = []
    if '"--image"' in shared_text or "'--image'" in shared_text:
        violations.append((E2E_ORCHESTRATOR, 0, "shared orchestrator owns --image CLI arg"))
    for snippet in (
        'inputs["image"] = manifest["test_image"]',
        'args={"path": manifest["test_image"]}',
        "Vision/segmentation need test image",
    ):
        if snippet in loader_text:
            violations.append(
                (
                    E2E_MANIFEST_LOADER,
                    0,
                    f"shared manifest loader owns image-specific mapping {snippet}",
                )
            )

    image_families: set[str] = set()
    for manifest in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        inputs = raw.get("inputs", {})
        if (
            raw.get("test_image")
            or raw.get("image_path")
            or (isinstance(inputs, dict) and ("image" in inputs or "image_path" in inputs))
        ):
            image_families.add(manifest.parent.parent.name)

    for family in sorted(image_families):
        provider = E2E_MODELS / family / "e2e_plugins" / "repro.py"
        if not provider.is_file():
            violations.append((provider, 0, "missing image-family repro provider"))
            continue
        text = provider.read_text(encoding="utf-8", errors="ignore")
        if "repro_provider" not in text:
            violations.append((provider, 0, "repro provider module does not export repro_provider"))
        family_manifest = E2E_MODELS / family / "MODEL.toml"
        manifest_text = family_manifest.read_text(encoding="utf-8", errors="ignore")
        for snippet in (
            "input_fields = [",
            '{ input = "image", manifest = "test_image" }',
            'preflight_asset_fields = ["test_image"]',
        ):
            if snippet not in manifest_text:
                violations.append(
                    (
                        family_manifest,
                        0,
                        f"missing image-owned loader mapping {snippet}",
                    )
                )

    assert not violations, _format_violations(violations)


def test_audio_input_loader_mappings_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep audio asset and input normalization policy in owning families.
    Preconditions: audio-owning families can declare input_fields and
    preflight_asset_fields in MODEL.toml.
    Postconditions: shared manifest loader has no audio/speech asset branches;
    every manifest family with audio inputs declares the mapping locally.
    """
    loader_text = E2E_MANIFEST_LOADER.read_text(encoding="utf-8", errors="ignore")
    violations = []
    for snippet in (
        'inputs["audio"] = manifest["test_input_audio"]',
        'args={"path": manifest["test_input_audio"]}',
        'args={"path": manifest["speech_reference_tokens"]}',
        "Audio models need test audio",
        "Speech reference tokens",
    ):
        if snippet in loader_text:
            violations.append(
                (
                    E2E_MANIFEST_LOADER,
                    0,
                    f"shared manifest loader owns audio-specific mapping {snippet}",
                )
            )

    audio_families: set[str] = set()
    token_families: set[str] = set()
    for manifest in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        family = manifest.parent.parent.name
        if raw.get("test_input_audio"):
            audio_families.add(family)
        if raw.get("speech_reference_tokens"):
            token_families.add(family)

    for family in sorted(audio_families):
        family_manifest = E2E_MODELS / family / "MODEL.toml"
        text = family_manifest.read_text(encoding="utf-8", errors="ignore")
        for snippet in (
            "input_fields = [",
            '{ input = "audio", manifest = "test_input_audio" }',
            "test_input_audio",
        ):
            if snippet not in text:
                violations.append(
                    (
                        family_manifest,
                        0,
                        f"missing audio-owned loader mapping {snippet}",
                    )
                )

    for family in sorted(token_families):
        family_manifest = E2E_MODELS / family / "MODEL.toml"
        text = family_manifest.read_text(encoding="utf-8", errors="ignore")
        if "speech_reference_tokens" not in text:
            violations.append(
                (
                    family_manifest,
                    0,
                    "missing speech-reference-token asset mapping",
                )
            )

    assert not violations, _format_violations(violations)


def test_model_specific_manifest_inputs_are_family_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep task/model input normalization out of the shared manifest loader.
    Preconditions: family MODEL.toml files can declare input_fields mappings.
    Postconditions: segmentation, diffusion, and neural-operator input fields
    are declared by owning model families, not central loader branches.
    """
    loader_text = E2E_MANIFEST_LOADER.read_text(encoding="utf-8", errors="ignore")
    violations = []
    for snippet in (
        'inputs["point_x"] = manifest["point_x"]',
        "Point prompts for prompted segmentation",
        'inputs["video_num_frames"] = manifest["video_num_frames"]',
        'manifest.get("video_num_frames")',
        'inputs["image_height"] = manifest["image_height"]',
        'manifest.get("image_height")',
        'for key in ("negative_prompt", "cfg_scale", "height", "width",',
        'for key in ("temperature", "top_p", "top_k", "min_p", "seed", "guidance_scale")',
        'for key in ("field_input", "branch_input", "trunk_input", "output_field")',
    ):
        if snippet in loader_text:
            violations.append(
                (
                    E2E_MANIFEST_LOADER,
                    0,
                    f"shared manifest loader owns model-specific input mapping {snippet}",
                )
            )

    model_input_fields = (
        "point_x",
        "point_y",
        "num_expected_masks",
        "video_num_frames",
        "video_height",
        "video_width",
        "image_height",
        "image_width",
        "negative_prompt",
        "cfg_scale",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "field_input",
        "branch_input",
        "trunk_input",
        "output_field",
    )
    fields_by_family: dict[str, set[str]] = {}
    for manifest in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        family = manifest.parent.parent.name
        fields = {field for field in model_input_fields if field in raw}
        if fields:
            fields_by_family.setdefault(family, set()).update(fields)

    for family, fields in sorted(fields_by_family.items()):
        family_manifest = E2E_MODELS / family / "MODEL.toml"
        text = family_manifest.read_text(encoding="utf-8", errors="ignore")
        if "input_fields = [" not in text:
            violations.append((family_manifest, 0, "missing family-owned input_fields"))
            continue
        for field in sorted(fields):
            snippet = f'manifest = "{field}"'
            if snippet not in text:
                violations.append(
                    (
                        family_manifest,
                        0,
                        f"missing family-owned input mapping for {field}",
                    )
                )

    assert not violations, _format_violations(violations)


def test_text_generation_runtime_and_e2e_behavior_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent shared text-generation/seq2seq implementations from
    coupling model families.
    Preconditions: model-owned runtime DSOs and E2E sidecars are generated.
    Postconditions: legacy shared runtime model folders are absent, build files
    do not reference them, and shared text E2E sidecars remain inert.
    """
    violations = []
    legacy_runtime_dirs = (
        RUNTIME_MODELS / "text_generation",
        RUNTIME_MODELS / "seq2seq",
    )
    for path in legacy_runtime_dirs:
        if path.exists():
            violations.append((path, 0, "shared text runtime model folder must be removed"))

    build_files = (
        CMAKE_ROOT,
        REPO_ROOT / "cmake" / "trtmc_pipeline_plugins.cmake",
    )
    forbidden_build_terms = (
        "runtime/models/text_generation",
        "runtime/models/seq2seq",
        "libtrtmc_model_text_generation",
        "libtrtmc_model_seq2seq",
    )
    for path in build_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden_build_terms:
            if term in text:
                violations.append((path, 0, f"shared build file references {term}"))

    forbidden_runtime_copy_terms = (
        "serves ALL decoder-only LLMs",
        "one class, many models",
    )
    for path in sorted(RUNTIME_MODELS.glob("*/pipeline.h")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden_runtime_copy_terms:
            if term in text:
                violations.append(
                    (path, 0, f"model-owned runtime copy uses shared-text wording: {term}")
                )

    placeholder_paths = (
        E2E_SHARED_TEXT_GENERATION_RUNNER,
        REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "text.py",
        E2E_SHARED_HF_TRANSFORMERS,
    )
    for path in placeholder_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text, filename=str(path))
        if "plugin = None" not in text:
            violations.append((path, 0, "shared text E2E sidecar must be inert"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                violations.append(
                    (
                        path,
                        getattr(node, "lineno", 0),
                        "shared text E2E sidecar defines concrete behavior",
                    )
                )

    assert not violations, _format_violations(violations)


def test_nemotron_labs_diffusion_generation_cli_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Nemotron Labs Diffusion CLI switches out of the shared text runner.
    Preconditions: Nemotron Labs Diffusion owns a local text-generation runner.
    Postconditions: shared text generation has only generic sampling flags.
    """
    forbidden = (
        "--generation-mode",
        "--block-length",
        "--threshold",
        "generation_mode",
        "block_length",
        'inputs.get("threshold")',
    )
    shared_text = E2E_SHARED_TEXT_GENERATION_RUNNER.read_text(encoding="utf-8")
    violations = [
        (E2E_SHARED_TEXT_GENERATION_RUNNER, 0, f"shared text runner contains {needle}")
        for needle in forbidden
        if needle in shared_text
    ]

    owned_runner = (
        E2E_MODELS / "nemotron_labs_diffusion" / "e2e_plugins" / "runners" / "text_generation.py"
    )
    owned_text = owned_runner.read_text(encoding="utf-8") if owned_runner.is_file() else ""
    if not owned_runner.is_file():
        violations.append((owned_runner, 0, "missing Nemotron-owned text-generation runner"))
    for needle in forbidden:
        if needle not in owned_text:
            violations.append((owned_runner, 0, f"missing Nemotron-owned CLI term {needle}"))

    model_test = E2E_MODELS / "nemotron_labs_diffusion" / "test_nemotron_labs_diffusion_runner.py"
    if not model_test.is_file():
        violations.append((model_test, 0, "missing Nemotron-owned runner unit test"))

    assert not violations, _format_violations(violations)


def test_model_e2e_text_runners_use_family_owned_debug_runners() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep active text-generation E2E debug execution model-owned.
    Preconditions: active text E2E manifests declare their runtime strategies.
    Postconditions: model-local text runners call family debug_runner.py
    factories instead of the shared debug_runner.runner_from_bundle dispatcher.
    """
    text_runtime_strategies = {
        "mamba_ssm_recurrent",
        "rwkv_recurrent",
        "nemotron_h_hybrid_mamba_attention",
        "qwen3_5_hybrid_mamba_attention",
        "marian_translation",
        "seq2seq_encoder_decoder",
        "text_to_text",
        "nemotron_labs_diffusion",
    }
    active_families: set[str] = set()
    for manifest_path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        runtime_strategy = str(manifest.get("runtime_strategy") or "")
        task_strategy = str(manifest.get("task_strategy") or "")
        if (
            task_strategy == "text_generation_causal"
            or runtime_strategy.endswith("_decoder_kv_cache")
            or runtime_strategy.endswith("_decoder_moe")
            or runtime_strategy in text_runtime_strategies
        ):
            active_families.add(manifest_path.parents[1].name)

    violations = []
    for runner_path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/text_generation.py")):
        text = runner_path.read_text(encoding="utf-8")
        family = runner_path.parents[2].name
        tree = ast.parse(text, filename=str(runner_path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "tensorrt_model_connect.debug_runner"
                and any(alias.name == "runner_from_bundle" for alias in node.names)
            ):
                violations.append(
                    (
                        runner_path,
                        node.lineno,
                        "model text runner imports shared runner_from_bundle",
                    )
                )
        for match in re.finditer(
            r"from tensorrt_model_connect\.debug_runner import \((.*?)\)",
            text,
            flags=re.DOTALL,
        ):
            imported_block = match.group(1)
            leaked = sorted(
                name
                for name in (
                    "runner_from_bundle",
                    "load_config_from_bundle",
                    "load_engine_from_bundle",
                    "load_section_from_bundle",
                )
                if name in imported_block
            )
            if leaked:
                line = text.count("\n", 0, match.start()) + 1
                violations.append(
                    (
                        runner_path,
                        line,
                        "embedded text runner script imports shared helper(s): "
                        + ", ".join(leaked),
                    )
                )
        if family not in active_families:
            if "plugin = None" not in text:
                violations.append((runner_path, 0, "inactive text runner is not inert"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    violations.append(
                        (runner_path, node.lineno, "inactive text runner defines behavior")
                    )
        elif "plugin = TextGenerationCausalRunner()" not in text:
            violations.append((runner_path, 0, "active text runner is not registered"))

    for runner_path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/*.py")):
        if runner_path.name == "text_generation.py":
            continue
        text = runner_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(runner_path))
        imports_common = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 1 and node.module == "text_generation":
                violations.append(
                    (runner_path, node.lineno, "runner imports text_generation helpers")
                )
            if node.level == 1 and node.module == "_runtime_common":
                imports_common = True
        if imports_common:
            common_path = runner_path.parent / "_runtime_common.py"
            if not common_path.is_file():
                violations.append((common_path, 0, "missing family-local runtime helpers"))

    for comparator_path in sorted(E2E_MODELS.glob("*/e2e_plugins/comparators/text.py")):
        text = comparator_path.read_text(encoding="utf-8")
        family = comparator_path.parents[2].name
        tree = ast.parse(text, filename=str(comparator_path))
        if family not in active_families:
            if "plugin = None" not in text:
                violations.append((comparator_path, 0, "inactive text comparator is not inert"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    violations.append(
                        (comparator_path, node.lineno, "inactive text comparator defines behavior")
                    )
        elif "plugin = TextComparator()" not in text:
            violations.append((comparator_path, 0, "active text comparator is not registered"))

    for comparator_root in sorted(E2E_MODELS.glob("*/e2e_plugins/comparator.py")):
        family = comparator_root.parents[1].name
        text = comparator_root.read_text(encoding="utf-8")
        if "from .comparators.text import" in text and family not in active_families:
            violations.append((comparator_root, 0, "non-text family imports text comparator"))

    for family in sorted(active_families):
        runner_path = E2E_MODELS / family / "e2e_plugins" / "runners" / "text_generation.py"
        family_debug_runner = FAMILIES / family / "debug_runner.py"
        if not runner_path.is_file():
            violations.append((runner_path, 0, "missing model-owned text runner"))
            continue
        if not family_debug_runner.is_file():
            violations.append((family_debug_runner, 0, "missing family-owned debug runner"))
            continue
        text = runner_path.read_text(encoding="utf-8")
        expected_import = f"from tensorrt_model_connect.families.{family}.debug_runner import"
        if expected_import not in text:
            violations.append(
                (runner_path, 0, "active text runner missing family debug_runner import")
            )
        if "family_runner_from_bundle(" not in text:
            violations.append(
                (runner_path, 0, "active text runner missing family runner factory call")
            )

    assert not violations, _format_violations(violations)


def test_model_debug_runner_tests_use_family_owned_factories() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model debug-runner tests pointed at model-owned factories.
    Preconditions: model tests exercise family debug_runner.py adapters.
    Postconditions: model-local debug-runner tests do not call the shared
    debug_runner dispatcher or bundle helper surface.
    """
    forbidden = {
        "runner_from_bundle",
        "load_config_from_bundle",
        "load_engine_from_bundle",
        "load_section_from_bundle",
    }
    violations = []
    for path in sorted(E2E_MODELS.glob("*/test_*debug_runner.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "tensorrt_model_connect.debug_runner"
            ):
                leaked = sorted(alias.name for alias in node.names if alias.name in forbidden)
                if leaked:
                    violations.append(
                        (
                            path,
                            node.lineno,
                            "model debug-runner test imports shared helper(s): "
                            + ", ".join(leaked),
                        )
                    )

    assert not violations, _format_violations(violations)


def test_model_owned_code_imports_only_debug_runner_infrastructure() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared debug_runner usage limited to infrastructure.
    Preconditions: model-owned Python may need shared distributed setup.
    Postconditions: model-owned code may import TensorParallelNcclGroup, but
    not shared runner factories, bundle readers, preprocessing, or TRT helpers.
    """
    allowed = {"TensorParallelNcclGroup"}
    violations = []
    for root in (FAMILIES, E2E_MODELS):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(
                r"from tensorrt_model_connect\.debug_runner import(?: \((.*?)\)|([^\n]*))",
                text,
                flags=re.DOTALL,
            ):
                body = match.group(1) if match.group(1) is not None else match.group(2) or ""
                if match.group(1) is None:
                    body = body.splitlines()[0]
                imported = []
                for raw_name in body.replace("\n", " ").split(","):
                    name = raw_name.strip()
                    if not name:
                        continue
                    name = name.split(" as ", 1)[0].strip()
                    if name:
                        imported.append(name)
                leaked = sorted(name for name in imported if name not in allowed)
                if leaked:
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(
                        (
                            path,
                            line,
                            "model-owned code imports shared debug_runner symbol(s): "
                            + ", ".join(leaked),
                        )
                    )

    assert not violations, _format_violations(violations)


def test_family_debug_runners_do_not_import_shared_debug_runner_module() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete TRT debug execution owned by each family.
    Preconditions: families expose debug_runner.py modules for runtime tests.
    Postconditions: family debug_runner.py files do not import shared debug
    runner helpers, readers, dispatchers, or runner classes.
    """
    violations = []
    for path in sorted(FAMILIES.glob("*/debug_runner.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "tensorrt_model_connect.debug_runner"
            ):
                imported = ", ".join(alias.name for alias in node.names)
                violations.append(
                    (
                        path,
                        node.lineno,
                        "family debug_runner imports shared debug_runner: " + imported,
                    )
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tensorrt_model_connect.debug_runner":
                        violations.append(
                            (
                                path,
                                node.lineno,
                                "family debug_runner imports shared debug_runner module",
                            )
                        )

    assert not violations, _format_violations(violations)


def test_qwen_debug_runner_path_does_not_import_shared_text_runner_helpers() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Qwen debug execution in Qwen-owned code.
    Preconditions: Qwen has a family debug_runner.py and model-local E2E code.
    Postconditions: Qwen no longer imports shared debug_runner bundle helpers or
    shared debug_runner dispatch for text generation.
    """
    forbidden_helpers = {
        "runner_from_bundle",
        "load_config_from_bundle",
        "load_engine_from_bundle",
        "load_section_from_bundle",
    }
    qwen_family_root = FAMILIES / "qwen"
    qwen_e2e_root = E2E_MODELS / "qwen"
    violations = []
    for path in sorted([*qwen_family_root.rglob("*.py"), *qwen_e2e_root.rglob("*.py")]):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "tensorrt_model_connect.debug_runner"
            ):
                continue
            imported = {alias.name for alias in node.names}
            if path.is_relative_to(qwen_family_root):
                violations.append(
                    (path, node.lineno, "Qwen family code imports shared debug_runner")
                )
                continue
            shared_behavior = sorted(imported & forbidden_helpers)
            if shared_behavior:
                violations.append(
                    (
                        path,
                        node.lineno,
                        "Qwen E2E code imports shared text runner helper(s): "
                        + ", ".join(shared_behavior),
                    )
                )

    assert not violations, _format_violations(violations)


def test_hf_transformers_model_plugins_do_not_name_sibling_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep HF Transformers special reference behavior in owning model folders.
    Preconditions: model-owned hf_transformers.py sidecars are generated per family.
    Postconditions: no HF reference sidecar carries another family's special branch.
    """
    family_term_groups = (
        (
            {"bark"},
            (
                "BarkModel",
                "_run_text_to_audio_ref",
                "hf_text_to_audio",
            ),
        ),
        (
            {"canary"},
            (
                "Canary",
                "canary",
                "_run_canary_ref",
                "nemo_canary_stt",
            ),
        ),
        (
            {"nemotron_speech_streaming"},
            (
                "Nemotron speech",
                "_run_nemo_speech_ref",
                "nemo_speech_stt",
            ),
        ),
        (
            {"nemotron_labs_diffusion"},
            (
                "nemotron_labs_diffusion",
                "Nemotron Labs",
                "linear_spec",
                "_run_nemotron_labs_diffusion_generation",
            ),
        ),
        (
            {"locateanything"},
            (
                "LocateAnything",
                "locateanything",
                "_run_locateanything",
                "patchify_chw",
            ),
        ),
        (
            {"qwen_vl"},
            (
                "qwen",
                "Qwen",
            ),
        ),
        (
            {"internvl"},
            (
                "internvl",
                "InternVL",
            ),
        ),
        (
            {"dpr"},
            (
                "DPRContextEncoder",
                "DPRContextEncoderTokenizerFast",
                "dpr_context_embed",
                'model_type == "dpr"',
                "model_type == 'dpr'",
            ),
        ),
        (
            {"timm_vit"},
            (
                "import timm",
                "timm.create_model",
                "_run_image_classification_ref",
            ),
        ),
        (
            {"sam", "sam3"},
            (
                "SamModel",
                "SamProcessor",
                "_run_prompted_segmentation_ref",
                "hf_prompted_segmentation",
            ),
        ),
    )
    stale_terms = ("_vl_fallback_prompt",)
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/references/hf_transformers.py")):
        owner = path.parts[-4]
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in stale_terms:
            if needle in text:
                violations.append((path, 0, f"{owner} HF reference contains stale {needle}"))
        sibling_terms = tuple(
            term for owners, terms in family_term_groups if owner not in owners for term in terms
        )
        violations.extend(
            (path, 0, f"{owner} HF reference contains sibling behavior {needle}")
            for needle in sibling_terms
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_unused_hf_transformers_model_sidecars_are_inert() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent generated unused HF sidecars from carrying copied reference logic.
    Preconditions: model reference.py files declare the active reference backends.
    Postconditions: hf_transformers.py files are placeholders unless the model imports them.
    """
    behavior_terms = (
        "class HfTransformersReference",
        "AutoModel",
        "AutoTokenizer",
        "run_reference_subprocess",
        "_run_full_generation",
        "_run_full_inference",
    )
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/references/hf_transformers.py")):
        model_dir = path.parents[2]
        reference_entrypoint = model_dir / "e2e_plugins" / "reference.py"
        if not reference_entrypoint.is_file():
            continue
        reference_text = reference_entrypoint.read_text(encoding="utf-8", errors="ignore")
        if "from .references.hf_transformers import" in reference_text:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "plugin = None" not in text:
            violations.append((path, 0, "unused HF reference sidecar is not inert"))
        violations.extend(
            (path, 0, f"unused HF reference sidecar contains {needle}")
            for needle in behavior_terms
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_audio_harness_has_no_named_family_branches() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep audio model behavior in tests/e2e/models/<family>/e2e_plugins.
    Preconditions: shared audio harness files provide only generic orchestration.
    Postconditions: Bark and Magpie runner/reference/comparator behavior is absent.
    """
    forbidden = (
        "bark",
        "Bark",
        "audio_bark",
        "bark_dump",
        "sem_tokens",
        "coarse_tokens",
        "magpie",
        "Magpie",
        "MagpieTTS",
        "audio_magpie",
        "text_to_audio_magpie",
        "nemo_magpie",
    )
    violations = []
    for path in E2E_SHARED_AUDIO_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared audio harness contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_single_family_e2e_task_sidecars_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete E2E task execution in the owning model folder.
    Preconditions: model-local E2E plugin entrypoints import their owned sidecars.
    Postconditions: shared and non-owner copied sidecars are inert placeholders.
    """
    cases = {
        "image_classification": {
            "owners": {"timm_vit"},
            "paths": (
                "e2e_plugins/runners/image_classification.py",
                "e2e_plugins/comparators/image_classification.py",
            ),
            "shared_paths": (
                REPO_ROOT / "tests" / "e2e_harness" / "runners" / "image_classification.py",
                REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "image_classification.py",
            ),
            "behavior_terms": (
                "class ImageClassificationRunner",
                "class ImageClassificationComparator",
                'return "image_classification"',
                "ctx.binary_path",
                "top1_match",
            ),
        },
        "diffusion_text_generation": {
            "owners": {"elf_flow"},
            "paths": (
                "e2e_plugins/runners/diffusion_text_generation.py",
                "e2e_plugins/comparators/diffusion_text_generation.py",
            ),
            "shared_paths": (
                REPO_ROOT / "tests" / "e2e_harness" / "runners" / "diffusion_text_generation.py",
                REPO_ROOT
                / "tests"
                / "e2e_harness"
                / "comparators"
                / "diffusion_text_generation.py",
            ),
            "behavior_terms": (
                "class DiffusionTextGenerationRunner",
                "class DiffusionTextGenerationComparator",
                'return "diffusion_text_generation"',
                "elf_conditional_text",
                "ELF replay",
                "generated_samples",
            ),
        },
    }
    violations = []
    for task, spec in cases.items():
        owners = spec["owners"]
        behavior_terms = spec["behavior_terms"]
        for path in spec["shared_paths"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "plugin = None" not in text:
                violations.append((path, 0, f"shared {task} sidecar is not inert"))
            violations.extend(
                (path, 0, f"shared {task} sidecar contains behavior {needle}")
                for needle in behavior_terms
                if needle in text
            )

        for model_dir in sorted(E2E_MODELS.iterdir()):
            if not model_dir.is_dir() or model_dir.name in owners:
                continue
            for rel in spec["paths"]:
                path = model_dir / rel
                if not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "plugin = None" not in text:
                    violations.append((path, 0, f"non-owner {task} sidecar is not inert"))
                violations.extend(
                    (path, 0, f"non-owner {task} sidecar contains behavior {needle}")
                    for needle in behavior_terms
                    if needle in text
                )

    expected_repro = (
        E2E_MODELS / "elf_flow" / "e2e_plugins" / "repro.py",
        E2E_MODELS / "timm_vit" / "e2e_plugins" / "repro.py",
    )
    violations.extend(
        (path, 0, "missing model-owned repro command provider")
        for path in expected_repro
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_single_family_contract_plugins_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep single-family compare logic in model-owned contract plugins.
    Preconditions: shared contract plugin package is used only for generic contracts.
    Postconditions: shared contract plugins do not define model-family behavior.
    """
    forbidden_files = (
        E2E_SHARED_CONTRACT_PLUGINS / "asr.py",
        E2E_SHARED_CONTRACT_PLUGINS / "tts.py",
        E2E_SHARED_CONTRACT_PLUGINS / "multimodal_chat.py",
        E2E_SHARED_CONTRACT_PLUGINS / "speech_to_speech.py",
        E2E_SHARED_CONTRACT_PLUGINS / "nemotron_labs_diffusion.py",
    )
    violations = [
        (path, 0, "single-family contract plugin must be model-owned")
        for path in forbidden_files
        if path.exists()
    ]

    forbidden_reference_families = (
        "asr_whisper",
        "asr_canary",
        "tts_bark",
        "tts_magpie",
        "chat_qwen3_posttrained",
        "multimodal_chat_qwen35",
        "s2s_personaplex",
        "nemotron_labs_diffusion_model_card",
        "dpr_context_embed",
        "DPRContextEncoder",
        "elf_unconditional_text",
        "elf_conditional_text",
        "ElfDiffusionText",
    )
    for path in sorted(E2E_SHARED_CONTRACT_PLUGINS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared contract plugin contains model reference family {needle}")
            for needle in forbidden_reference_families
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_contract_plugins_do_not_branch_on_reference_family() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific contract setup in manifests or model plugins.
    Preconditions: shared contract plugins may dispatch by registration only.
    Postconditions: shared contract plugins do not inspect concrete families.
    """
    forbidden_terms = (
        "case.reference_family",
        "_PRE_FORMATTED_MARKERS",
    )
    violations = []
    for path in sorted(E2E_SHARED_CONTRACT_PLUGINS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared contract plugin owns model-specific branch {term}")
            for term in forbidden_terms
            if term in text
        )

    assert not violations, _format_violations(violations)


def test_sampling_contract_is_qwen_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep sampling validation with the Qwen family that owns the manifests.
    Preconditions: Qwen top-p manifests declare sampling_top_p.
    Postconditions: shared sampling plugin is inert and Qwen owns the contract.
    """
    shared_text = (E2E_SHARED_CONTRACT_PLUGINS / "sampling.py").read_text(encoding="utf-8")
    qwen_contract = (E2E_MODELS / "qwen" / "e2e_plugins" / "contract.py").read_text(
        encoding="utf-8"
    )

    violations = []
    if "plugin = None" not in shared_text:
        violations.append(
            (
                E2E_SHARED_CONTRACT_PLUGINS / "sampling.py",
                0,
                "shared sampling plugin should be inert",
            )
        )
    if "sampling_top_p" in shared_text:
        violations.append(
            (
                E2E_SHARED_CONTRACT_PLUGINS / "sampling.py",
                0,
                "shared sampling plugin should not own sampling_top_p",
            )
        )
    for needle in ("class QwenSamplingPlugin", 'reference_families = ["sampling_top_p"]'):
        if needle not in qwen_contract:
            violations.append(
                (
                    E2E_MODELS / "qwen" / "e2e_plugins" / "contract.py",
                    0,
                    f"Qwen contract missing {needle}",
                )
            )

    assert not violations, _format_violations(violations)


def test_embedding_and_encoder_contracts_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep embedding and encoder contract behavior with owning model families.
    Preconditions: manifests declare embedding/encoder reference families.
    Postconditions: shared plugins are inert and model contracts own the behavior.
    """
    shared_contracts = {
        "embedding.py": (
            "sentence_transformer_embed",
            "bge_retrieval_embed",
            "vl_embed_retrieval",
            "EmbeddingPlugin",
        ),
        "encoder_features.py": (
            "encoder_base_features",
            "EncoderFeaturesPlugin",
        ),
    }
    owned_reference_families = {
        "sentence_transformer_embed",
        "bge_retrieval_embed",
        "vl_embed_retrieval",
        "encoder_base_features",
    }
    violations = []

    for filename, forbidden_terms in shared_contracts.items():
        path = E2E_SHARED_CONTRACT_PLUGINS / filename
        text = path.read_text(encoding="utf-8")
        if "plugin = None" not in text:
            violations.append((path, 0, f"shared {filename} plugin should be inert"))
        violations.extend(
            (path, 0, f"shared {filename} should not own {term}")
            for term in forbidden_terms
            if term in text
        )

    for path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reference_family = manifest.get("reference_family")
        if reference_family not in owned_reference_families:
            continue

        contract = path.parents[1] / "e2e_plugins" / "contract.py"
        if not contract.exists():
            violations.append(
                (
                    contract,
                    0,
                    f"missing model-owned contract for {reference_family}",
                )
            )
            continue

        contract_text = contract.read_text(encoding="utf-8")
        if reference_family not in contract_text:
            violations.append(
                (
                    contract,
                    0,
                    f"model contract missing {reference_family}",
                )
            )

    assert not violations, _format_violations(violations)


def test_migrated_contract_plugins_are_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete contract comparison behavior in model-owned modules.
    Preconditions: manifests declare migrated reference families.
    Postconditions: shared contract plugin modules are inert placeholders.
    """
    shared_contracts = {
        "causal_continuation.py": (
            "causal_base_continuation",
            "code_base_completion",
            "seq2seq_base_weak",
            "CausalContinuationPlugin",
        ),
        "chat_instruct.py": (
            "chat_instruct_template",
            "ChatInstructPlugin",
        ),
        "reranking.py": (
            "vl_rerank",
            "RerankingPlugin",
        ),
        "segmentation.py": (
            "semantic_segmentation",
            "SegmentationPlugin",
        ),
        "time_series_classification.py": (
            "time_series_classification",
            "TimeSeriesClassificationPlugin",
        ),
        "time_series_point_forecast.py": (
            "time_series_point_forecast",
            "TimeSeriesPointForecastPlugin",
        ),
        "time_series_quantile_forecast.py": (
            "time_series_quantile_forecast",
            "TimeSeriesQuantileForecastPlugin",
        ),
        "time_series_regression.py": (
            "time_series_regression",
            "TimeSeriesRegressionPlugin",
        ),
        "translation.py": (
            "translation_chat_template",
            "seq2seq_text2text",
            "seq2seq_translation",
            "TranslationPlugin",
        ),
        "vl_qa.py": (
            "vl_instruct_qa",
            "ocr_markdown",
            "VLQAPlugin",
        ),
    }
    migrated_reference_families = {
        "causal_base_continuation",
        "code_base_completion",
        "seq2seq_base_weak",
        "chat_instruct_template",
        "vl_rerank",
        "semantic_segmentation",
        "time_series_point_forecast",
        "time_series_quantile_forecast",
        "time_series_regression",
        "translation_chat_template",
        "seq2seq_text2text",
        "seq2seq_translation",
        "vl_instruct_qa",
        "ocr_markdown",
    }
    violations = []

    for filename, forbidden_terms in shared_contracts.items():
        path = E2E_SHARED_CONTRACT_PLUGINS / filename
        text = path.read_text(encoding="utf-8")
        if "plugin = None" not in text:
            violations.append((path, 0, f"shared {filename} plugin should be inert"))
        violations.extend(
            (path, 0, f"shared {filename} should not own {term}")
            for term in forbidden_terms
            if term in text
        )

    helper_path = E2E_SHARED_CONTRACT_PLUGINS / "_time_series_helpers.py"
    helper_text = helper_path.read_text(encoding="utf-8")
    violations.extend(
        (helper_path, 0, f"shared time-series helper should not own {term}")
        for term in (
            "reshape_trt_like_reference",
            "relative_l2",
            "max_pointwise_error",
            "finite_metric",
        )
        if term in helper_text
    )
    comparator_helper = REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "_helpers.py"
    comparator_helper_text = comparator_helper.read_text(encoding="utf-8")
    violations.extend(
        (comparator_helper, 0, f"shared comparator helper should not own {term}")
        for term in (
            "cosine_similarity",
            "levenshtein_distance",
            "normalized_edit_distance",
        )
        if term in comparator_helper_text
    )
    base_path = E2E_SHARED_CONTRACT_PLUGINS / "base.py"
    base_text = base_path.read_text(encoding="utf-8")
    violations.extend(
        (base_path, 0, f"shared contract protocol should not own helper {term}")
        for term in (
            "def contract_config(",
            "def normalize_text(",
            "def strip_prompt_echo(",
            "def strip_chat_markup(",
            "def extract_answer(",
            "def levenshtein_ned(",
            "def make_pass(",
            "def make_fail(",
            "def make_skip(",
            "def make_error(",
            "_CHAT_ROLE_PREFIXES",
            "_CHAT_TURN_MARKERS",
        )
        if term in base_text
    )

    for contract in sorted(E2E_MODELS.glob("*/e2e_plugins/contract.py")):
        text = contract.read_text(encoding="utf-8")
        if "tests.e2e_harness.plugins.base" in text:
            violations.append(
                (
                    contract,
                    0,
                    "model-owned contract imports shared contract helper behavior",
                )
            )

    for path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reference_family = manifest.get("reference_family")
        if reference_family not in migrated_reference_families:
            continue

        contract = path.parents[1] / "e2e_plugins" / "contract.py"
        if not contract.exists():
            violations.append(
                (
                    contract,
                    0,
                    f"missing model-owned contract for {reference_family}",
                )
            )
            continue

        contract_text = contract.read_text(encoding="utf-8")
        if reference_family not in contract_text:
            violations.append(
                (
                    contract,
                    0,
                    f"model contract missing {reference_family}",
                )
            )

    assert not violations, _format_violations(violations)


def test_reference_contract_config_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep reference backend knobs in model-owned manifests.
    Preconditions: manifests may declare metadata.contract_config.
    Postconditions: shared contract plugins can pass through owned config.
    """
    required_config_keys = {
        "chat_instruct_template": ("use_chat_template", "enable_thinking"),
        "diffusers_image_gen": ("use_diffusers",),
        "diffusers_video_gen": ("use_diffusers", "video_mode"),
        "encoder_base_features": ("auto_class",),
        "ocr_markdown": ("use_processor", "use_chat_template", "ocr_mode"),
        "sentence_transformer_embed": ("use_sentence_transformers",),
        "seq2seq_base_weak": ("preserve_prompt_echo", "seq2seq_reconstruction"),
        "seq2seq_text2text": ("auto_class",),
        "seq2seq_translation": ("auto_class",),
        "translation_chat_template": ("use_chat_template",),
        "vl_instruct_qa": ("use_processor", "use_chat_template"),
    }
    violations = []
    for path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_keys = required_config_keys.get(manifest.get("reference_family"))
        if not expected_keys:
            continue
        metadata = manifest.get("metadata", {})
        config = metadata.get("contract_config", {}) if isinstance(metadata, dict) else {}
        missing = [
            key for key in expected_keys if not isinstance(config, dict) or key not in config
        ]
        violations.extend(
            (path, 0, f"manifest missing metadata.contract_config.{key}") for key in missing
        )

    assert not violations, _format_violations(violations)


def test_every_reference_family_manifest_has_model_owned_contract() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent comparator-only fallbacks from hiding missing model contracts.
    Preconditions: manifests may declare a reference_family/user_contract.
    Postconditions: that family owns an e2e_plugins/contract.py mentioning it.
    """
    violations = []
    for path in sorted(E2E_MODELS.glob("*/manifests/*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        reference_family = manifest.get("reference_family")
        if not reference_family:
            continue

        contract = path.parents[1] / "e2e_plugins" / "contract.py"
        if not contract.exists():
            violations.append((contract, 0, f"missing model-owned contract for {reference_family}"))
            continue

        contract_text = contract.read_text(encoding="utf-8")
        if reference_family not in contract_text:
            violations.append((contract, 0, f"model contract missing {reference_family}"))
        if "tests.e2e_harness.plugins.base" in contract_text:
            violations.append(
                (contract, 0, "model contract imports shared contract helper behavior")
            )

    assert not violations, _format_violations(violations)


def test_model_owned_python_does_not_import_sibling_models() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-owned Python code from coupling to sibling families.
    Preconditions: model code lives under tests/e2e/models or families.
    Postconditions: imports may target the same model family, not another one.
    """
    roots = (
        (E2E_MODELS, "tests.e2e.models."),
        (FAMILIES, "tensorrt_model_connect.families."),
    )
    violations = []
    for root, prefix in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                owner = path.relative_to(root).parts[0]
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (IndexError, SyntaxError, UnicodeDecodeError) as exc:
                violations.append((path, 0, f"could not inspect imports: {exc}"))
                continue

            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]

                for module in modules:
                    if not module.startswith(prefix):
                        continue
                    imported_owner = module[len(prefix) :].split(".", 1)[0]
                    if imported_owner and imported_owner != owner:
                        violations.append(
                            (
                                path,
                                getattr(node, "lineno", 0),
                                f"imports sibling model module {module}",
                            )
                        )

    assert not violations, _format_violations(violations)


def test_e2e_models_root_has_no_shared_python_helpers() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared E2E harness helpers out of model-owned namespace.
    Preconditions: tests/e2e/models contains model-family directories.
    Postconditions: root Python helpers live under tests/e2e_harness instead.
    """
    violations = [
        (path, 0, "shared E2E Python helpers belong under tests/e2e_harness")
        for path in sorted(E2E_MODELS.glob("*.py"))
        if path.name != "__init__.py"
    ]

    assert not violations, _format_violations(violations)


def test_model_owned_unit_tests_do_not_live_under_tests_tools() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete model unit tests with their owning model.
    Preconditions: model-owned E2E plugin folders may carry focused unit tests.
    Postconditions: root tests/tools has no single-model behavior tests.
    """
    violations = []
    moved_tests = {
        REPO_ROOT / "tests" / "tools" / "test_nemotron_labs_diffusion_plugin.py": (
            E2E_MODELS / "nemotron_labs_diffusion" / "test_nemotron_labs_diffusion_contract.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_qwen3_omni_hidden_state_flow.py": (
            E2E_MODELS / "qwen3_omni" / "test_hidden_state_flow.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_elf_diffusion_text_contract.py": (
            E2E_MODELS / "elf_flow" / "test_elf_flow_contract.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_riva_translate_model_card_contract.py": (
            E2E_MODELS / "mistral" / "test_riva_translate_model_card_contract.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_prompted_segmentation_harness.py": (
            E2E_MODELS / "sam" / "test_sam_prompted_segmentation_harness.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_diffusion_runner_generic.py": (
            E2E_MODELS / "flux" / "test_flux_diffusion_runner_generic.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_diffusion_quality_gates.py": (
            E2E_MODELS / "flux" / "test_flux_diffusion_quality_gates.py"
        ),
        REPO_ROOT / "tests" / "tools" / "test_vl_qa_plugin.py": (
            E2E_MODELS / "qwen_vl" / "test_vl_qa_contract.py"
        ),
    }
    for old_root_test, model_test in moved_tests.items():
        if old_root_test.exists():
            violations.append((old_root_test, 0, "model-owned unit test lives in tests/tools"))
        if not model_test.is_file():
            violations.append((model_test, 0, "missing model-owned unit test"))

    sam3_prompted_harness = E2E_MODELS / "sam3" / "test_sam3_prompted_segmentation_harness.py"
    if not sam3_prompted_harness.is_file():
        violations.append((sam3_prompted_harness, 0, "missing model-owned unit test"))
    deepseek_ocr_contract = E2E_MODELS / "deepseek_ocr" / "test_ocr_contract.py"
    if not deepseek_ocr_contract.is_file():
        violations.append((deepseek_ocr_contract, 0, "missing model-owned unit test"))

    assert not violations, _format_violations(violations)


def test_root_e2e_runner_cli_alignment_has_no_model_owned_cases() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete model runner CLI checks with the owning model.
    Preconditions: shared runner sidecars are inert placeholders.
    Postconditions: model-owned runner cases live under tests/e2e/models.
    """
    root_test = REPO_ROOT / "tests" / "tools" / "test_e2e_runner_cli_alignment.py"
    forbidden = (
        "tests.e2e.models.",
        "audio_magpie",
        "audio_bark",
        "prompted_segmentation_sam3",
        "SamPromptedSegmentationRunner",
        "Sam3PromptedSegmentationRunner",
    )
    expected_model_tests = (
        E2E_MODELS / "magpie_tts" / "test_magpie_runner_cli_alignment.py",
        E2E_MODELS / "bark" / "test_bark_runner_cli_alignment.py",
        E2E_MODELS / "sam" / "test_sam_runner_cli_alignment.py",
        E2E_MODELS / "sam3" / "test_sam3_runner_cli_alignment.py",
        E2E_MODELS / "eagle_vlm" / "test_eagle_vlm_embedding_runner.py",
        E2E_MODELS / "chronos_bolt" / "test_chronos_bolt_neural_operator_runner.py",
        E2E_MODELS / "qwen3_omni" / "test_qwen3_omni_runner.py",
        E2E_MODELS / "locateanything" / "test_locateanything_object_detection_runner.py",
    )
    violations = []
    if root_test.exists():
        text = root_test.read_text(encoding="utf-8")
        violations.append((root_test, 0, "root CLI alignment test should be model-owned"))
        violations.extend(
            (root_test, 0, f"root CLI alignment test contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )
    violations.extend(
        (path, 0, "missing model-owned CLI alignment test")
        for path in expected_model_tests
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_root_hf_transformers_helper_tests_have_no_model_owned_references() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific HF reference checks with owning model tests.
    Preconditions: shared HF Transformers reference is an inert placeholder.
    Postconditions: HF reference behavior tests live under tests/e2e/models.
    """
    root_tests = (
        REPO_ROOT / "tests" / "tools" / "test_hf_transformers_reference_helpers.py",
        REPO_ROOT / "tests" / "tools" / "test_hf_transformers_vl_reference.py",
    )
    forbidden = (
        "tests.e2e.models.",
        "qwen_vl_hf_transformers",
        "internvl_hf_transformers",
        "locateanything_hf_transformers",
        "nemotron_hf_transformers",
        "sam3_reference",
        "sam3_hf_base",
        "LocateAnything",
        "Nemotron-Labs-Diffusion",
        "prompted_segmentation_sam3",
        "<IMG_CONTEXT>",
        "<|image_pad|>",
    )
    expected_model_tests = (
        E2E_MODELS / "qwen_vl" / "test_qwen_vl_hf_transformers_reference.py",
        E2E_MODELS / "internvl" / "test_internvl_hf_transformers_reference.py",
        E2E_MODELS / "locateanything" / "test_locateanything_hf_transformers_reference.py",
        E2E_MODELS / "sam3" / "test_sam3_hf_transformers_reference.py",
        E2E_MODELS
        / "nemotron_labs_diffusion"
        / "test_nemotron_labs_diffusion_hf_transformers_reference.py",
    )
    violations = []
    for root_test in root_tests:
        if not root_test.is_file():
            continue
        text = root_test.read_text(encoding="utf-8")
        violations.extend(
            (root_test, 0, f"root HF helper test contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )
        violations.append((root_test, 0, "root HF helper test should be model-owned"))
    violations.extend(
        (path, 0, "missing model-owned HF reference test")
        for path in expected_model_tests
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_audio_model_plugins_do_not_name_sibling_audio_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Bark and Magpie test behavior in their own model plugin folders.
    Preconditions: Bark and Magpie model-owned E2E plugin files are present.
    Postconditions: neither plugin folder carries the other family's behavior.
    """
    sibling_forbidden = {
        "bark": (
            "magpie",
            "Magpie",
            "MagpieTTS",
            "audio_magpie",
            "text_to_audio_magpie",
            "nemo_magpie",
        ),
        "magpie_tts": (
            "bark",
            "Bark",
            "audio_bark",
            "bark_dump",
            "sem_tokens",
            "coarse_tokens",
            "hf_bark",
        ),
    }
    violations = []
    for family, forbidden in sibling_forbidden.items():
        model_dir = E2E_MODELS / family
        for path in sorted(model_dir.rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            violations.extend(
                (path, 0, f"{family} E2E plugin contains sibling audio behavior {needle}")
                for needle in forbidden
                if needle in text
            )

    assert not violations, _format_violations(violations)


def test_shared_torch_reference_has_no_model_owned_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared torch_reference.py limited to generic golden snapshots.
    Preconditions: model-specific torch references live under model E2E plugin folders.
    Postconditions: shared torch_reference.py names no PersonaPlex or time-series behavior.
    """
    forbidden = (
        "PersonaPlex",
        "personaplex",
        "speech_reference_tokens",
        "speech_to_speech",
        "speech_to_text",
        "Whisper",
        "WhisperProcessor",
        "PatchTST",
        "patchtst",
        "PatchTSMixer",
        "patchtsmixer",
        "TimesFM",
        "TimesFm",
        "timesfm",
        "Chronos",
        "chronos",
        "chronos_bolt",
        "_run_time_series",
    )
    text = E2E_SHARED_TORCH_REFERENCE.read_text(encoding="utf-8")
    violations = [
        (E2E_SHARED_TORCH_REFERENCE, 0, f"shared torch reference contains {needle}")
        for needle in forbidden
        if needle in text
    ]

    assert not violations, _format_violations(violations)


def test_torch_reference_model_plugins_do_not_name_sibling_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep torch-reference behavior owned by the one family that uses it.
    Preconditions: PersonaPlex and time-series model folders own torch_reference.py.
    Postconditions: no torch owner carries sibling family reference behavior.
    """
    family_terms = {
        "personaplex": (
            "PersonaPlex",
            "personaplex",
            "speech_reference_tokens",
            "speech_to_speech",
        ),
        "patchtst": (
            "PatchTST",
            "patchtst",
        ),
        "patchtsmixer": (
            "PatchTSMixer",
            "patchtsmixer",
        ),
        "timesfm": (
            "TimesFM",
            "TimesFm",
            "timesfm",
        ),
        "chronos_bolt": (
            "Chronos",
            "chronos",
            "chronos_bolt",
        ),
    }
    violations = []
    for owner in sorted(family_terms):
        path = E2E_MODELS / owner / "e2e_plugins" / "references" / "torch_reference.py"
        text = path.read_text(encoding="utf-8", errors="ignore")
        sibling_terms = tuple(
            term for family, terms in family_terms.items() if family != owner for term in terms
        )
        violations.extend(
            (path, 0, f"{owner} torch reference contains sibling behavior {needle}")
            for needle in sibling_terms
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_non_torch_reference_model_plugins_do_not_carry_torch_behavior() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent generated unused torch_reference.py sidecars from carrying behavior.
    Preconditions: only PersonaPlex and time-series folders own torch reference behavior.
    Postconditions: every non-owner torch_reference.py is an inert placeholder.
    """
    torch_reference_owners = {
        "personaplex",
        "patchtst",
        "patchtsmixer",
        "timesfm",
        "chronos_bolt",
    }
    forbidden = (
        "PersonaPlex",
        "speech_reference_tokens",
        "speech_to_speech",
        "speech_to_text",
        "WhisperProcessor",
        "WhisperForConditionalGeneration",
        "PatchTST",
        "patchtst",
        "PatchTSMixer",
        "patchtsmixer",
        "TimesFM",
        "TimesFm",
        "timesfm",
        "Chronos",
        "chronos",
        "chronos_bolt",
        "_run_time_series",
        "_run_forward",
    )
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/references/torch_reference.py")):
        family = path.parts[-4]
        if family in torch_reference_owners:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"non-owner torch reference contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_magpie_nemo_reference_lives_only_in_magpie_model_plugins() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Magpie NeMo reference behavior in the Magpie E2E plugin only.
    Preconditions: model-owned E2E reference modules are generated per family.
    Postconditions: non-Magpie families do not carry copied Magpie NeMo code.
    """
    forbidden = (
        "magpie",
        "Magpie",
        "MagpieTTS",
        "audio_magpie",
        "text_to_audio_magpie",
        "nemo_magpie",
    )
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/references/nemo_reference.py")):
        if path.parts[-4] == "magpie_tts":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"non-Magpie NeMo reference contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_speech_to_text_hf_reference_lives_only_in_asr_model_plugins() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep speech-to-text HF reference behavior in ASR model folders.
    Preconditions: generated HF reference sidecars exist across model folders.
    Postconditions: non-ASR families do not carry copied speech-to-text
    reference dispatch or AutoModelForSpeechSeq2Seq code.
    """
    owners = {"whisper", "canary", "nemotron_speech_streaming"}
    forbidden = (
        "_run_speech_to_text_ref",
        'task == "speech_to_text"',
        "AutoModelForSpeechSeq2Seq",
        "hf_speech_to_text",
    )
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/references/hf_transformers.py")):
        if path.parts[-4] in owners:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"non-ASR HF reference contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_bark_text_to_audio_behavior_lives_only_in_bark_model_plugins() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep Bark runner/reference/comparator behavior in the Bark E2E plugin.
    Preconditions: model-owned E2E plugin files are generated per family.
    Postconditions: non-Bark families do not carry copied Bark text-to-audio code.
    """
    forbidden = (
        "bark",
        "Bark",
        "audio_bark",
        "bark_dump",
        "sem_tokens",
        "coarse_tokens",
        "hf_bark",
    )
    relative_files = (
        "e2e_plugins/runners/audio_speech.py",
        "e2e_plugins/comparators/text_to_audio.py",
        "e2e_plugins/references/torch_reference.py",
        "e2e_plugins/references/hf_transformers.py",
    )
    violations = []
    for model_dir in sorted(E2E_MODELS.iterdir()):
        if model_dir.name == "bark" or not model_dir.is_dir():
            continue
        for rel in relative_files:
            path = model_dir / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            violations.extend(
                (path, 0, f"non-Bark E2E plugin contains {needle}")
                for needle in forbidden
                if needle in text
            )

    assert not violations, _format_violations(violations)


def test_text_to_audio_comparator_sidecars_are_owned_by_audio_models() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep text-to-audio comparator implementations in audio model folders.
    Preconditions: generated comparator sidecars exist across model folders.
    Postconditions: only Bark and Magpie sidecars implement text-to-audio compare behavior.
    """
    owners = {"bark", "magpie_tts"}
    behavior_terms = (
        "class TextToAudioComparator",
        'return "text_to_audio"',
        "audio_duration",
        "waveform",
        "sample_rate",
    )
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/comparators/text_to_audio.py")):
        family = path.parts[-4]
        if family in owners:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "plugin = None" not in text:
            violations.append((path, 0, "non-audio text-to-audio comparator is not inert"))
        violations.extend(
            (path, 0, f"non-audio text-to-audio comparator contains {needle}")
            for needle in behavior_terms
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_audio_speech_runner_sidecars_are_task_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent copied audio/speech E2E runners from registering sibling
    model-family behavior.
    Preconditions: generated audio_speech.py sidecars exist across families.
    Postconditions: only the owning families implement their task-specific
    runner, and no owner file registers extra sibling runners.
    """
    owners = {
        "bark": ("TextToAudioRunner", "text_to_audio"),
        "magpie_tts": ("TextToAudioRunner", "text_to_audio"),
        "whisper": ("SpeechToTextRunner", "whisper_speech_to_text"),
        "canary": ("SpeechToTextRunner", "canary_speech_to_text"),
        "nemotron_speech_streaming": (
            "SpeechToTextRunner",
            NEMOTRON_SPEECH_STREAMING_RUNTIME_STRATEGY,
        ),
        "personaplex": ("SpeechToSpeechRunner", PERSONAPLEX_RUNTIME_STRATEGY),
    }
    runner_classes = {"SpeechToTextRunner", "TextToAudioRunner", "SpeechToSpeechRunner"}
    strategy_returns = {
        "speech_to_text",
        "whisper_speech_to_text",
        "canary_speech_to_text",
        NEMOTRON_SPEECH_STREAMING_RUNTIME_STRATEGY,
        "text_to_audio",
        "speech_to_speech",
        PERSONAPLEX_RUNTIME_STRATEGY,
    }
    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/audio_speech.py")):
        family = path.parts[-4]
        text = path.read_text(encoding="utf-8", errors="ignore")
        if family not in owners:
            if "plugin = None" not in text:
                violations.append((path, 0, "non-owner audio/speech runner is not inert"))
            for class_name in runner_classes:
                if f"class {class_name}" in text:
                    violations.append((path, 0, f"non-owner runner defines {class_name}"))
            for strategy in strategy_returns:
                if f'return "{strategy}"' in text:
                    violations.append((path, 0, f"non-owner runner returns {strategy}"))
            continue

        owned_class, owned_strategy = owners[family]
        if f"class {owned_class}" not in text:
            violations.append((path, 0, f"owner missing {owned_class}"))
        if f'return "{owned_strategy}"' not in text:
            violations.append((path, 0, f"owner missing {owned_strategy}"))
        if f"plugin = {owned_class}()" not in text:
            violations.append((path, 0, f"owner does not register {owned_class}"))
        if "register_runner(" in text:
            violations.append((path, 0, "owner runner registers sibling runners"))
        for class_name in sorted(runner_classes - {owned_class}):
            if f"class {class_name}" in text:
                violations.append((path, 0, f"owner runner defines sibling {class_name}"))
        for strategy in sorted(strategy_returns - {owned_strategy}):
            if f'return "{strategy}"' in text:
                violations.append((path, 0, f"owner runner returns sibling {strategy}"))

    assert not violations, _format_violations(violations)


def test_audio_speech_comparator_sidecars_are_task_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent generated audio comparator umbrellas and copied comparator
    implementations from registering sibling model-family behavior.
    Preconditions: generated audio comparator sidecars exist across families.
    Postconditions: task-specific comparator modules are active only for their
    owning families, and audio.py remains an inert umbrella placeholder.
    """
    task_sidecars = {
        "text_to_audio.py": {
            "owners": {"bark", "magpie_tts"},
            "class": "TextToAudioComparator",
            "plugin": "plugin = TextToAudioComparator()",
        },
        "speech_to_text.py": {
            "owners": {"whisper", "canary", "nemotron_speech_streaming"},
            "class": "SpeechToTextComparator",
            "plugin": "plugin = SpeechToTextComparator()",
        },
        "speech_to_speech.py": {
            "owners": {"personaplex"},
            "class": "SpeechToSpeechComparator",
            "plugin": "plugin = SpeechToSpeechComparator()",
        },
    }
    violations = []
    for filename, spec in task_sidecars.items():
        owners = spec["owners"]
        class_name = spec["class"]
        plugin_line = spec["plugin"]
        for path in sorted(E2E_MODELS.glob(f"*/e2e_plugins/comparators/{filename}")):
            family = path.parts[-4]
            text = path.read_text(encoding="utf-8", errors="ignore")
            if family in owners:
                if f"class {class_name}" not in text:
                    violations.append((path, 0, f"owner missing {class_name}"))
                if plugin_line not in text:
                    violations.append((path, 0, f"owner missing {plugin_line}"))
            else:
                if "plugin = None" not in text:
                    violations.append((path, 0, f"non-owner {filename} is not inert"))
                if f"class {class_name}" in text:
                    violations.append((path, 0, f"non-owner {filename} defines {class_name}"))

    forbidden_umbrella_terms = (
        "SpeechToTextComparator",
        "TextToAudioComparator",
        "SpeechToSpeechComparator",
        "register_comparator(",
    )
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/comparators/audio.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "plugin = None" not in text:
            violations.append((path, 0, "audio umbrella comparator is not inert"))
        for term in forbidden_umbrella_terms:
            if term in text:
                violations.append((path, 0, f"audio umbrella contains {term}"))

    assert not violations, _format_violations(violations)


def test_generated_e2e_task_sidecars_are_task_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent generated E2E sidecars from carrying active behavior for
    task strategies that the model family does not own.
    Preconditions: generated sidecars exist under each model family.
    Postconditions: active sidecar implementations appear only in owner
    families; non-owner copies are inert placeholders.
    """
    encoder_owners = {
        "albert",
        "bert",
        "convbert",
        "deberta",
        "distilbert",
        "dpr",
        "electra",
        "fnet",
        "modernbert",
        "mpnet",
        "roberta",
        "xlnet",
    }
    diffusion_owners = {"flux", "ltx_video", "pixart", "qwen_image", "wan_t2v", "z_image"}
    simple_sidecars = [
        (
            "runners",
            "vision_language.py",
            set(VL_RUNTIME_FAMILIES),
            "VisionLanguageRunner",
            "vision_language_generation",
        ),
        (
            "comparators",
            "vision_language.py",
            set(VL_RUNTIME_FAMILIES),
            "VisionLanguageComparator",
            "vision_language_generation",
        ),
        ("runners", "embedding.py", {"eagle_vlm"}, "EmbeddingRunner", "embedding"),
        ("comparators", "embedding.py", {"eagle_vlm"}, "EmbeddingComparator", "embedding"),
        ("runners", "encoder_only.py", encoder_owners, "EncoderOnlyRunner", "encoder_only_nlp"),
        (
            "comparators",
            "encoder_only.py",
            encoder_owners,
            "EncoderOnlyComparator",
            "encoder_only_nlp",
        ),
        ("runners", "reranking.py", {"eagle_vlm"}, "RerankingRunner", "reranking"),
        ("comparators", "reranking.py", {"eagle_vlm"}, "RerankingComparator", "reranking"),
        (
            "runners",
            "neural_operator.py",
            {"chronos_bolt", "patchtsmixer", "patchtst", "timesfm"},
            "NeuralOperatorRunner",
            "neural_operator",
        ),
        (
            "comparators",
            "neural_operator.py",
            {"chronos_bolt", "patchtsmixer", "patchtst", "timesfm"},
            "NeuralOperatorComparator",
            "neural_operator",
        ),
        (
            "runners",
            "object_detection.py",
            {"locateanything"},
            "ObjectDetectionRunner",
            "object_detection",
        ),
        (
            "runners",
            "image_classification.py",
            {"timm_vit"},
            "ImageClassificationRunner",
            "image_classification",
        ),
        (
            "comparators",
            "image_classification.py",
            {"timm_vit"},
            "ImageClassificationComparator",
            "image_classification",
        ),
        (
            "runners",
            "diffusion.py",
            diffusion_owners,
            "DiffusionMediaRunner",
            "diffusion_media_generation",
        ),
        (
            "comparators",
            "diffusion.py",
            diffusion_owners,
            "DiffusionComparator",
            "diffusion_media_generation",
        ),
        (
            "runners",
            "diffusion_text_generation.py",
            {"elf_flow"},
            "DiffusionTextGenerationRunner",
            "diffusion_text_generation",
        ),
        (
            "comparators",
            "diffusion_text_generation.py",
            {"elf_flow"},
            "DiffusionTextGenerationComparator",
            "diffusion_text_generation",
        ),
        ("runners", "omni.py", {"qwen3_omni"}, "OmniMultimodalRunner", QWEN3_OMNI_RUNTIME_STRATEGY),
        ("comparators", "omni.py", {"qwen3_omni"}, "OmniComparator", "omni_multimodal"),
    ]

    violations = []
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "register_runner(" in text:
            violations.append((path, 0, "generated runner sidecar registers extra runners"))
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/comparators/*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "register_comparator(" in text:
            violations.append((path, 0, "generated comparator sidecar registers extra comparators"))

    for kind, filename, owners, class_name, strategy in simple_sidecars:
        for path in sorted(E2E_MODELS.glob(f"*/e2e_plugins/{kind}/{filename}")):
            family = path.parts[-4]
            text = path.read_text(encoding="utf-8", errors="ignore")
            if family in owners:
                if f"class {class_name}" not in text:
                    violations.append((path, 0, f"owner missing {class_name}"))
                if f'return "{strategy}"' not in text:
                    violations.append((path, 0, f"owner missing strategy {strategy}"))
                if f"plugin = {class_name}()" not in text:
                    violations.append((path, 0, f"owner missing plugin {class_name}"))
            else:
                if "plugin = None" not in text:
                    violations.append((path, 0, "non-owner sidecar is not inert"))
                if f"class {class_name}" in text:
                    violations.append((path, 0, f"non-owner sidecar defines {class_name}"))
                if f'return "{strategy}"' in text:
                    violations.append((path, 0, f"non-owner sidecar returns {strategy}"))

    segmentation_classes = {
        "segformer": ("SegmentationComparator", "segmentation"),
        "sam": ("PromptedSegmentationComparator", "prompted_segmentation"),
        "sam3": ("PromptedSegmentationComparator", "prompted_segmentation"),
    }
    all_segmentation_classes = {
        "SegmentationComparator",
        "PromptedSegmentationComparator",
        "ObjectDetectionComparator",
    }
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/comparators/segmentation.py")):
        family = path.parts[-4]
        text = path.read_text(encoding="utf-8", errors="ignore")
        if family not in segmentation_classes:
            if "plugin = None" not in text:
                violations.append((path, 0, "non-segmentation comparator is not inert"))
            for class_name in sorted(all_segmentation_classes):
                if f"class {class_name}" in text:
                    violations.append(
                        (path, 0, f"non-owner segmentation comparator defines {class_name}")
                    )
            continue

        owned_class, strategy = segmentation_classes[family]
        if f"class {owned_class}" not in text:
            violations.append((path, 0, f"owner missing {owned_class}"))
        if f'return "{strategy}"' not in text:
            violations.append((path, 0, f"owner missing strategy {strategy}"))
        if f"plugin = {owned_class}()" not in text:
            violations.append((path, 0, f"owner missing plugin {owned_class}"))
        for class_name in sorted(all_segmentation_classes - {owned_class}):
            if f"class {class_name}" in text:
                violations.append(
                    (path, 0, f"owner segmentation comparator defines sibling {class_name}")
                )
        if "register_comparator(" in text:
            violations.append((path, 0, "segmentation comparator registers sibling classes"))

    segmentation_runner_classes = {
        "segformer": ("SegmentationRunner", "segmentation"),
        "sam": ("PromptedSegmentationRunner", "prompted_segmentation"),
        "sam3": ("PromptedSegmentationRunner", "prompted_segmentation"),
    }
    for path in sorted(E2E_MODELS.glob("*/e2e_plugins/runners/segmentation.py")):
        family = path.parts[-4]
        text = path.read_text(encoding="utf-8", errors="ignore")
        if family not in segmentation_runner_classes:
            if "plugin = None" not in text:
                violations.append((path, 0, "non-segmentation runner is not inert"))
            for class_name in ("SegmentationRunner", "PromptedSegmentationRunner"):
                if f"class {class_name}" in text:
                    violations.append(
                        (path, 0, f"non-owner segmentation runner defines {class_name}")
                    )
            continue
        owned_class, strategy = segmentation_runner_classes[family]
        if f"class {owned_class}" not in text:
            violations.append((path, 0, f"owner missing {owned_class}"))
        if f'return "{strategy}"' not in text:
            violations.append((path, 0, f"owner missing strategy {strategy}"))
        for class_name in {"SegmentationRunner", "PromptedSegmentationRunner"} - {owned_class}:
            if f"class {class_name}" in text:
                violations.append(
                    (path, 0, f"owner segmentation runner defines sibling {class_name}")
                )

    assert not violations, _format_violations(violations)


def test_family_builders_do_not_import_sibling_or_forbidden_shared_helpers() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent builder code from coupling one family to another family.
    Preconditions: python/tensorrt_model_connect/families/<model> folders exist.
    Postconditions: family builder code imports local helpers or generic APIs only.
    """
    family_ids = _family_model_ids()
    violations: list[tuple[Path, int, str]] = []

    for owner in sorted(family_ids):
        for path in (FAMILIES / owner).rglob("*.py"):
            if "tests" in path.relative_to(FAMILIES / owner).parts:
                continue
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="ignore"),
                filename=str(path),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        _check_absolute_import(
                            alias.name, owner, family_ids, path, node.lineno, violations
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level == 0:
                        if module == "tensorrt_model_connect":
                            for alias in node.names:
                                if alias.name in _FORBIDDEN_SHARED_BUILDER_MODULES:
                                    violations.append(
                                        (
                                            path,
                                            node.lineno,
                                            f"imports shared {module}.{alias.name}",
                                        )
                                    )
                        _check_absolute_import(
                            module, owner, family_ids, path, node.lineno, violations
                        )
                    else:
                        if node.level >= 2 and not module:
                            for alias in node.names:
                                if alias.name in _FORBIDDEN_SHARED_BUILDER_MODULES:
                                    violations.append(
                                        (
                                            path,
                                            node.lineno,
                                            f"imports shared helper {'.' * node.level}{alias.name}",
                                        )
                                    )
                        _check_relative_import(
                            module,
                            node.level,
                            owner,
                            family_ids,
                            path,
                            node.lineno,
                            violations,
                        )

    assert not violations, _format_violations(violations)


def _check_absolute_import(
    module: str,
    owner: str,
    family_ids: set[str],
    path: Path,
    line_no: int,
    violations: list[tuple[Path, int, str]],
) -> None:
    parts = module.split(".")
    if parts[:2] != ["tensorrt_model_connect", "families"]:
        if (
            len(parts) >= 2
            and parts[0] == "tensorrt_model_connect"
            and parts[1] in _FORBIDDEN_SHARED_BUILDER_MODULES
        ):
            violations.append((path, line_no, f"imports shared {module}"))
        return

    if len(parts) >= 3 and parts[2] in family_ids and parts[2] != owner:
        violations.append((path, line_no, f"imports sibling family {module}"))


def _check_relative_import(
    module: str,
    level: int,
    owner: str,
    family_ids: set[str],
    path: Path,
    line_no: int,
    violations: list[tuple[Path, int, str]],
) -> None:
    first = module.split(".", 1)[0] if module else ""
    if level == 2 and first in family_ids and first != owner:
        violations.append((path, line_no, f"imports sibling family ..{module}"))
    if level >= 3 and first in _FORBIDDEN_SHARED_BUILDER_MODULES:
        violations.append((path, line_no, f"imports shared helper ...{module}"))


def test_model_owned_e2e_assets_are_local_and_complete() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep each model E2E contract runnable from its own model folder.
    Preconditions: tests/e2e/models/<model>/MODEL.toml declares manifests.
    Postconditions: each model has local entrypoints, plugins, and thresholds.
    """
    violations: list[tuple[Path, int, str]] = []

    for model_dir in sorted(E2E_MODELS.iterdir()):
        if not (model_dir / "MODEL.toml").is_file():
            continue
        expected_entrypoint = model_dir / f"test_{model_dir.name}_e2e.py"
        for required in (
            model_dir / "runner.py",
            expected_entrypoint,
            model_dir / "e2e_plugins",
        ):
            if not required.exists():
                violations.append((model_dir, 0, f"missing {required.name}"))

        for manifest in sorted((model_dir / "manifests").glob("*.json")):
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            for testcase in raw.get("testcases", []):
                threshold = model_dir / "thresholds" / f"{testcase['name']}.json"
                if not threshold.is_file():
                    violations.append(
                        (
                            manifest,
                            0,
                            f"missing threshold sidecar {threshold.relative_to(REPO_ROOT)}",
                        )
                    )

        for path in model_dir.rglob("*.py"):
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                if _FORBIDDEN_E2E_IMPORT_RE.search(line):
                    violations.append(
                        (
                            path,
                            line_no,
                            "imports shared E2E runner/reference/comparator",
                        )
                    )

    assert not violations, _format_violations(violations)


def test_lazy_family_packages_preserve_plugin_instance_api() -> None:
    """Lazy package imports must not replace ``family.plugin`` with a module.

    Importlib publishes a directly imported ``family.plugin`` submodule on the
    parent package.  Every lazy family initializer must intercept that write so
    package consumers and registry discovery continue to receive the
    model-owned ``FamilyPlugin`` instance, independent of test import order.
    """
    violations: list[tuple[Path, int, str]] = []
    guard = 'if name == "plugin" and isinstance(value, types.ModuleType):'

    for init_path in sorted(FAMILIES.glob("*/__init__.py")):
        source = init_path.read_text(encoding="utf-8")
        if "_plugin = None" not in source:
            continue
        if guard not in source:
            violations.append(
                (init_path, 0, "lazy package does not preserve plugin instance")
            )

    assert not violations, _format_violations(violations)
