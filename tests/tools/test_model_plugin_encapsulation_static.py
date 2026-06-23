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
PY_RUNTIME_CONFIG_SCHEMAS = REPO_ROOT / "python" / "tensorrt_model_connect" / "runtime_config" / "schemas"
BUNDLE_WRITER = REPO_ROOT / "python" / "tensorrt_model_connect" / "bundle_writer.py"
CONFIG_PY = REPO_ROOT / "python" / "tensorrt_model_connect" / "config.py"
PYTHON_PROFILES = REPO_ROOT / "python" / "tensorrt_model_connect" / "python_profiles.toml"
ENGINE_BUILDER = REPO_ROOT / "python" / "tensorrt_model_connect" / "engine_builder.py"
BUILD_CLI = REPO_ROOT / "python" / "tensorrt_model_connect" / "build_cli.py"
DEBUG_RUNNER = REPO_ROOT / "python" / "tensorrt_model_connect" / "debug_runner.py"
DEBUG_RUNNER_TEST = REPO_ROOT / "tests" / "builder" / "test_debug_runner.py"
DEBUG_RUNNER_EXTENDED_TEST = (
    REPO_ROOT / "tests" / "builder" / "test_debug_runner_extended.py"
)
SHARED_MANIFEST_VALIDATION_TEST = REPO_ROOT / "tests" / "builder" / "test_manifest_validation.py"
FP8_CALIBRATE = REPO_ROOT / "python" / "tensorrt_model_connect" / "fp8_calibrate.py"
SHARED_GENERIC_HELPER_FILES = (
    REPO_ROOT / "python" / "tensorrt_model_connect" / "graph_ops.py",
    DEBUG_RUNNER,
    REPO_ROOT / "python" / "tensorrt_model_connect" / "triattention_export.py",
    REPO_ROOT / "python" / "tensorrt_model_connect" / "schedulers" / "flow_match_euler.py",
    BUILD_CLI,
    REPO_ROOT / "tests" / "builder" / "test_graph_ops_extended.py",
)
SHARED_DECODER_BUILDER_FILES = (
    REPO_ROOT / "python" / "tensorrt_model_connect" / "builders" / "default_decoder.py",
    REPO_ROOT / "python" / "tensorrt_model_connect" / "builders" / "default_dual_profile_decoder.py",
    REPO_ROOT / "python" / "tensorrt_model_connect" / "builders" / "default_dual_profile_decoder_tp.py",
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
RUNTIME_RECURRENT_CONTRACTS = RUNTIME_DOMAINS / "recurrent" / "recurrent_step_contracts.h"
RUNTIME_AUDIO_DOMAIN_FILES = (
    RUNTIME_DOMAINS / "audio" / "mel_spectrogram.h",
    RUNTIME_DOMAINS / "audio" / "mel_spectrogram.cpp",
)
RUNTIME_MULTIMODAL_PREPROCESSOR_FILES = (
    RUNTIME_DOMAINS / "multimodal" / "image_preprocessor.h",
    RUNTIME_DOMAINS / "multimodal" / "image_preprocessor.cpp",
    RUNTIME_DOMAIN_INCLUDES / "multimodal" / "image_transform_helper.h",
)
PIPELINE_FACTORY = REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp"
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
SHARED_RUNTIME_LEAK_FILES = (
    RUNTIME_DIFFUSION_DOMAINS / "diffusion_types.h",
    REPO_ROOT / "src" / "runtime" / "core" / "flow_match_euler_scheduler.cpp",
    REPO_ROOT / "include" / "trtmc" / "runtime" / "device_ops.h",
    REPO_ROOT / "include" / "trtmc" / "runtime" / "hybrid_state.h",
)
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
E2E_SHARED_HF_TRANSFORMERS = REPO_ROOT / "tests" / "e2e_harness" / "references" / "hf_transformers.py"
E2E_SHARED_TORCH_REFERENCE = REPO_ROOT / "tests" / "e2e_harness" / "references" / "torch_reference.py"
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
E2E_SHARED_PROMPTED_SEGMENTATION_REFERENCE_FILES = (
    E2E_SHARED_HF_TRANSFORMERS,
)
E2E_SHARED_PROMPTED_SEGMENTATION_RUNTIME_FILES = (
    E2E_ORCHESTRATOR,
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "segmentation.py",
    REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "segmentation.py",
)
E2E_SHARED_DOC_FILES = (
    E2E_CONTRACTS,
    REPO_ROOT / "tests" / "e2e_harness" / "runners" / "vision_language.py",
)
TEST_IMPACT = REPO_ROOT / "tools" / "test_impact.py"
SHARED_DIFF_VL_TOOL = REPO_ROOT / "tools" / "diff_vl.py"
SHARED_DIFF_LOGITS_TOOL = REPO_ROOT / "tools" / "diff_logits.py"
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
    REPO_ROOT / "tests" / "builder" / "test_checkpoint_mapper.py",
    REPO_ROOT / "tests" / "builder" / "test_checkpoint_mapper_coverage.py",
    REPO_ROOT / "tests" / "builder" / "test_quantization.py",
    REPO_ROOT / "tests" / "builder" / "test_parallel_config.py",
    REPO_ROOT / "tests" / "builder" / "test_trt_compat_boundary.py",
    REPO_ROOT / "tests" / "builder" / "test_triattention_export.py",
)
ROOT_MODEL_SCRIPT_WRAPPERS = {
    REPO_ROOT / "scripts" / "magpie_tokenizer.py": (
        FAMILIES / "magpie_tts" / "magpie_tokenizer.py"
    ),
    REPO_ROOT / "scripts" / "magpie_codec_bridge.py": (
        FAMILIES / "magpie_tts" / "codec_bridge.py"
    ),
    REPO_ROOT / "scripts" / "profile_magpie_tts.py": (
        FAMILIES / "magpie_tts" / "profile.py"
    ),
    REPO_ROOT / "scripts" / "prepare_lance_model.py": (
        FAMILIES / "lance" / "prepare_model.py"
    ),
    REPO_ROOT / "scripts" / "_build_fp8_onnx_monolithic.py": (
        FAMILIES / "flux" / "build_fp8_onnx_monolithic.py"
    ),
    REPO_ROOT / "scripts" / "_inject_fp8_qdq_proto.py": (
        FAMILIES / "flux" / "inject_fp8_qdq_proto.py"
    ),
    REPO_ROOT / "scripts" / "_mk_fp8_bf16_bundle.py": (
        FAMILIES / "flux" / "mk_fp8_bf16_bundle.py"
    ),
    REPO_ROOT / "tools" / "diff_personaplex.py": (
        FAMILIES / "personaplex" / "diff_personaplex.py"
    ),
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
        E2E_MODELS
        / "nemotron_speech_streaming"
        / "test_nemotron_speech_streaming_builder.py"
    ),
    REPO_ROOT / "tests" / "builder" / "test_engine_nemotron_speech_streaming_tp.py": (
        E2E_MODELS
        / "nemotron_speech_streaming"
        / "test_nemotron_speech_streaming_builder_tp.py"
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
MODEL_OWNED_BUILDER_TESTS.update({
    REPO_ROOT / "tests" / "builder" / "test_engine_internvl_tp.py": E2E_MODELS / "internvl" / "test_internvl_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_xlnet_tp.py": E2E_MODELS / "xlnet" / "test_xlnet_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_t5_tp.py": E2E_MODELS / "t5" / "test_t5_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_segformer_tp.py": E2E_MODELS / "segformer" / "test_segformer_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_sam_tp.py": E2E_MODELS / "sam" / "test_sam_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_rwkv_tp.py": E2E_MODELS / "rwkv" / "test_rwkv_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_roberta_tp.py": E2E_MODELS / "roberta" / "test_roberta_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_phi_moe_tp.py": E2E_MODELS / "phi_moe" / "test_phi_moe_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_personaplex_tp.py": E2E_MODELS / "personaplex" / "test_personaplex_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_opt_tp.py": E2E_MODELS / "opt" / "test_opt_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_olmo_tp.py": E2E_MODELS / "olmo" / "test_olmo_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_olmo2_tp.py": E2E_MODELS / "olmo2" / "test_olmo2_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_mpnet_tp.py": E2E_MODELS / "mpnet" / "test_mpnet_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_modernbert_tp.py": E2E_MODELS / "modernbert" / "test_modernbert_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_mixtral_tp.py": E2E_MODELS / "mixtral" / "test_mixtral_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_marian_tp.py": E2E_MODELS / "marian" / "test_marian_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_mamba_tp.py": E2E_MODELS / "mamba" / "test_mamba_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_granite_tp.py": E2E_MODELS / "granite" / "test_granite_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_gpt_oss_tp.py": E2E_MODELS / "gpt_oss" / "test_gpt_oss_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_gemma_tp.py": E2E_MODELS / "gemma" / "test_gemma_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_fnet_tp.py": E2E_MODELS / "fnet" / "test_fnet_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_falcon_tp.py": E2E_MODELS / "falcon" / "test_falcon_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_electra_tp.py": E2E_MODELS / "electra" / "test_electra_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_eagle_vlm_tp.py": E2E_MODELS / "eagle_vlm" / "test_eagle_vlm_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_dpr_tp.py": E2E_MODELS / "dpr" / "test_dpr_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_distilbert_tp.py": E2E_MODELS / "distilbert" / "test_distilbert_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_deepseek_v2_tp.py": E2E_MODELS / "deepseek_v2" / "test_deepseek_v2_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_deepseek_ocr_tp.py": E2E_MODELS / "deepseek_ocr" / "test_deepseek_ocr_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_deberta_tp.py": E2E_MODELS / "deberta" / "test_deberta_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_convbert_tp.py": E2E_MODELS / "convbert" / "test_convbert_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_bloom_tp.py": E2E_MODELS / "bloom" / "test_bloom_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_bert_tp.py": E2E_MODELS / "bert" / "test_bert_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_bart_tp.py": E2E_MODELS / "bart" / "test_bart_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_albert_tp.py": E2E_MODELS / "albert" / "test_albert_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_xglm.py": E2E_MODELS / "xglm" / "test_xglm_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_whisper.py": E2E_MODELS / "whisper" / "test_whisper_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_starcoder2.py": E2E_MODELS / "starcoder2" / "test_starcoder2_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_stablelm.py": E2E_MODELS / "stablelm" / "test_stablelm_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_segformer.py": E2E_MODELS / "segformer" / "test_segformer_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_rwkv.py": E2E_MODELS / "rwkv" / "test_rwkv_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_phi_moe.py": E2E_MODELS / "phi_moe" / "test_phi_moe_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_phi.py": E2E_MODELS / "phi" / "test_phi_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_opt.py": E2E_MODELS / "opt" / "test_opt_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_olmo.py": E2E_MODELS / "olmo" / "test_olmo_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_nemotron.py": E2E_MODELS / "nemotron" / "test_nemotron_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_mixtral.py": E2E_MODELS / "mixtral" / "test_mixtral_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_mistral.py": E2E_MODELS / "mistral" / "test_mistral_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_mamba.py": E2E_MODELS / "mamba" / "test_mamba_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_llama.py": E2E_MODELS / "llama" / "test_llama_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_internlm.py": E2E_MODELS / "internlm" / "test_internlm_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_granite.py": E2E_MODELS / "granite" / "test_granite_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_gpt_neox.py": E2E_MODELS / "gpt_neox" / "test_gpt_neox_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_gpt_neo.py": E2E_MODELS / "gpt_neo" / "test_gpt_neo_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_gpt2.py": E2E_MODELS / "gpt2" / "test_gpt2_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_gemma.py": E2E_MODELS / "gemma" / "test_gemma_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_falcon.py": E2E_MODELS / "falcon" / "test_falcon_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_codegen.py": E2E_MODELS / "codegen" / "test_codegen_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_bloom.py": E2E_MODELS / "bloom" / "test_bloom_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_engine_bark.py": E2E_MODELS / "bark" / "test_bark_builder_engine.py",
    REPO_ROOT / "tests" / "builder" / "test_family_marian_debug_runner.py": E2E_MODELS / "marian" / "test_marian_debug_runner.py",
    REPO_ROOT / "tests" / "builder" / "test_family_sam3.py": E2E_MODELS / "sam3" / "test_sam3_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_timm_vit.py": E2E_MODELS / "timm_vit" / "test_timm_vit_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_nemotron_h_tp.py": E2E_MODELS / "nemotron_h" / "test_nemotron_h_builder_tp.py",
    REPO_ROOT / "tests" / "builder" / "test_family_yolox.py": E2E_MODELS / "yolox" / "test_yolox_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_wan_t2v.py": E2E_MODELS / "wan_t2v" / "test_wan_t2v_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_sam.py": E2E_MODELS / "sam" / "test_sam_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_roberta.py": E2E_MODELS / "roberta" / "test_roberta_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_pixart.py": E2E_MODELS / "pixart" / "test_pixart_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_phi4mm.py": E2E_MODELS / "phi4_multimodal" / "test_phi4_multimodal_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_nemotron_h.py": E2E_MODELS / "nemotron_h" / "test_nemotron_h_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_mpnet.py": E2E_MODELS / "mpnet" / "test_mpnet_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_ltx_video.py": E2E_MODELS / "ltx_video" / "test_ltx_video_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_gpt_oss.py": E2E_MODELS / "gpt_oss" / "test_gpt_oss_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_glm.py": E2E_MODELS / "glm" / "test_glm_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_distilbert.py": E2E_MODELS / "distilbert" / "test_distilbert_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_deepseek_v2.py": E2E_MODELS / "deepseek_v2" / "test_deepseek_v2_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_bert.py": E2E_MODELS / "bert" / "test_bert_family_plugin.py",
    REPO_ROOT / "tests" / "builder" / "test_family_elf.py": E2E_MODELS / "elf_flow" / "test_elf_flow_family_plugin.py",
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
})
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
    "TestStablelmPlugin": (
        E2E_MODELS / "stablelm" / "test_stablelm_family_plugin_weights.py"
    ),
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
    "locateanything": (
        E2E_MODELS / "locateanything" / "test_locateanything_registry_contract.py"
    ),
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
SHARED_TIMM_VIT_TRT_PATH_TOOL = REPO_ROOT / "tools" / "validation" / "timm_vit" / "benchmark_trt_paths.py"
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
    REPO_ROOT / "tests" / "tools" / "test_model_plugin_isolation.py",
    REPO_ROOT / "tests" / "tools" / "test_perf_evolve_prompt.py",
    REPO_ROOT / "tests" / "tools" / "test_runtime_strategy_matrix_checker.py",
    REPO_ROOT / "tests" / "tools" / "test_sampling_contract_plugin.py",
    REPO_ROOT / "tests" / "tools" / "test_schedule_e2e.py",
    REPO_ROOT / "tests" / "tools" / "test_sol_estimate.py",
    REPO_ROOT / "tests" / "tools" / "test_task_eval.py",
    REPO_ROOT / "tests" / "tools" / "test_vl_qa_plugin.py",
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
MODEL_OWNED_DIFF_LOGITS_HANDLERS = (
    FAMILIES / "whisper" / "diff_logits.py",
)
MODEL_OWNED_DIFF_AUDIO_HANDLERS = (
    FAMILIES / "bark" / "diff_audio.py",
)
MODEL_OWNED_DIFF_T5_HANDLERS = (
    FAMILIES / "wan_t2v" / "diff_t5.py",
)
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
_FORBIDDEN_E2E_IMPORT_RE = re.compile(
    r"tests\.e2e_harness\.(?:runners|comparators|references)"
)
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
        path.name
        for path in FAMILIES.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    }


def _format_violations(violations: list[tuple[Path, int, str]]) -> str:
    return "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{line}: {detail}"
        for path, line, detail in violations
    )


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
                violations.append((
                    path,
                    line_no,
                    f"shared CMake names model source folder {match.group(1)}",
                ))

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
            violations.append((
                CONFIG_SCHEMA_CMAKE,
                0,
                f"shared config schema manifest contains model-owned schema {needle}",
            ))

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
        "bark": "runtime_config_schemas = [\"config_schema.cpp|register_audio_bark_schema\"]",
        "magpie": "runtime_config_schemas = [\"config_schema.cpp|register_audio_magpie_schema\"]",
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
        "test_diffusion_generation_plan.cpp",
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
        "test_speech_decode_stop_policy.cpp",
        "test_speech_depth_plan.cpp",
        "test_speech_generation_helpers.cpp",
        "test_speech_mimi_decode_plan.cpp",
        "test_speech_pipeline.cpp",
        "test_speech_runtime_plan.cpp",
        "test_speech_subprocess_seam.cpp",
        "test_speech_temporal_embed_plan.cpp",
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
    Preconditions: text-generation, audio, and diffusion model tests have local homes.
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
            "decoder_kv_cache",
            "text_generation",
            "libtrtmc_model_text_generation.so",
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
        "test_ipa_tokenizer.cpp": (
            "MagpieTTS",
        ),
        "test_perception_preprocess_seams.cpp": (
            "SAM",
            "Sam",
            "sam_",
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
        "test_text_generation_pipeline",
        "test_recurrent_pipeline",
        "test_encoder_pipeline",
        "test_vl_pipeline",
        "add_dependencies(test_model_plugin_loader trtmc_model_text_generation)",
    )
    violations = [
        (CMAKE_ROOT, 0, f"top-level CMake hardcodes model-owned test detail {needle}")
        for needle in forbidden
        if needle in text
    ]

    expected_manifest_entries = {
        "text_generation": "test_text_generation_pipeline|test_text_generation_pipeline.cpp",
        "recurrent": "test_recurrent_pipeline|test_recurrent_pipeline.cpp",
        "encoder": "test_encoder_pipeline|test_encoder_pipeline.cpp",
        "vision_language": "test_vl_pipeline|test_vl_pipeline.cpp",
    }
    for model, entry in expected_manifest_entries.items():
        manifest = RUNTIME_MODELS / model / "MODEL.toml"
        if entry not in manifest.read_text(encoding="utf-8"):
            violations.append((manifest, 0, f"missing model-owned C++ test {entry}"))

    assert not violations, _format_violations(violations)


def test_shared_chat_template_core_is_only_a_registry() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete chat template formats in runtime model-owned files.
    Preconditions: text_generation and recurrent models register their templates.
    Postconditions: shared chat_template core contains no model or format markers.
    """
    forbidden = (
        "ChatTemplateFormat::k",
        "kChatML",
        "kMistral",
        "kPhi",
        "kGemma",
        "kLlama3",
        "kNemotron",
        "im_start",
        "[INST]",
        "start_of_turn",
        "start_header_id",
        "extra_id_0",
        "SPECIAL_10",
        "truncate_history_thinking",
        "Nemotron",
        "Mistral",
        "Gemma",
        "Llama",
        "Phi",
    )
    violations = []
    for path in CHAT_TEMPLATE_CORE_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared chat template registry contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )

    expected_owned_files = (
        RUNTIME_MODELS / "text_generation" / "chat_templates.cpp",
        RUNTIME_MODELS / "recurrent" / "chat_templates.cpp",
    )
    violations.extend(
        (path, 0, "missing model-owned chat template registration")
        for path in expected_owned_files
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_runtime_core_and_domains_do_not_name_single_family_defaults() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model defaults/comments in model-owned runtime files.
    Preconditions: shared runtime core/domain files provide generic utilities.
    Postconditions: shared runtime files contain no single-family default strings.
    """
    forbidden_by_file = {
        "diffusion_types.h": (
            "wan_3d",
            "FLUX",
            "Flux",
            "flux",
            "video_height{480}",
            "video_width{832}",
            "video_num_frames{81}",
            "dit_dim{1536}",
            "text_encoder_dim{4096}",
        ),
        "flow_match_euler_scheduler.cpp": (
            "pipeline_flux",
            "QwenImage",
            "Qwen",
            "Z-Image",
            "FLUX",
            "Flux",
            "flux",
        ),
        "device_ops.h": ("Magpie", "magpie"),
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


def test_runtime_strategy_default_is_model_owned() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep legacy runtime_strategy fallback owned by the model manifest.
    Preconditions: generated plugin index exposes the manifest default.
    Postconditions: shared runtime files do not hardcode text-generation defaults.
    """
    violations = []
    for path in RUNTIME_STRATEGY_DEFAULT_FILES:
        text = path.read_text(encoding="utf-8")
        if "decoder_kv_cache" in text:
            violations.append(
                (path, 0, "shared runtime file hardcodes text-generation strategy default")
            )

    text_generation_manifest = RUNTIME_MODELS / "text_generation" / "MODEL.toml"
    manifest_text = text_generation_manifest.read_text(encoding="utf-8")
    if 'default_runtime_strategy = "decoder_kv_cache"' not in manifest_text:
        violations.append(
            (text_generation_manifest, 0, "missing text-generation-owned default strategy")
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
        (SHARED_DIFF_LOGITS_TOOL, 0,
         f"shared logit diff tool contains family term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned logit diff handler")
        for path in MODEL_OWNED_DIFF_LOGITS_HANDLERS + (
            FAMILIES / "mamba" / "diff_logits.py",
            FAMILIES / "rwkv" / "diff_logits.py",
        )
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_runner_parity_tool_uses_debug_runner_dispatch() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep runner parity generic and route model-owned runners through metadata.
    Preconditions: debug_runner.runner_from_bundle dispatches family runners.
    Postconditions: shared runner parity tool names no model-owned debug runners.
    """
    text = SHARED_RUNNER_PARITY_TOOL.read_text(encoding="utf-8")
    forbidden = (
        "MambaTrtRunner",
        "RwkvTrtRunner",
        "HybridTrtRunner",
        'runtime_strategy == "ssm_recurrent"',
        'runtime_strategy == "rwkv_recurrent"',
        'runtime_strategy == "hybrid_mamba_attention"',
    )
    violations = [
        (SHARED_RUNNER_PARITY_TOOL, 0,
         f"shared runner parity tool contains model-owned dispatch {needle}")
        for needle in forbidden
        if needle in text
    ]
    if "runner_from_bundle" not in text:
        violations.append((
            SHARED_RUNNER_PARITY_TOOL,
            0,
            "shared runner parity tool should use debug_runner.runner_from_bundle",
        ))

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
        (SHARED_DIFF_LAYERS_TOOL, 0,
         f"shared layer diff tool contains family-owned builder term {needle}")
        for needle in forbidden
        if needle in text
    ]
    for needle in ("family_has_capability", "debug_layer_outputs", "plugin.build_engine"):
        if needle not in text:
            violations.append((
                SHARED_DIFF_LAYERS_TOOL,
                0,
                f"shared layer diff tool missing dispatch marker {needle}",
            ))

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
        violations.append((
            E2E_CONTRACTS,
            0,
            "RunContext hardcodes speech_to_speech for --hf-python",
        ))
    if "runtime_cli_requires_hf_python" not in contracts_text:
        violations.append((
            E2E_CONTRACTS,
            0,
            "RunContext should read runtime_cli_requires_hf_python metadata",
        ))

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
        'runtime_strategy == "ssm_recurrent"',
        'runtime_strategy == "rwkv_recurrent"',
        'runtime_strategy == "hybrid_mamba_attention"',
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

    expected_hooks = (
        FAMILIES / "mamba" / "perf_hooks.py",
    )
    violations.extend(
        (path, 0, "missing model-owned performance hook")
        for path in expected_hooks
        if not path.is_file()
    )

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
        violations.append((shared_path, 0, "Qwen parity test should not live in shared tools tests"))
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
            if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
                violations.append((
                    path,
                    0,
                    f"entry {index} missing model-owned benchmark.command template",
                ))

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
        (SHARED_DIFF_AUDIO_TOOL, 0,
         f"shared audio diff tool contains family term {needle}")
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
        violations.append((
            root_tool,
            0,
            "root PersonaPlex diff wrapper must not exist",
        ))

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
        violations.append((
            SHARED_QWEN_AIME_BENCHMARK_TOOL,
            0,
            "root Qwen benchmark wrapper must not exist",
        ))

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
        violations.append((
            SHARED_TIMM_VIT_TRT_PATH_TOOL,
            0,
            "root timm_vit benchmark wrapper must not exist",
        ))

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
        violations.append((
            SHARED_QWEN_FLASHINFER_BENCHMARK_TOOL,
            0,
            "root Qwen FlashInfer benchmark wrapper must not exist",
        ))

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
        (SHARED_DEBUG_DIFFUSION_PIPELINE_TOOL, 0,
         f"shared diffusion debug tool contains family term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned diffusion debug handler")
        for path in MODEL_OWNED_DEBUG_DIFFUSION_PIPELINE_HANDLERS
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_shared_decoder_builders_describe_capabilities_not_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep shared decoder builders generic and family-neutral.
    Preconditions: model-specific decoder usage lives in family plugins.
    Postconditions: shared builder docs/comments do not name model families.
    """
    forbidden = (
        "Qwen",
        "qwen",
        "LLaMA",
        "Bark",
        "bark",
        "CodeGen",
        "GPT-J",
        "GPT-Neo",
        "OPT",
        "Bloom",
        "Falcon",
        "Mistral",
        "Gemma",
        "Phi",
        "Nemotron",
        "Mamba",
        "mamba",
        "RWKV",
        "rwkv",
        "MoE",
        "moe",
    )
    violations = []
    for path in SHARED_DECODER_BUILDER_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared decoder builder contains family term {needle}")
            for needle in forbidden
            if needle in text
        )

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
    )
    violations = [
        (path, 0, "single-owner runtime helper should live under src/runtime/models")
        for path in forbidden_paths
        if path.exists()
    ]

    perception_types = RUNTIME_DOMAINS / "perception" / "perception_types.h"
    text = perception_types.read_text(encoding="utf-8")
    if "Sam3Config" in text:
        violations.append((perception_types, 0, "Sam3Config should live under src/runtime/models"))
    for needle in ("SamConfig", "SamResult"):
        if needle in text:
            violations.append((
                perception_types,
                0,
                f"{needle} should live under src/runtime/models",
            ))
    for path in sorted((RUNTIME_DOMAINS / "perception").glob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in ("SAM", "Sam", "sam_"):
            if needle in text:
                violations.append((
                    path,
                    0,
                    f"shared perception domain contains single-family term {needle}",
                ))

    audio_configs = RUNTIME_DOMAINS / "audio" / "audio_configs.h"
    if audio_configs.exists():
        text = audio_configs.read_text(encoding="utf-8")
        for needle in ("MagpieTTSConfig", "SpeechConfig", "OmniConfig"):
            if needle in text:
                violations.append((
                    audio_configs,
                    0,
                    f"{needle} should live under src/runtime/models",
                ))

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
    for model in ("bark", "omni", "rnnt", "speech", "whisper"):
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
    forbidden = (
        "Qwen/" + "Qwen2.5-VL-3B-Instruct",
    )
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
    )
    violations = [
        (path, 0, "single-owner diffusion generation helper should live under src/runtime/models")
        for path in forbidden_paths
        if path.exists()
    ]

    forbidden_terms = (
        "FluxGenerationPlan",
        "WanGenerationPlan",
        "PixArtGenerationPlan",
        "run_flux_denoising_steps",
    )
    for path in sorted(RUNTIME_DIFFUSION_DOMAINS.glob("*.h")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(
            (path, 0, f"shared diffusion domain contains model generation plan {needle}")
            for needle in forbidden_terms
            if needle in text
        )

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
        violations.append((
            AUTOPILOT_DISCOVER,
            0,
            "autopilot discovery does not use family_probe_model_types",
        ))
    for needle in (
        '"qwen"',
        '"qwen2"',
        '"qwen3"',
        '"magpie_tts"',
        '"canary"',
        '"lance"',
    ):
        if needle in discover_text:
            violations.append((
                AUTOPILOT_DISCOVER,
                0,
                f"autopilot discovery hardcodes model type {needle}",
            ))

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
            violations.append((
                WARM_HF_CACHE,
                0,
                f"warm cache hardcodes family-owned asset {needle}",
            ))

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
            violations.append((
                shared_path,
                0,
                "model-specific builder test should not live in shared tests/builder",
            ))
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
        path: path.read_text(encoding="utf-8")
        for path in SHARED_PLUGIN_WEIGHT_TEST_FILES
    }
    support_path = REPO_ROOT / "tests" / "builder" / "family_plugin_test_support.py"
    support_text = support_path.read_text(encoding="utf-8")

    violations = []
    for class_name, owned_path in MODEL_OWNED_PLUGIN_WEIGHT_TESTS.items():
        for shared_path, shared_text in shared_texts.items():
            if class_name in shared_text:
                violations.append((
                    shared_path,
                    0,
                    f"model-specific plugin weight test {class_name} is shared",
                ))
        if class_name in support_text:
            violations.append((
                support_path,
                0,
                f"generic support imports model-specific test {class_name}",
            ))
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
                violations.append((
                    shared_path,
                    0,
                    f"shared plugin tests contain checkpoint-specific key {needle}",
                ))
        if needle in support_text:
            violations.append((
                support_path,
                0,
                f"shared plugin support contains checkpoint-specific key {needle}",
            ))

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
        "speech_to_speech",
        "speech_to_text_rnnt",
        "hybrid_mamba_attention",
        "ssm_recurrent",
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
    Intent: keep shared image preprocessing strategies generic and family-selected.
    Preconditions: model plugins write preprocessor_type into bundle config.
    Postconditions: shared runtime preprocessing does not expose model-named strategies.
    """
    forbidden = (
        "qwen_merge_group",
        "QwenMergeGroup",
        "locateanything_patchify",
        "LocateAnything",
    )
    violations = []
    for path in RUNTIME_MULTIMODAL_PREPROCESSOR_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared multimodal preprocessor contains {needle}")
            for needle in forbidden
            if needle in text
        )

    assert not violations, _format_violations(violations)


def test_shared_recurrent_contracts_are_model_neutral() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep recurrent-family output layouts in the recurrent model owner.
    Preconditions: shared recurrent domain exposes only generic state contracts.
    Postconditions: shared contracts do not name recurrent model variants, and
    model-owned output initializer coverage is registered by the recurrent
    manifest.
    """
    forbidden = (
        "initialize_rwkv_outputs",
        "initialize_mamba_outputs",
        "rwkv",
        "mamba",
    )
    text = RUNTIME_RECURRENT_CONTRACTS.read_text(encoding="utf-8")
    violations = [
        (RUNTIME_RECURRENT_CONTRACTS, 0, f"shared recurrent contract contains {needle}")
        for needle in forbidden
        if needle in text
    ]

    owned_header = RUNTIME_MODELS / "recurrent" / "recurrent_output_initializers.h"
    owned_test = CPP_TESTS / "models" / "recurrent" / "test_recurrent_output_initializers.cpp"
    manifest = RUNTIME_MODELS / "recurrent" / "MODEL.toml"
    expected_manifest_entry = (
        "test_recurrent_output_initializers|test_recurrent_output_initializers.cpp"
    )
    for path in (owned_header, owned_test):
        if not path.is_file():
            violations.append((path, 0, "missing recurrent-owned output initializer surface"))
    if expected_manifest_entry not in manifest.read_text(encoding="utf-8"):
        violations.append((manifest, 0, "missing recurrent-owned output initializer test"))

    assert not violations, _format_violations(violations)


def test_shared_audio_domain_has_no_model_owned_feature_extractors() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific audio feature-extractor policy in model owners.
    Preconditions: shared audio domain exposes neutral signal-processing helpers.
    Postconditions: NeMo/RNNT feature extraction lives under src/runtime/models/rnnt.
    """
    forbidden = (
        "NeMo",
        "nemo",
        "RNNT",
        "rnnt",
        "extract_nemo_mel_spectrogram",
    )
    violations = []
    for path in RUNTIME_AUDIO_DOMAIN_FILES:
        text = path.read_text(encoding="utf-8")
        violations.extend(
            (path, 0, f"shared audio domain contains model-owned term {needle}")
            for needle in forbidden
            if needle in text
        )

    owned_files = (
        RUNTIME_MODELS / "rnnt" / "audio_helpers.h",
        RUNTIME_MODELS / "rnnt" / "audio_helpers.cpp",
        CPP_TESTS / "models" / "rnnt" / "test_rnnt_audio_helpers.cpp",
    )
    violations.extend(
        (path, 0, "missing RNNT-owned audio helper surface")
        for path in owned_files
        if not path.is_file()
    )
    manifest = RUNTIME_MODELS / "rnnt" / "MODEL.toml"
    if "test_rnnt_audio_helpers|test_rnnt_audio_helpers.cpp" not in manifest.read_text(
        encoding="utf-8"
    ):
        violations.append((manifest, 0, "missing RNNT-owned audio helper test"))

    assert not violations, _format_violations(violations)


def test_shared_debug_runner_has_no_model_owned_runners() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific TRT debug runners in owning family modules.
    Preconditions: debug_runner.py exposes generic TRT/CUDA debugging helpers.
    Postconditions: shared debug runner does not define model-owned runner classes.
    """
    forbidden = (
        "class MambaTrtRunner",
        "class RwkvTrtRunner",
        "class HybridTrtRunner",
        "MambaTrtRunner(",
        "RwkvTrtRunner(",
        "HybridTrtRunner(",
        "ssm_recurrent",
        "rwkv_recurrent",
        "hybrid_mamba_attention",
        "marian_translation",
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
        FAMILIES / "whisper" / "debug_runner.py",
        FAMILIES / "qwen_image" / "debug_runner.py",
        FAMILIES / "marian" / "debug_runner.py",
    )
    violations.extend(
        (path, 0, "missing model-owned debug runner")
        for path in expected_owned_files
        if not path.is_file()
    )
    root_test_text = DEBUG_RUNNER_TEST.read_text(encoding="utf-8")
    root_extended_test_text = DEBUG_RUNNER_EXTENDED_TEST.read_text(encoding="utf-8")
    shared_test_forbidden = (
        "marian_translation",
        "rwkv_recurrent",
        "ssm_recurrent",
        "hybrid_mamba_attention",
        "MambaTrtRunner",
        "RwkvTrtRunner",
        "WhisperTrtRunner",
        "HybridTrtRunner",
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
        E2E_MODELS / "whisper" / "test_whisper_debug_runner.py",
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
            violations.append((
                E2E_ORCHESTRATOR,
                0,
                f"shared repro command contains model-owned detail {needle}",
            ))

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
                violations.append((
                    path,
                    0,
                    f"non-segmentation sidecar contains segmentation behavior {needle}",
                ))

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
            term
            for family, terms in family_terms.items()
            if family != owner
            for term in terms
        )
        for rel in relative_files:
            path = owner_dir / rel
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in sibling_terms:
                if needle in text:
                    violations.append((
                        path,
                        0,
                        f"{owner} diffusion E2E plugin contains sibling behavior {needle}",
                    ))

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
    Postconditions: shared HF Transformers reference contains no named-family branches.
    """
    forbidden = (
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
        "model_type == \"dpr\"",
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
        E2E_MODELS
        / "nemotron_labs_diffusion"
        / "e2e_plugins"
        / "runners"
        / "text_generation.py"
    )
    owned_text = owned_runner.read_text(encoding="utf-8") if owned_runner.is_file() else ""
    if not owned_runner.is_file():
        violations.append((owned_runner, 0, "missing Nemotron-owned text-generation runner"))
    for needle in forbidden:
        if needle not in owned_text:
            violations.append((owned_runner, 0, f"missing Nemotron-owned CLI term {needle}"))

    model_test = (
        E2E_MODELS
        / "nemotron_labs_diffusion"
        / "test_nemotron_labs_diffusion_runner.py"
    )
    if not model_test.is_file():
        violations.append((model_test, 0, "missing Nemotron-owned runner unit test"))

    assert not violations, _format_violations(violations)


def test_hf_transformers_model_plugins_do_not_name_sibling_families() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep HF Transformers special reference behavior in owning model folders.
    Preconditions: model-owned hf_transformers.py sidecars are generated per family.
    Postconditions: no HF reference sidecar carries another family's special branch.
    """
    family_term_groups = (
        ({"bark"}, (
            "BarkModel",
            "_run_text_to_audio_ref",
            "hf_text_to_audio",
        )),
        ({"canary"}, (
            "Canary",
            "canary",
            "_run_canary_ref",
            "nemo_canary_stt",
        )),
        ({"nemotron_speech_streaming"}, (
            "Nemotron speech",
            "_run_nemo_speech_ref",
            "nemo_speech_stt",
        )),
        ({"nemotron_labs_diffusion"}, (
            "nemotron_labs_diffusion",
            "Nemotron Labs",
            "linear_spec",
            "_run_nemotron_labs_diffusion_generation",
        )),
        ({"locateanything"}, (
            "LocateAnything",
            "locateanything",
            "_run_locateanything",
            "patchify_chw",
        )),
        ({"qwen_vl"}, (
            "qwen",
            "Qwen",
        )),
        ({"internvl"}, (
            "internvl",
            "InternVL",
        )),
        ({"dpr"}, (
            "DPRContextEncoder",
            "DPRContextEncoderTokenizerFast",
            "dpr_context_embed",
            "model_type == \"dpr\"",
            "model_type == 'dpr'",
        )),
        ({"timm_vit"}, (
            "import timm",
            "timm.create_model",
            "_run_image_classification_ref",
        )),
        ({"sam", "sam3"}, (
            "SamModel",
            "SamProcessor",
            "_run_prompted_segmentation_ref",
            "hf_prompted_segmentation",
        )),
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
            term
            for owners, terms in family_term_groups
            if owner not in owners
            for term in terms
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
                "return \"image_classification\"",
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
                REPO_ROOT / "tests" / "e2e_harness" / "comparators" / "diffusion_text_generation.py",
            ),
            "behavior_terms": (
                "class DiffusionTextGenerationRunner",
                "class DiffusionTextGenerationComparator",
                "return \"diffusion_text_generation\"",
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
            key for key in expected_keys
            if not isinstance(config, dict) or key not in config
        ]
        violations.extend(
            (path, 0, f"manifest missing metadata.contract_config.{key}")
            for key in missing
        )

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
    }
    for old_root_test, model_test in moved_tests.items():
        if old_root_test.exists():
            violations.append((old_root_test, 0, "model-owned unit test lives in tests/tools"))
        if not model_test.is_file():
            violations.append((model_test, 0, "missing model-owned unit test"))

    sam3_prompted_harness = (
        E2E_MODELS / "sam3" / "test_sam3_prompted_segmentation_harness.py"
    )
    if not sam3_prompted_harness.is_file():
        violations.append((sam3_prompted_harness, 0, "missing model-owned unit test"))

    assert not violations, _format_violations(violations)


def test_root_e2e_runner_cli_alignment_has_no_model_owned_cases() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep concrete model runner CLI checks with the owning model.
    Preconditions: shared CLI alignment tests cover generic runner contracts only.
    Postconditions: model-owned runner cases live under tests/e2e/models.
    """
    root_test = REPO_ROOT / "tests" / "tools" / "test_e2e_runner_cli_alignment.py"
    text = root_test.read_text(encoding="utf-8")
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
    )
    violations = [
        (root_test, 0, f"root CLI alignment test contains model-owned term {needle}")
        for needle in forbidden
        if needle in text
    ]
    violations.extend(
        (path, 0, "missing model-owned CLI alignment test")
        for path in expected_model_tests
        if not path.is_file()
    )

    assert not violations, _format_violations(violations)


def test_root_hf_transformers_helper_tests_have_no_model_owned_references() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: keep model-specific HF reference checks with owning model tests.
    Preconditions: root HF helper tests cover only generic shared helpers.
    Postconditions: model reference tests live under tests/e2e/models.
    """
    root_test = REPO_ROOT / "tests" / "tools" / "test_hf_transformers_reference_helpers.py"
    text = root_test.read_text(encoding="utf-8")
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
    violations = [
        (root_test, 0, f"root HF helper test contains model-owned term {needle}")
        for needle in forbidden
        if needle in text
    ]
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
            term
            for family, terms in family_terms.items()
            if family != owner
            for term in terms
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
        "return \"text_to_audio\"",
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
                                    violations.append((
                                        path,
                                        node.lineno,
                                        f"imports shared {module}.{alias.name}",
                                    ))
                        _check_absolute_import(
                            module, owner, family_ids, path, node.lineno, violations
                        )
                    else:
                        if node.level >= 2 and not module:
                            for alias in node.names:
                                if alias.name in _FORBIDDEN_SHARED_BUILDER_MODULES:
                                    violations.append((
                                        path,
                                        node.lineno,
                                        f"imports shared helper {'.' * node.level}{alias.name}",
                                    ))
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
            threshold = model_dir / "thresholds" / manifest.name
            if not threshold.is_file():
                violations.append((
                    manifest,
                    0,
                    f"missing threshold sidecar {threshold.relative_to(REPO_ROOT)}",
                ))

        for path in model_dir.rglob("*.py"):
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                if _FORBIDDEN_E2E_IMPORT_RE.search(line):
                    violations.append((
                        path,
                        line_no,
                        "imports shared E2E runner/reference/comparator",
                    ))

    assert not violations, _format_violations(violations)
