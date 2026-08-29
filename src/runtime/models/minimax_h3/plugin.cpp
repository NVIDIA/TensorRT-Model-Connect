/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/minimax_h3/native_plugin_loader.h"
#include "runtime/models/minimax_h3/pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using SectionMap = std::unordered_map<std::string, BundleSectionInfo>;

struct CacheConfig {
    bool enabled{false};
    float threshold{0.025F};
};

bool is_lower_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

MiniMaxH3Workflow load_workflow(const PipelineContext& ctx) {
    const std::string value = extract_json_string(ctx.config_json, "workflow", "t2va");
    if (value == "t2va")
        return MiniMaxH3Workflow::kT2va;
    if (value == "fl2va")
        return MiniMaxH3Workflow::kFl2va;
    if (value == "ref2va")
        return MiniMaxH3Workflow::kRef2va;
    throw std::runtime_error("MiniMax-H3 bundle has an invalid workflow");
}

SectionMap index_sections(const BundleInfo& info, bool first_block_cache,
                          MiniMaxH3Workflow workflow) {
    constexpr std::array<const char*, 5> monolithic_names = {
        "text_encoder_plan", "adaln_precompute_plan", "denoiser_plan", "vae_tile_decoder_plan",
        "audio_vae_decoder_plan"};
    constexpr std::array<const char*, 7> first_block_cache_names = {
        "text_encoder_plan",     "adaln_precompute_plan", "denoiser_head_plan",
        "denoiser_tail_plan",    "denoiser_finish_plan",  "vae_tile_decoder_plan",
        "audio_vae_decoder_plan"};
    SectionMap sections;
    const auto add_section = [&](const char* name) {
        const auto it =
            std::find_if(info.sections.begin(), info.sections.end(),
                         [name](const BundleSectionInfo& item) { return item.name == name; });
        if (it == info.sections.end() || it->size == 0)
            throw std::runtime_error(std::string("MiniMax-H3 bundle is missing ") + name);
        sections.emplace(name, *it);
    };
    constexpr std::array<const char*, 7> fl2va_names = {
        "language_conditioner_plan", "vision_conditioner_plan", "vae_encoder_tile_t1_plan",
        "adaln_precompute_plan",     "fl2va_denoiser_plan",     "vae_tile_decoder_plan",
        "audio_vae_decoder_plan"};
    if (workflow == MiniMaxH3Workflow::kFl2va) {
        for (const char* name : fl2va_names)
            add_section(name);
    } else if (workflow == MiniMaxH3Workflow::kRef2va) {
        constexpr std::array<const char*, 10> ref2va_names = {
            "language_conditioner_plan",     "vision_conditioner_image_plan",
            "vision_conditioner_video_plan", "vae_encoder_tile_t1_plan",
            "vae_encoder_tile_t17_plan",     "audio_vae_encoder_plan",
            "adaln_precompute_plan",         "ref2va_denoiser_plan",
            "vae_tile_decoder_plan",         "audio_vae_decoder_plan"};
        for (const char* name : ref2va_names)
            add_section(name);
    } else if (first_block_cache) {
        for (const char* name : first_block_cache_names)
            add_section(name);
    } else {
        for (const char* name : monolithic_names)
            add_section(name);
    }
    return sections;
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* data = find_section(bundle, "tokenizer.json");
    if (data == nullptr || data->empty())
        throw std::runtime_error("MiniMax-H3 bundle is missing tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data->data(), data->size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-H3 could not create the native Qwen BPE tokenizer");
    return tokenizer;
}

void validate_audio_profile(const PipelineContext& ctx) {
    if (extract_json_int(ctx.config_json, "audio_sample_rate", 32000) != 32000 ||
        extract_json_int(ctx.config_json, "audio_latent_frames", 207) != 207 ||
        extract_json_int(ctx.config_json, "audio_output_samples", 165600) != 165600)
        throw std::runtime_error("MiniMax-H3 bundle has an incompatible audio output profile");
}

void validate_fl2va_profile(const PipelineContext& ctx) {
    if (extract_json_int(ctx.config_json, "min_text_rows", 1) != 1 ||
        extract_json_int(ctx.config_json, "max_text_rows", 4096) != 4096 ||
        extract_json_int(ctx.config_json, "fl2va_keyframe_rows", 1008) != 1008 ||
        extract_json_int(ctx.config_json, "fl2va_vae_tile_size", 256) != 256 ||
        extract_json_int(ctx.config_json, "fl2va_vae_tile_min_overlap", 64) != 64)
        throw std::runtime_error("MiniMax-H3 FL2VA bundle has an incompatible dynamic profile");
    for (const char* name :
         {"processor/preprocessor_config.json", "processor/video_preprocessor_config.json"}) {
        const auto* section = find_section(ctx.bundle, name);
        if (section == nullptr || section->empty())
            throw std::runtime_error(std::string("MiniMax-H3 FL2VA bundle is missing ") + name);
    }
}

void validate_ref2va_profile(const PipelineContext& ctx) {
    constexpr std::array<std::pair<const char*, int32_t>, 17> expected = {{
        {"min_text_rows", 1},
        {"opt_text_rows", 8192},
        {"max_text_rows", 262144},
        {"ref2va_min_condition_video_rows", 0},
        {"ref2va_opt_condition_video_rows", 4096},
        {"ref2va_min_condition_audio_rows", 0},
        {"ref2va_opt_condition_audio_rows", 0},
        {"ref2va_max_condition_video_rows", 258120},
        {"ref2va_max_condition_audio_rows", 2408},
        {"ref2va_max_images", 9},
        {"ref2va_max_videos", 3},
        {"ref2va_max_audios", 3},
        {"ref2va_max_references", 12},
        {"ref2va_reference_min_seconds", 2},
        {"ref2va_reference_max_seconds", 15},
        {"ref2va_vae_tile_size", 256},
        {"ref2va_vae_tile_min_overlap", 64},
    }};
    for (const auto& [name, value] : expected) {
        // These values describe the serialized TensorRT optimization-profile
        // ABI. Missing fields must not inherit runtime defaults: an older
        // Ref2VA plan can otherwise claim audio-only support while retaining
        // a 4,096-row minimum visual condition.
        if (extract_json_int(ctx.config_json, name, -1) != value)
            throw std::runtime_error(
                "MiniMax-H3 Ref2VA bundle has an incompatible dynamic profile");
    }
    if (extract_json_string(ctx.config_json, "ref2va_audio_encoder_implementation", "") !=
            "aten-torchscript-fp32-v1" ||
        extract_json_int(ctx.config_json, "ref2va_audio_encoder_plugin_count", -1) != 1 ||
        extract_json_string(ctx.config_json, "ref2va_audio_encoder_module_format", "") !=
            "torchscript-plan-constant-v1" ||
        extract_json_string(ctx.config_json, "ref2va_audio_encoder_weight_norm", "") !=
            "cuda-frozen-effective-v1" ||
        extract_json_int(ctx.config_json, "ref2va_audio_encoder_hop_length", -1) != 800 ||
        extract_json_int(ctx.config_json, "ref2va_audio_encoder_output_channels", -1) != 32)
        throw std::runtime_error(
            "MiniMax-H3 Ref2VA bundle has an incompatible exact audio encoder implementation");
    if (extract_json_bool(ctx.config_json, "ref2va_audio_encoder_cuda_graphs", true))
        throw std::runtime_error(
            "MiniMax-H3 Ref2VA bundle must disable CUDA graphs for the exact audio encoder");
    if (!extract_json_bool(ctx.config_json, "ref2va_audio_encoder_cudnn_tf32", false) ||
        extract_json_bool(ctx.config_json, "ref2va_audio_encoder_matmul_tf32", true) ||
        extract_json_bool(ctx.config_json, "ref2va_audio_encoder_graph_optimizer", true) ||
        !extract_json_bool(ctx.config_json, "ref2va_audio_encoder_cudnn_enabled", false) ||
        extract_json_bool(ctx.config_json, "ref2va_audio_encoder_cudnn_benchmark", true) ||
        extract_json_bool(ctx.config_json, "ref2va_audio_encoder_cudnn_deterministic", true))
        throw std::runtime_error(
            "MiniMax-H3 Ref2VA bundle has incompatible audio encoder execution precision");
    if (extract_json_int_array(ctx.config_json, "ref2va_audio_encoder_input_profile", 3) !=
        std::vector<int32_t>({64000, 165600, 480000}))
        throw std::runtime_error(
            "MiniMax-H3 Ref2VA bundle has an incompatible exact audio encoder profile");
    const int32_t audio_encoder_module_bytes =
        extract_json_int(ctx.config_json, "ref2va_audio_encoder_module_bytes", -1);
    const std::string audio_encoder_module_sha256 =
        extract_json_string(ctx.config_json, "ref2va_audio_encoder_module_sha256", "");
    if (audio_encoder_module_bytes < (300 << 20) || audio_encoder_module_bytes > (400 << 20) ||
        !is_lower_sha256(audio_encoder_module_sha256))
        throw std::runtime_error(
            "MiniMax-H3 Ref2VA bundle has incompatible audio encoder trace provenance");
    if (extract_json_string(ctx.config_json, "ref2va_vision_plan_layout", "") !=
            "split-image-video-v1" ||
        extract_json_string(ctx.config_json, "minimax_h3_native_plugin_section", "") !=
            "minimax_h3_native_plugin_so" ||
        extract_json_string(ctx.config_json, "minimax_h3_native_plugin_artifact", "") !=
            "libtrtmc_minimax_h3_native_plugin.so" ||
        extract_json_int(ctx.config_json, "minimax_h3_native_plugin_abi", -1) != 1 ||
        extract_json_string(ctx.config_json, "minimax_h3_native_plugin_identity", "") !=
            "trtmc.minimax_h3.native_plugin:aten-ops:1" ||
        extract_json_string(ctx.config_json, "ref2va_language_attention_implementation", "") !=
            "tensorrt-bf16-iattention-v1" ||
        extract_json_string(ctx.config_json, "ref2va_language_attention_precision", "") != "bf16" ||
        extract_json_string(ctx.config_json, "ref2va_language_q_pre_scale_precision", "") !=
            "bf16" ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_attention_implementation", "") !=
            "aten-bf16-sdpa-v1" ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_attention_precision", "") !=
            "bf16" ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_attention_scale", "") !=
            "fp64:0x1.e2b7dddfefa66p-4" ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_patch_implementation", "") !=
            "aten-bf16-conv3d-v1" ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_patch_precision", "") != "bf16" ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_linear_implementation", "") !=
            "aten-bf16-linear-v1" ||
        extract_json_int(ctx.config_json, "ref2va_image_vision_linear_count", -1) != 116 ||
        extract_json_string(ctx.config_json, "ref2va_image_vision_layer_norm_implementation", "") !=
            "aten-bf16-layer-norm-v1" ||
        extract_json_int(ctx.config_json, "ref2va_image_vision_layer_norm_count", -1) != 58 ||
        extract_json_string(ctx.config_json, "ref2va_video_vision_attention_implementation", "") !=
            "tensorrt-fp16-iattention-v1" ||
        extract_json_string(ctx.config_json, "ref2va_video_vision_attention_precision", "") !=
            "fp16" ||
        extract_json_string(ctx.config_json, "ref2va_video_vision_q_pre_scale_precision", "") !=
            "fp16")
        throw std::runtime_error("MiniMax-H3 Ref2VA bundle has an incompatible vision plan layout");
    if (extract_json_int_array(ctx.config_json, "ref2va_image_vision_patch_input_shape", 2) !=
            std::vector<int32_t>({-1, 1536}) ||
        extract_json_int_array(ctx.config_json, "ref2va_image_vision_patch_weight_shape", 5) !=
            std::vector<int32_t>({1152, 3, 2, 16, 16}) ||
        extract_json_int_array(ctx.config_json, "ref2va_image_vision_patch_bias_shape", 1) !=
            std::vector<int32_t>({1152}) ||
        extract_json_int_array(ctx.config_json, "ref2va_image_vision_patch_kernel", 3) !=
            std::vector<int32_t>({2, 16, 16}) ||
        extract_json_int_array(ctx.config_json, "ref2va_image_vision_patch_stride", 3) !=
            std::vector<int32_t>({2, 16, 16}) ||
        extract_json_int_array(ctx.config_json, "ref2va_image_vision_patch_output_shape", 2) !=
            std::vector<int32_t>({-1, 1152}))
        throw std::runtime_error("MiniMax-H3 Ref2VA bundle has an incompatible patch plugin ABI");
    for (const char* name :
         {"processor/preprocessor_config.json", "processor/video_preprocessor_config.json"}) {
        const auto* section = find_section(ctx.bundle, name);
        if (section == nullptr || section->empty())
            throw std::runtime_error(std::string("MiniMax-H3 Ref2VA bundle is missing ") + name);
    }
}

void validate_profile(const PipelineContext& ctx) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("MiniMax-H3 requires the TensorRT backend");
    if (extract_json_int(ctx.config_json, "context_parallel_size", 1) != 1)
        throw std::runtime_error("MiniMax-H3 requires context_parallel_size=1");
    if (extract_json_int(ctx.config_json, "padded_sequence_length", 38247) != 38247)
        throw std::runtime_error("MiniMax-H3 requires 38247 unpadded sequence rows");
    if (extract_json_int(ctx.config_json, "vae_tile_batch", 28) != 28)
        throw std::runtime_error("MiniMax-H3 requires vae_tile_batch=28");
    validate_audio_profile(ctx);
    const auto workflow = load_workflow(ctx);
    if (workflow == MiniMaxH3Workflow::kFl2va)
        validate_fl2va_profile(ctx);
    else if (workflow == MiniMaxH3Workflow::kRef2va)
        validate_ref2va_profile(ctx);
}

CacheConfig load_cache_config(const PipelineContext& ctx) {
    CacheConfig result;
    const std::string mode =
        extract_json_string(ctx.config_json, "denoiser_cache_mode", "monolithic");
    if (mode != "monolithic" && mode != "first_block")
        throw std::runtime_error("MiniMax-H3 bundle has an invalid denoiser_cache_mode");
    result.enabled = extract_json_bool(ctx.config_json, "first_block_cache", false);
    if (result.enabled != (mode == "first_block"))
        throw std::runtime_error("MiniMax-H3 bundle cache mode and profile flag disagree");
    result.threshold =
        extract_json_float(ctx.config_json, "first_block_cache_threshold", result.threshold);
    if (ctx.runtime_config != nullptr &&
        ctx.runtime_config->source_of("minimax_h3", "first_block_cache_threshold") !=
            config::Layer::SchemaDefault) {
        result.threshold = static_cast<float>(
            ctx.runtime_config->get<double>("minimax_h3", "first_block_cache_threshold"));
    }
    if (!std::isfinite(result.threshold) || result.threshold <= 0.0F)
        throw std::runtime_error(
            "MiniMax-H3 first_block_cache_threshold must be finite and positive");
    return result;
}

MiniMaxH3ModuleLoader make_module_loader(const PipelineContext& ctx, SectionMap sections) {
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    return [sections = std::move(sections), bundle_path, runtime_cache, backend,
            cuda_graphs](const std::string& name, cudaStream_t stream) {
        const auto it = sections.find(name);
        if (it == sections.end())
            throw std::runtime_error("Unknown MiniMax-H3 plan section: " + name);
        auto plan = ReadBundleSection(bundle_path, it->second);
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache.c_str();
        // The exact audio encoder lazily copies and loads its embedded
        // TorchScript module on first enqueue, which is not CUDA-capture safe.
        // It is a one-shot conditioner plan; all other plans keep the caller's
        // CUDA-graph policy.
        options.cuda_graphs = cuda_graphs && name != "audio_vae_encoder_plan";
        return backend->create_module(plan.data(), plan.size(), options);
    };
}

MiniMaxH3ProfileModuleLoader make_profile_module_loader(const PipelineContext& ctx,
                                                        SectionMap sections) {
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    return [sections = std::move(sections), bundle_path, runtime_cache, backend,
            cuda_graphs](const std::string& name, cudaStream_t stream, int32_t profile) {
        const auto it = sections.find(name);
        if (it == sections.end())
            throw std::runtime_error("Unknown MiniMax-H3 profiled plan section: " + name);
        auto plan = ReadBundleSection(bundle_path, it->second);
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache.c_str();
        options.cuda_graphs = cuda_graphs;
        options.optimization_profile = profile;
        return backend->create_module(plan.data(), plan.size(), options);
    };
}

} // namespace

class MiniMaxH3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        validate_profile(ctx);
        const MiniMaxH3Workflow workflow = load_workflow(ctx);
        if (workflow == MiniMaxH3Workflow::kRef2va)
            load_minimax_h3_native_plugin(ctx);
        const CacheConfig cache = load_cache_config(ctx);
        auto sections = index_sections(ctx.bundle.info, cache.enabled, workflow);
        auto loader = make_module_loader(ctx, sections);
        auto profile_loader = make_profile_module_loader(ctx, std::move(sections));
        return std::make_unique<MiniMaxH3Pipeline>(
            std::move(loader), load_tokenizer(ctx.bundle), ctx.bundle.info.model_id, cache.enabled,
            cache.threshold, workflow, std::move(profile_loader));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_minimax_h3_plugin, MiniMaxH3Plugin,
                                       "diffusion_minimax_h3");

} // namespace trtmc
