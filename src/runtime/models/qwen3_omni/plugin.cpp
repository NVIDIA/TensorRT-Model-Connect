/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Qwen3OmniPlugin: handles "qwen3_omni_multimodal" strategy.
// Omni pipeline with TensorRT Thinker/Code2Wav and the official model-owned Talker bridge.

#include "plugin_helpers.h"
#include "runtime/models/qwen3_omni/pipeline.h"
#include "runtime/models/qwen3_omni/talker_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream output;
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    output << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / kGiB << " GiB";
    return output.str();
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error("Qwen3-Omni native KV byte accounting overflow");
    return lhs * rhs;
}

void validate_scalar(const ITrtModule& module, const char* name, const char* role) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error(std::string("Qwen3-Omni ") + role + " native KV input '" + name +
                                 "' must be int32 [1]");
    }
}

void validate_native_module(const PipelineContext& ctx, const ITrtModule& module,
                            const char* role) {
    validate_scalar(module, "cache_write_indices", role);
    validate_scalar(module, "key_value_lengths", role);
    if (module.has_input("attention_mask"))
        throw std::runtime_error(std::string("Qwen3-Omni ") + role +
                                 " must not expose attention_mask");
    const int32_t head_dim = ctx.config.head_dim;
    const std::vector<int64_t> expected{1, ctx.config.num_kv_heads, ctx.config.max_cache_length,
                                        head_dim};
    for (int32_t layer = 0; layer < ctx.config.num_layers; ++layer) {
        const auto suffix = "_" + std::to_string(layer);
        for (const auto& pair :
             {std::pair{std::string("cache_k") + suffix, std::string("present_k") + suffix},
              std::pair{std::string("cache_v") + suffix, std::string("present_v") + suffix}}) {
            if (!module.has_input(pair.first) || !module.has_output(pair.second) ||
                module.tensor_shape(pair.first) != expected ||
                module.tensor_shape(pair.second) != expected ||
                module.tensor_dtype(pair.first) != DType::kBFloat16 ||
                module.tensor_dtype(pair.second) != DType::kBFloat16) {
                throw std::runtime_error(std::string("Qwen3-Omni ") + role +
                                         " native KV cache/present must be BF16 "
                                         "[1,num_kv_heads,capacity,head_dim]");
            }
        }
    }
}

std::uint64_t native_cache_bytes(const PipelineContext& ctx, int32_t kv_dim) {
    auto bytes = checked_multiply(static_cast<std::uint64_t>(ctx.config.num_layers),
                                  static_cast<std::uint64_t>(ctx.config.max_cache_length));
    bytes = checked_multiply(bytes, static_cast<std::uint64_t>(kv_dim));
    bytes = checked_multiply(bytes, 2); // BF16 bytes.
    return checked_multiply(bytes, 2);  // K and V.
}

void admit_cache_allocation(const PipelineContext& ctx, std::uint64_t bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen3-Omni CUDA memory query failed: ") +
                                 cudaGetErrorString(status));
    }
    const auto free = static_cast<std::uint64_t>(free_bytes);
    const auto total = static_cast<std::uint64_t>(total_bytes);
    const auto reserve = std::max<std::uint64_t>(2ULL << 30, total / 10);
    const auto available = free > reserve ? free - reserve : 0;
    if (bytes > available) {
        throw std::runtime_error(
            "Qwen3-Omni native KV admission failed before allocation: capacity=" +
            std::to_string(ctx.config.max_cache_length) + ", required=" + format_bytes(bytes) +
            ", free=" + format_bytes(free) + ", reserve=" + format_bytes(reserve));
    }
}

void validate_native_bundle_contract(const PipelineContext& ctx) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error("Qwen3-Omni bundle does not declare native KV contract version 1");
    }
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::invalid_argument("Qwen3-Omni allocates the complete official context cache; "
                                    "kv_cache_size_bytes is not supported");
    }
}

std::unique_ptr<TrtModule> load_required_code2wav(const PipelineContext& ctx,
                                                  const ModuleCreateOptions& options) {
    auto loaded = try_load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "code2wav_engine_plan"), "code2wav", options);
    if (!loaded.module || !loaded.module->ok()) {
        throw std::runtime_error(
            "OmniPipeline: required official Code2Wav engine is missing from bundle");
    }
    return std::move(loaded.module);
}

std::unique_ptr<TrtModule> load_resident_component(const PipelineContext& ctx,
                                                   const ModuleCreateOptions& options,
                                                   const char* section_name, const char* label) {
    const auto* plan = find_section(ctx.bundle, section_name);
    if (plan == nullptr || plan->empty())
        return {};
    return load_trt_module_from_plan(ctx.backend, plan, label, options).module;
}

OmniConfig build_omni_config(const PipelineContext& ctx) {
    const auto& json = ctx.config_json;
    OmniConfig config;
    config.sample_rate = extract_json_int(json, "audio_sample_rate", 24000);
    config.thinker_hidden_size = ctx.config.hidden_size;
    config.thinker_vocab_size = ctx.config.vocab_size;
    config.thinker_num_layers = ctx.config.num_layers;
    config.thinker_num_heads = ctx.config.num_heads;
    config.thinker_eos_token_id = extract_json_int(json, "im_end_token_id", 151645);
    config.num_experts = extract_json_int(json, "num_local_experts", 8);
    config.num_experts_per_tok = extract_json_int(json, "num_experts_per_tok", 2);
    config.talker_hidden_size = extract_json_int(json, "omni_talker_hidden_size", 0);
    config.talker_num_layers = extract_json_int(json, "omni_talker_num_layers", 0);
    config.talker_n_codebooks = extract_json_int(json, "omni_n_codebooks", 16);
    config.talker_codebook_size = extract_json_int(json, "omni_codebook_size", 2048);
    config.code2wav_max_frames = extract_json_int(json, "omni_code2wav_max_frames", 32);
    config.code2wav_upsample_factor = extract_json_int(json, "omni_code2wav_upsample_factor", 1920);
    config.code2wav_output_delay = extract_json_int(json, "omni_code2wav_output_delay", 555);
    config.hf_python = ctx.hf_python;
    config.talker_model_id = extract_json_string(
        json, "omni_talker_model_id",
        extract_json_string(json, "omni_talker_model_path", ctx.bundle.info.model_id));
    config.talker_model_revision = extract_json_string(json, "omni_talker_model_revision", "");
    if (const char* override_path = std::getenv("TRTMC_QWEN3_OMNI_MODEL_PATH");
        override_path != nullptr && override_path[0] != '\0') {
        config.talker_model_id = override_path;
        config.talker_model_revision.clear();
    }
    return config;
}

std::unique_ptr<Qwen3OmniInferenceState>
allocate_thinker_state(const PipelineContext& ctx, int32_t kv_dim, cudaStream_t stream) {
    const auto cache_bytes = native_cache_bytes(ctx, kv_dim);
    admit_cache_allocation(ctx, cache_bytes);
    std::cerr << "[trtmc] Qwen3-Omni native KV cache capacity=" << ctx.config.max_cache_length
              << " tokens, allocation=" << format_bytes(cache_bytes) << '\n';
    std::unique_ptr<Qwen3OmniInferenceState> state = std::make_unique<Qwen3OmniKvCache>(
        ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream, DType::kBFloat16);
    if (!state->ok())
        throw std::runtime_error("OmniPipeline: failed to allocate native Thinker KV cache");
    return state;
}

} // namespace

class Qwen3OmniPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        validate_native_bundle_contract(ctx);

        auto decode_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "omni thinker decode", opts);
        cudaStream_t stream = decode_loaded.module->stream();
        auto prefill_opts = opts;
        prefill_opts.stream = stream;
        auto prefill_loaded =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "prefill_engine_plan"),
                                      "omni thinker prefill", prefill_opts);
        validate_native_module(ctx, *decode_loaded.module, "decode engine");
        validate_native_module(ctx, *prefill_loaded.module, "prefill engine");
        int32_t kv_dim = compute_kv_dim(ctx.config);

        auto code2wav_module = load_required_code2wav(ctx, prefill_opts);
        auto vision_module =
            load_resident_component(ctx, prefill_opts, "vision_engine_plan", "omni vision encoder");
        auto audio_encoder_module =
            load_resident_component(ctx, prefill_opts, "audio_encoder_plan", "omni audio encoder");

        auto omni_cfg = build_omni_config(ctx);

        auto talker_runtime = std::make_unique<Qwen3OmniTalkerRuntime>(
            omni_cfg.hf_python, omni_cfg.talker_model_id, omni_cfg.talker_model_revision,
            omni_cfg.talker_n_codebooks, omni_cfg.code2wav_max_frames);
        talker_runtime->start();

        // Thinker, Code2Wav, optional vision/audio engines, and the official
        // Talker CUDA model are all resident now. Size the complete native KV
        // cache against the actual remaining device memory, never a stale
        // pre-component estimate.
        auto thinker_state = allocate_thinker_state(ctx, kv_dim, stream);

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<OmniPipeline>(
            std::move(decode_loaded.module), std::move(thinker_state), std::move(code2wav_module),
            std::move(omni_cfg), stream, std::move(tokenizer), ctx.bundle.info.model_id,
            std::move(prefill_loaded.module), std::move(talker_runtime), std::move(vision_module),
            std::move(audio_encoder_module));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_omni_plugin, Qwen3OmniPlugin,
                                       "qwen3_omni_multimodal");

} // namespace trtmc
