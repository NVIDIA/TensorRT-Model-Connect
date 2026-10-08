/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/pipeline.h"
#include "families/minimax_h3/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::minimax_h3_factory {
namespace {

using PlanMap = std::unordered_map<std::string, std::vector<char>>;

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-H3 tokenizer.json is not its required BPE tokenizer");
    return tokenizer;
}

PlanMap load_plans(const BundleReader& bundle, bool first_block_cache) {
    constexpr std::array<const char*, 5> monolithic = {
        "text_encoder.plan", "adaln.plan", "denoiser.plan", "vae.plan", "audio_vae.plan"};
    constexpr std::array<const char*, 7> split = {
        "text_encoder.plan",    "adaln.plan", "denoiser.head.plan", "denoiser.tail.plan",
        "denoiser.finish.plan", "vae.plan",   "audio_vae.plan"};
    PlanMap plans;
    if (first_block_cache) {
        for (const char* name : split)
            plans.emplace(name, require_section(bundle, name));
    } else {
        for (const char* name : monolithic)
            plans.emplace(name, require_section(bundle, name));
    }
    return plans;
}

MiniMaxH3ModuleLoader make_loader(IBackend& backend, PlanMap plans,
                                  std::optional<std::uint64_t> text_encoder_weight_budget,
                                  std::optional<std::uint64_t> denoiser_weight_budget) {
    return [&backend, plans = std::move(plans), text_encoder_weight_budget,
            denoiser_weight_budget](const std::string& name, cudaStream_t stream) {
        const auto found = plans.find(name);
        if (found == plans.end())
            throw std::runtime_error("MiniMax-H3 requested undeclared plan: " + name);
        ModuleCreateOptions options{};
        options.stream = stream;
        if (name == "text_encoder.plan")
            options.weight_streaming_budget_bytes = text_encoder_weight_budget;
        if (name == "denoiser.plan")
            options.weight_streaming_budget_bytes = denoiser_weight_budget;
        auto module = backend.create_module(found->second.data(), found->second.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("MiniMax-H3 failed to load plan: " + name);
        return module;
    };
}

std::optional<std::uint64_t> optional_weight_budget(const nlohmann::json& config,
                                                    const char* field) {
    if (!config.contains(field) || config.at(field).is_null())
        return std::nullopt;
    const auto& value = config.at(field);
    if (!value.is_number_unsigned() &&
        (!value.is_number_integer() || value.get<std::int64_t>() < 0)) {
        throw std::runtime_error(std::string("MiniMax-H3 ") + field + " must be non-negative");
    }
    return value.get<std::uint64_t>();
}

} // namespace
} // namespace trtmc::minimax_h3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("minimax_h3 does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime = minimax_h3_factory::require_section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime.begin(), runtime.end());
    if (config.at("context_parallel_size").get<std::int32_t>() != 1 ||
        config.at("padded_sequence_length").get<std::int32_t>() != 38247 ||
        config.at("vae_tile_batch").get<std::int32_t>() != 28 ||
        config.at("audio_sample_rate").get<std::int32_t>() != 32000 ||
        config.at("audio_channels").get<std::int32_t>() != 2 ||
        config.at("audio_samples_per_channel").get<std::int32_t>() != 165600) {
        throw std::runtime_error("MiniMax-H3 runtime.json declares an unsupported profile");
    }
    const bool cache = config.at("first_block_cache").get<bool>();
    const auto mode = config.at("denoiser_cache_mode").get<std::string>();
    if ((cache && mode != "first_block") || (!cache && mode != "monolithic"))
        throw std::runtime_error("MiniMax-H3 cache mode is inconsistent");
    const float threshold = config.at("first_block_cache_threshold").get<float>();
    if (!std::isfinite(threshold) || threshold <= 0.0F)
        throw std::runtime_error("MiniMax-H3 cache threshold must be finite and positive");
    const auto shared_weight_budget =
        minimax_h3_factory::optional_weight_budget(config, "weight_streaming_budget_bytes");
    auto text_encoder_weight_budget = minimax_h3_factory::optional_weight_budget(
        config, "text_encoder_weight_streaming_budget_bytes");
    auto denoiser_weight_budget = minimax_h3_factory::optional_weight_budget(
        config, "denoiser_weight_streaming_budget_bytes");
    if (!text_encoder_weight_budget)
        text_encoder_weight_budget = shared_weight_budget;
    if (!denoiser_weight_budget)
        denoiser_weight_budget = shared_weight_budget;
    MiniMaxH3GenerationConfig generation;
    generation.profile = config.value("generation_profile", "minimax-h3-base");
    generation.num_inference_steps = config.at("num_inference_steps").get<std::int32_t>();
    generation.video_scheduler_shift = config.value("video_scheduler_shift", 12.0F);
    generation.audio_scheduler_shift = config.value("audio_scheduler_shift", 3.0F);
    generation.dmd_denoising_steps =
        config.value("dmd_denoising_steps", std::vector<std::int32_t>{});
    const bool base_profile =
        generation.profile == "minimax-h3-base" && generation.num_inference_steps == 50 &&
        generation.video_scheduler_shift == 12.0F && generation.audio_scheduler_shift == 3.0F &&
        generation.dmd_denoising_steps.empty();
    const bool fast_profile =
        generation.profile == "fasth3-dense-4step" && generation.num_inference_steps == 5 &&
        generation.video_scheduler_shift == 12.0F && generation.audio_scheduler_shift == 3.0F &&
        generation.dmd_denoising_steps == std::vector<std::int32_t>{999, 749, 500, 250};
    const bool fast_vsa_profile =
        generation.profile == "fasth3-vsa-4step" && generation.num_inference_steps == 5 &&
        generation.video_scheduler_shift == 12.0F && generation.audio_scheduler_shift == 3.0F &&
        generation.dmd_denoising_steps == std::vector<std::int32_t>{999, 749, 500, 250} &&
        config.value("attention_backend", "") == "VIDEO_SPARSE_ATTN_H3" &&
        config.value("vsa_tile_size", 0) == 64 && config.value("vsa_sparsity", 0.0F) == 0.9F &&
        config.value("vsa_kernel", "") == "sm100a";
    if (!base_profile && !fast_profile && !fast_vsa_profile)
        throw std::runtime_error(
            "MiniMax-H3 runtime.json declares an unsupported generation profile");
    if (config.value("transformer_forwards", generation.num_inference_steps - 1) !=
        generation.num_inference_steps - 1)
        throw std::runtime_error("MiniMax-H3 transformer forward count is inconsistent");
    return new MiniMaxH3Pipeline(
        minimax_h3_factory::make_loader(context.backend,
                                        minimax_h3_factory::load_plans(context.reader, cache),
                                        text_encoder_weight_budget, denoiser_weight_budget),
        minimax_h3_factory::load_tokenizer(context.reader), "", std::move(generation), cache,
        threshold);
}
