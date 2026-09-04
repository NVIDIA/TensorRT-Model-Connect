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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

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
    constexpr std::array<const char*, 4> monolithic = {"text_encoder.plan", "adaln.plan",
                                                       "denoiser.plan", "vae.plan"};
    constexpr std::array<const char*, 6> split = {"text_encoder.plan",    "adaln.plan",
                                                  "denoiser.head.plan",   "denoiser.tail.plan",
                                                  "denoiser.finish.plan", "vae.plan"};
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

MiniMaxH3ModuleLoader make_loader(IBackend& backend, PlanMap plans) {
    return [&backend, plans = std::move(plans)](const std::string& name, cudaStream_t stream) {
        const auto found = plans.find(name);
        if (found == plans.end())
            throw std::runtime_error("MiniMax-H3 requested undeclared plan: " + name);
        ModuleCreateOptions options{};
        options.stream = stream;
        auto module = backend.create_module(found->second.data(), found->second.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("MiniMax-H3 failed to load plan: " + name);
        return module;
    };
}

} // namespace
} // namespace trtmc::minimax_h3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto& runtime = minimax_h3_factory::require_section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime.begin(), runtime.end());
    if (config.at("context_parallel_size").get<std::int32_t>() != 1 ||
        config.at("padded_sequence_length").get<std::int32_t>() != 38247 ||
        config.at("vae_tile_batch").get<std::int32_t>() != 28) {
        throw std::runtime_error("MiniMax-H3 runtime.json declares an unsupported profile");
    }
    const bool cache = config.at("first_block_cache").get<bool>();
    const auto mode = config.at("denoiser_cache_mode").get<std::string>();
    if ((cache && mode != "first_block") || (!cache && mode != "monolithic"))
        throw std::runtime_error("MiniMax-H3 cache mode is inconsistent");
    const float threshold = config.at("first_block_cache_threshold").get<float>();
    if (!std::isfinite(threshold) || threshold <= 0.0F)
        throw std::runtime_error("MiniMax-H3 cache threshold must be finite and positive");
    return new MiniMaxH3Pipeline(
        minimax_h3_factory::make_loader(context.backend,
                                        minimax_h3_factory::load_plans(context.reader, cache)),
        minimax_h3_factory::load_tokenizer(context.reader), "", cache, threshold);
}
