/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/pipeline.h"
#include "families/minimax_h3/runtime/plan_loader.h"
#include "families/minimax_h3/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::minimax_h3_factory {
namespace {

std::unique_ptr<ITokenizer> load_tokenizer(const MiniMaxH3PlanSource& source) {
    const auto data = source.read_required("tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-H3 tokenizer.json is not its required BPE tokenizer");
    return tokenizer;
}

std::vector<std::string> plan_names(bool first_block_cache) {
    if (first_block_cache)
        return {"text_encoder.plan",  "adaln.plan",           "denoiser.head.plan",
                "denoiser.tail.plan", "denoiser.finish.plan", "vae.plan"};
    return {"text_encoder.plan", "adaln.plan", "denoiser.plan", "vae.plan"};
}

} // namespace
} // namespace trtmc::minimax_h3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("minimax_h3 does not support --kv-cache-size");
    using namespace trtmc;
    const auto source = std::make_shared<MiniMaxH3PlanSource>(context.reader);
    const auto runtime = source->read_required("runtime.json");
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
        make_minimax_h3_plan_loader(context.backend, source, minimax_h3_factory::plan_names(cache)),
        minimax_h3_factory::load_tokenizer(*source), "", cache, threshold);
}
