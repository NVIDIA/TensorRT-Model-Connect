/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::siglip2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::vector<float> triple(const nlohmann::json& json, const char* name) {
    auto values = json.at(name).get<std::vector<float>>();
    if (values.size() != 3)
        throw std::runtime_error(std::string("siglip2 runtime.json has invalid ") + name);
    return values;
}

Siglip2RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Siglip2RuntimeConfig config;
    config.preprocess.patch_size = json.at("patch_size").get<std::int32_t>();
    config.preprocess.max_num_patches = json.at("max_num_patches").get<std::int32_t>();
    config.preprocess.position_grid = json.at("position_grid").get<std::int32_t>();
    config.preprocess.rescale_factor = json.at("rescale_factor").get<double>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.resample = json.at("resample").get<std::int32_t>();
    config.max_text_len = json.at("max_text_len").get<std::int32_t>();
    config.embedding_dim = json.at("embedding_dim").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    config.logit_scale = json.at("logit_scale").get<float>();
    config.logit_bias = json.at("logit_bias").get<float>();
    if (config.preprocess.patch_size <= 0 || config.preprocess.max_num_patches <= 0 ||
        config.preprocess.position_grid <= 0 || !(config.preprocess.rescale_factor > 0.0) ||
        config.max_text_len <= 0 || config.embedding_dim <= 0 || config.pad_token_id < 0 ||
        !(config.logit_scale > 0.0F)) {
        throw std::runtime_error("siglip2 runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("siglip2 ") + what + " engine failed to load");
    return engine;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IImageToEmbedding::kTask && task != internal::ITextToEmbedding::kTask &&
        task != internal::ITextImageToRelevance::kTask) {
        throw std::runtime_error("siglip2 bundle declares an unsupported task: " + task);
    }
    return task;
}

} // namespace
} // namespace trtmc::siglip2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("siglip2 does not support --kv-cache-size");
    const auto config_data = trtmc::siglip2::require_section(context.reader, "runtime.json");
    const auto vision_plan = trtmc::siglip2::require_section(context.reader, "vision.plan");
    const auto text_plan = trtmc::siglip2::require_section(context.reader, "text.plan");
    const auto tokenizer_data = trtmc::siglip2::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::siglip2::parse_config(config_data);
    auto primary_task = trtmc::siglip2::require_task(context.reader);
    auto tokenizer = trtmc::CreateSiglip2Tokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                                   /*add_special_tokens=*/true);
    if (!tokenizer)
        throw std::runtime_error("siglip2 tokenizer failed to load");
    auto vision = trtmc::siglip2::load_engine(context.backend, vision_plan, "vision");
    auto text = trtmc::siglip2::load_engine(context.backend, text_plan, "text");
    return new trtmc::Siglip2Pipeline(std::move(vision), std::move(text),
                                      std::shared_ptr<trtmc::ITokenizer>(std::move(tokenizer)),
                                      std::move(config), std::move(primary_task));
}
