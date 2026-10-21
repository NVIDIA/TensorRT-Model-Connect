/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/clip/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::clip {
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
        throw std::runtime_error(std::string("clip runtime.json has invalid ") + name);
    return values;
}

ClipRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    ClipRuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.max_text_len = json.at("max_text_len").get<std::int32_t>();
    config.projection_dim = json.at("projection_dim").get<std::int32_t>();
    config.logit_scale = json.at("logit_scale").get<float>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.preprocess.input_image_h != config.preprocess.input_image_w ||
        config.max_text_len <= 0 || config.projection_dim <= 0 || config.logit_scale <= 0.0F) {
        throw std::runtime_error("clip runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("clip ") + what + " engine failed to load");
    return engine;
}

std::string require_task(const BundleReader& bundle) {
    const auto& task = bundle.info().task;
    if (task != internal::IImageToEmbedding::kTask && task != internal::ITextToEmbedding::kTask &&
        task != internal::ITextImageToRelevance::kTask) {
        throw std::runtime_error("clip bundle declares an unsupported task: " + task);
    }
    return task;
}

} // namespace
} // namespace trtmc::clip

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("clip does not support --kv-cache-size");
    const auto config_data = trtmc::clip::require_section(context.reader, "runtime.json");
    const auto vision_plan = trtmc::clip::require_section(context.reader, "vision.plan");
    const auto text_plan = trtmc::clip::require_section(context.reader, "text.plan");
    const auto tokenizer_data = trtmc::clip::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::clip::parse_config(config_data);
    auto primary_task = trtmc::clip::require_task(context.reader);
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                               /*add_special_tokens=*/true);
    if (!tokenizer)
        throw std::runtime_error("clip tokenizer failed to load");
    auto vision = trtmc::clip::load_engine(context.backend, vision_plan, "vision");
    auto text = trtmc::clip::load_engine(context.backend, text_plan, "text");
    return new trtmc::ClipPipeline(std::move(vision), std::move(text),
                                   std::shared_ptr<trtmc::ITokenizer>(std::move(tokenizer)),
                                   std::move(config), std::move(primary_task));
}
