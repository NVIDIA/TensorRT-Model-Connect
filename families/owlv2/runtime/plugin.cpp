/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/owlv2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::owlv2 {
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
        throw std::runtime_error(std::string("owlv2 runtime.json has invalid ") + name);
    return values;
}

Owlv2RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Owlv2RuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.max_text_len = json.at("max_text_len").get<std::int32_t>();
    config.num_patches = json.at("num_patches").get<std::int32_t>();
    config.grid = json.at("grid").get<std::int32_t>();
    config.query_dim = json.at("query_dim").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.preprocess.input_image_h != config.preprocess.input_image_w ||
        config.max_text_len <= 0 || config.num_patches <= 0 || config.grid <= 0 ||
        config.query_dim <= 0 || config.pad_token_id < 0) {
        throw std::runtime_error("owlv2 runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("owlv2 ") + what + " engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::owlv2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("owlv2 does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::IImageTextToBoxes::kTask)
        throw std::runtime_error("owlv2 bundle declares an unsupported task: " + task);

    const auto config_data = trtmc::owlv2::require_section(context.reader, "runtime.json");
    const auto vision_plan = trtmc::owlv2::require_section(context.reader, "vision.plan");
    const auto text_plan = trtmc::owlv2::require_section(context.reader, "text.plan");
    const auto tokenizer_data = trtmc::owlv2::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::owlv2::parse_config(config_data);
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                               /*add_special_tokens=*/true);
    if (!tokenizer)
        throw std::runtime_error("owlv2 tokenizer failed to load");
    auto vision = trtmc::owlv2::load_engine(context.backend, vision_plan, "vision");
    auto text = trtmc::owlv2::load_engine(context.backend, text_plan, "text");
    return new trtmc::Owlv2Pipeline(std::move(vision), std::move(text),
                                    std::shared_ptr<trtmc::ITokenizer>(std::move(tokenizer)),
                                    std::move(config));
}
