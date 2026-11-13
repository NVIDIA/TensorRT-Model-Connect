/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/grounding_dino/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::grounding_dino {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::array<float, 3> triple(const nlohmann::json& json, const char* name) {
    const auto values = json.at(name).get<std::vector<float>>();
    if (values.size() != 3)
        throw std::runtime_error(std::string("grounding_dino runtime.json has invalid ") + name);
    return {values[0], values[1], values[2]};
}

GroundingDinoRuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    GroundingDinoRuntimeConfig config;
    config.preprocess.input_image_h = json.at("input_image_h").get<std::int32_t>();
    config.preprocess.input_image_w = json.at("input_image_w").get<std::int32_t>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.max_text_len = json.at("max_text_len").get<std::int32_t>();
    config.num_queries = json.at("num_queries").get<std::int32_t>();
    config.d_model = json.at("d_model").get<std::int32_t>();
    config.box_threshold = json.at("box_threshold").get<float>();
    config.text_threshold = json.at("text_threshold").get<float>();
    if (config.preprocess.input_image_h <= 0 || config.preprocess.input_image_w <= 0 ||
        config.max_text_len <= 0 || config.num_queries <= 0 || config.d_model <= 0 ||
        config.d_model % 2 != 0 || config.box_threshold < 0.0F || config.box_threshold > 1.0F ||
        config.text_threshold < 0.0F || config.text_threshold > 1.0F) {
        throw std::runtime_error("grounding_dino runtime.json does not match its runtime contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("grounding_dino engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::grounding_dino

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("grounding_dino does not support --kv-cache-size");
    const auto config_data = trtmc::grounding_dino::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::grounding_dino::require_section(context.reader, "detector.plan");
    const auto tokenizer_data =
        trtmc::grounding_dino::require_section(context.reader, "tokenizer.json");

    auto config = trtmc::grounding_dino::parse_config(config_data);
    auto tokenizer = trtmc::CreateWordPieceTokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                                     /*add_special_tokens=*/true);
    if (!tokenizer)
        throw std::runtime_error("grounding_dino tokenizer failed to load");
    auto engine = trtmc::grounding_dino::load_engine(context.backend, plan);
    return new trtmc::GroundingDinoPipeline(
        std::move(engine), std::shared_ptr<trtmc::ITokenizer>(std::move(tokenizer)),
        std::move(config));
}
