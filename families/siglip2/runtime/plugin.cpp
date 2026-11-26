/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/siglip2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <cstdint>
#include <limits>
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

// The fields of runtime.json are taken only with their own JSON type and range: a fraction, a
// bool, a string or a wider value is refused, never converted into the field.
std::int32_t int32_field(const nlohmann::json& json, const char* name) {
    const auto& value = json.at(name);
    const bool fits = value.is_number_unsigned()
                          ? value.get<std::uint64_t>() <= static_cast<std::uint64_t>(INT32_MAX)
                          : value.is_number_integer() && value.get<std::int64_t>() >= INT32_MIN &&
                                value.get<std::int64_t>() <= INT32_MAX;
    if (!fits)
        throw std::runtime_error(std::string("siglip2 runtime.json ") + name +
                                 " is not an int32 integer");
    return static_cast<std::int32_t>(value.get<std::int64_t>());
}

template <typename T>
T number_field(const nlohmann::json& value, const char* name) {
    if (!value.is_number() ||
        !(std::fabs(value.get<double>()) <= static_cast<double>(std::numeric_limits<T>::max())))
        throw std::runtime_error(std::string("siglip2 runtime.json ") + name +
                                 " is not a finite number in range");
    return static_cast<T>(value.get<double>());
}

std::vector<float> triple(const nlohmann::json& json, const char* name) {
    const auto& values = json.at(name);
    if (!values.is_array() || values.size() != 3)
        throw std::runtime_error(std::string("siglip2 runtime.json has invalid ") + name);
    std::vector<float> result;
    for (const auto& value : values)
        result.push_back(number_field<float>(value, name));
    return result;
}

} // namespace

Siglip2RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Siglip2RuntimeConfig config;
    config.preprocess.patch_size = int32_field(json, "patch_size");
    config.preprocess.max_num_patches = int32_field(json, "max_num_patches");
    config.preprocess.position_grid = int32_field(json, "position_grid");
    config.preprocess.rescale_factor =
        number_field<double>(json.at("rescale_factor"), "rescale_factor");
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.resample = int32_field(json, "resample");
    config.max_text_len = int32_field(json, "max_text_len");
    config.embedding_dim = int32_field(json, "embedding_dim");
    config.pad_token_id = int32_field(json, "pad_token_id");
    config.logit_scale = number_field<float>(json.at("logit_scale"), "logit_scale");
    config.logit_bias = number_field<float>(json.at("logit_bias"), "logit_bias");
    if (config.preprocess.patch_size <= 0 || config.preprocess.max_num_patches <= 0 ||
        config.preprocess.position_grid <= 0 || !(config.preprocess.rescale_factor > 0.0) ||
        config.max_text_len <= 0 || config.embedding_dim <= 0 || config.pad_token_id < 0 ||
        !(config.logit_scale > 0.0F)) {
        throw std::runtime_error("siglip2 runtime.json does not match its runtime contract");
    }
    for (const float deviation : config.preprocess.image_std) {
        if (deviation == 0.0F)
            throw std::runtime_error("siglip2 runtime.json image_std must be nonzero");
    }
    return config;
}

namespace {

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

    auto config = trtmc::siglip2::parse_runtime_config(config_data);
    auto primary_task = trtmc::siglip2::require_task(context.reader);
    auto tokenizer =
        trtmc::siglip2::CreateSiglip2Tokenizer(tokenizer_data.data(), tokenizer_data.size(),
                                               /*add_special_tokens=*/true);
    if (!tokenizer)
        throw std::runtime_error("siglip2 tokenizer failed to load");
    auto vision = trtmc::siglip2::load_engine(context.backend, vision_plan, "vision");
    auto text = trtmc::siglip2::load_engine(context.backend, text_plan, "text");
    return new trtmc::Siglip2Pipeline(
        std::move(vision), std::move(text),
        std::shared_ptr<trtmc::siglip2::ITokenizer>(std::move(tokenizer)), std::move(config),
        std::move(primary_task));
}
