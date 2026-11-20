/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wav2vec2/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::wav2vec2 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

Wav2Vec2RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    Wav2Vec2RuntimeConfig config;
    config.sample_rate = json.at("sample_rate").get<std::int32_t>();
    config.do_normalize = json.at("do_normalize").get<bool>();
    config.conv_kernel = json.at("conv_kernel").get<std::vector<std::int32_t>>();
    config.conv_stride = json.at("conv_stride").get<std::vector<std::int32_t>>();
    config.min_samples = json.at("min_samples").get<std::int64_t>();
    config.max_samples = json.at("max_samples").get<std::int64_t>();
    config.vocabulary.tokens = json.at("tokens").get<std::vector<std::string>>();
    config.vocabulary.blank_token_id = json.at("blank_token_id").get<std::int32_t>();
    config.vocabulary.word_delimiter_token = json.at("word_delimiter_token").get<std::string>();

    if (config.sample_rate <= 0)
        throw std::runtime_error("wav2vec2 runtime.json states a non-positive sample rate");
    if (config.conv_kernel.empty() || config.conv_kernel.size() != config.conv_stride.size())
        throw std::runtime_error("wav2vec2 runtime.json encoder geometry is inconsistent");
    if (config.min_samples <= 0 || config.max_samples < config.min_samples)
        throw std::runtime_error("wav2vec2 runtime.json states an empty sample range");
    if (config.vocabulary.tokens.empty())
        throw std::runtime_error("wav2vec2 runtime.json carries no vocabulary");
    const auto blank = static_cast<std::size_t>(config.vocabulary.blank_token_id);
    if (config.vocabulary.blank_token_id < 0 || blank >= config.vocabulary.tokens.size())
        throw std::runtime_error("wav2vec2 runtime.json blank id is outside the vocabulary");
    return config;
}

} // namespace
} // namespace trtmc::wav2vec2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("wav2vec2 does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ISpeechTranscription::kTask)
        throw std::runtime_error("wav2vec2 bundle declares an unsupported task: " + task);

    const auto config_data = trtmc::wav2vec2::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::wav2vec2::require_section(context.reader, "model.plan");
    auto config = trtmc::wav2vec2::parse_config(config_data);

    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("wav2vec2 engine failed to load");
    return new trtmc::Wav2Vec2Pipeline(std::move(engine), std::move(config));
}
