/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ast/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::ast {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::vector<float> float_section(const BundleReader& bundle, const char* name,
                                 std::size_t expected) {
    const auto raw = require_section(bundle, name);
    if (raw.size() != expected * sizeof(float))
        throw std::runtime_error(std::string("ast bundle section has the wrong length: ") + name);
    std::vector<float> values(expected);
    std::memcpy(values.data(), raw.data(), raw.size());
    return values;
}

AstRuntimeConfig parse_config(const std::vector<char>& data, const BundleReader& bundle) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    AstRuntimeConfig config;
    auto& fbank = config.fbank;
    fbank.sample_rate = json.at("sample_rate").get<std::int32_t>();
    fbank.frame_length = json.at("frame_length").get<std::int32_t>();
    fbank.frame_shift = json.at("frame_shift").get<std::int32_t>();
    fbank.n_fft = json.at("n_fft").get<std::int32_t>();
    fbank.num_mel_bins = json.at("num_mel_bins").get<std::int32_t>();
    fbank.max_length = json.at("max_length").get<std::int32_t>();
    fbank.mel_floor = json.at("mel_floor").get<float>();
    fbank.preemphasis = json.at("preemphasis").get<float>();
    fbank.feature_mean = json.at("feature_mean").get<float>();
    fbank.feature_scale = json.at("feature_scale").get<float>();
    if (fbank.sample_rate <= 0 || fbank.frame_length <= 1 || fbank.frame_shift <= 0 ||
        fbank.n_fft < fbank.frame_length || fbank.num_mel_bins <= 0 || fbank.max_length <= 0) {
        throw std::runtime_error("ast runtime.json does not match its runtime contract");
    }
    config.labels = json.at("labels").get<std::vector<std::string>>();
    if (config.labels.empty())
        throw std::runtime_error("ast runtime.json carries no class labels");
    for (const auto& label : config.labels)
        if (label.empty())
            throw std::runtime_error("ast runtime.json carries an empty class label");

    fbank.window =
        float_section(bundle, "window.f32", static_cast<std::size_t>(fbank.frame_length));
    fbank.mel_filters = float_section(bundle, "mel_filters.f32",
                                      static_cast<std::size_t>(fbank.num_mel_bins) *
                                          static_cast<std::size_t>(fbank.n_fft / 2 + 1));
    return config;
}

} // namespace
} // namespace trtmc::ast

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("ast does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::IAudioToClassScores::kTask)
        throw std::runtime_error("ast bundle declares an unsupported task: " + task);

    const auto config_data = trtmc::ast::require_section(context.reader, "runtime.json");
    const auto plan = trtmc::ast::require_section(context.reader, "model.plan");
    auto config = trtmc::ast::parse_config(config_data, context.reader);

    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("ast engine failed to load");
    return new trtmc::AstPipeline(std::move(engine), std::move(config));
}
