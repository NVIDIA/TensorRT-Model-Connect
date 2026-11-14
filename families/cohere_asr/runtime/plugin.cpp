/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere_asr/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace {
std::vector<char> section(const trtmc::BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length)
        throw std::runtime_error(std::string("cohere_asr missing section ") + name);
    return reader.read_section(name);
}
std::vector<float> floats(const std::vector<char>& data) {
    if (data.size() % sizeof(float))
        throw std::runtime_error("invalid cohere_asr frontend buffer");
    std::vector<float> result(data.size() / sizeof(float));
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}
} // namespace
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("cohere_asr does not support --kv-cache-size");
    auto raw = section(context.reader, "runtime.json");
    auto metadata = nlohmann::json::parse(raw.begin(), raw.end());
    trtmc::cohere_asr::Config config;
    config.max_samples = metadata.at("max_samples");
    config.max_tokens = metadata.at("max_tokens");
    config.eos_id = metadata.at("eos_id");
    config.arabic_prompt = metadata.at("arabic_prompt").get<std::vector<int32_t>>();
    config.english_prompt = metadata.at("english_prompt").get<std::vector<int32_t>>();
    if (config.max_samples != 560000 || config.max_tokens < 16 || config.max_tokens > 1024 ||
        config.eos_id != 3 || config.arabic_prompt.size() != 9 ||
        config.english_prompt.size() != 9 || metadata.at("sample_rate") != 16000)
        throw std::runtime_error("cohere_asr runtime contract mismatch");
    config.window = floats(section(context.reader, "window.f32"));
    config.filters = floats(section(context.reader, "mel_filters.f32"));
    raw = section(context.reader, "tokens.json");
    const auto tokenizer = nlohmann::json::parse(raw.begin(), raw.end());
    config.pieces = tokenizer.at("pieces").get<std::vector<std::string>>();
    for (auto id : tokenizer.at("special_ids"))
        config.special.insert(id.get<int32_t>());
    if (config.pieces.size() != 16384 || config.window.size() != 400 ||
        config.filters.size() != 128 * 257)
        throw std::runtime_error("cohere_asr vocabulary or frontend shape mismatch");
    auto encoder_plan = section(context.reader, "encoder.plan");
    auto decoder_plan = section(context.reader, "decoder.plan");
    trtmc::ModuleCreateOptions options{};
    auto encoder = context.backend.create_module(encoder_plan.data(), encoder_plan.size(), options);
    auto decoder = context.backend.create_module(decoder_plan.data(), decoder_plan.size(), options);
    return new trtmc::cohere_asr::Pipeline(std::move(encoder), std::move(decoder),
                                           std::move(config));
}
