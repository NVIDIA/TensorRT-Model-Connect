/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ministral3/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::ministral3_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("ministral3 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::int64_t require_positive(const nlohmann::json& runtime, const char* name) {
    if (!runtime.contains(name) || !runtime.at(name).is_number_integer())
        throw std::invalid_argument("ministral3 runtime.json requires integer " +
                                    std::string(name));
    const auto value = runtime.at(name).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("ministral3 runtime.json has invalid " + std::string(name));
    return value;
}

std::string require_string(const nlohmann::json& runtime, const char* name) {
    if (!runtime.contains(name) || !runtime.at(name).is_string())
        throw std::invalid_argument("ministral3 runtime.json requires string " + std::string(name));
    return runtime.at(name).get<std::string>();
}

ministral3::EncoderSettings settings(const BundleReader& bundle) {
    const auto data = require_section(bundle, "runtime.json");
    const auto runtime = nlohmann::json::parse(data.begin(), data.end());
    if (!runtime.is_object())
        throw std::invalid_argument("ministral3 runtime.json must be an object");
    ministral3::EncoderSettings result;
    result.task = bundle.info().task;
    result.hidden_size = require_positive(runtime, "hidden_size");
    result.vocab_size = require_positive(runtime, "vocab_size");
    result.max_sequence_length = require_positive(runtime, "max_sequence_length");
    result.query_prompt = require_string(runtime, "query_prompt");
    result.document_prompt = require_string(runtime, "document_prompt");
    return result;
}

std::unique_ptr<ITrtModule> load_encoder(IBackend& backend, const BundleReader& bundle) {
    const auto plan = require_section(bundle, "engine.plan");
    const auto start = std::chrono::steady_clock::now();
    auto module = backend.create_module(plan.data(), plan.size(), ModuleCreateOptions{});
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "[trtmc.load_timing] label=\"engine.plan\" load_deserialize_ms=" << elapsed
              << " plan_bytes=" << plan.size() << '\n';
    if (!module || !module->ok())
        throw std::runtime_error("ministral3 failed to load engine.plan");
    module->set_timing_label("engine.plan");
    return module;
}

} // namespace
} // namespace trtmc::ministral3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("ministral3 does not support --kv-cache-size");
    auto settings = trtmc::ministral3_factory::settings(context.reader);
    const auto tokenizer_json =
        trtmc::ministral3_factory::require_section(context.reader, "tokenizer.json");
    auto tokenizer =
        trtmc::ministral3::TekkenTokenizer::load(tokenizer_json.data(), tokenizer_json.size());
    auto encoder = trtmc::ministral3_factory::load_encoder(context.backend, context.reader);
    return new trtmc::ministral3::EncoderPipeline(std::move(encoder), std::move(tokenizer),
                                                  std::move(settings));
}
