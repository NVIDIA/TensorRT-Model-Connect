/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::vibevoice {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load(IBackend& backend, const BundleReader& bundle, const char* name,
                                 cudaStream_t stream) {
    const auto plan = require_section(bundle, name);
    ModuleCreateOptions options{};
    options.stream = stream;
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("vibevoice engine failed to load: ") + name);
    module->set_timing_label(name);
    return module;
}

std::vector<std::string> added_tokens(const std::vector<char>& tokenizer_json) {
    const auto json = nlohmann::json::parse(tokenizer_json.begin(), tokenizer_json.end());
    std::vector<std::string> tokens;
    for (const auto& entry : json.at("added_tokens"))
        tokens.push_back(entry.at("content").get<std::string>());
    return tokens;
}

} // namespace
} // namespace trtmc::vibevoice

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    using namespace trtmc::vibevoice;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("vibevoice does not support --kv-cache-size");
    if (context.reader.info().task != internal::ITextToSpeech::kTask)
        throw std::runtime_error("vibevoice bundle declares an unsupported task: " +
                                 context.reader.info().task);
    const auto runtime = require_section(context.reader, "runtime.json");
    auto config = parse_runtime_config(nlohmann::json::parse(runtime.begin(), runtime.end()));

    Engines engines;
    const auto lm_plan = require_section(context.reader, "lm.plan");
    auto lm = context.backend.create_dual_profile_modules(lm_plan.data(), lm_plan.size(),
                                                          ModuleCreateOptions{});
    if (!lm.prefill || !lm.decode || !lm.prefill->ok() || !lm.decode->ok())
        throw std::runtime_error("vibevoice language model needs its prefill and decode profiles");
    engines.lm_prefill = std::move(lm.prefill);
    engines.lm_decode = std::move(lm.decode);
    engines.lm_prefill->set_timing_label("lm.prefill");
    engines.lm_decode->set_timing_label("lm.decode");
    const auto stream = engines.lm_decode->stream();
    engines.embed = load(context.backend, context.reader, "embed.plan", stream);
    engines.head = load(context.backend, context.reader, "diffusion_head.plan", stream);
    engines.feedback = load(context.backend, context.reader, "feedback.plan", stream);
    engines.voice = load(context.backend, context.reader, "voice_encoder.plan", stream);

    const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size());
    if (!tokenizer)
        throw std::runtime_error("vibevoice bundle tokenizer failed to load");
    return new VibeVoicePipeline(std::move(engines), std::move(config), std::move(tokenizer),
                                 added_tokens(tokenizer_json));
}
