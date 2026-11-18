/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/granite_speech/runtime/pipeline.h"
#include "families/granite_speech/runtime/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::vector<char> require_section(const trtmc::BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("granite_speech bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::string_view text(const std::vector<char>& data) {
    return {data.data(), data.size()};
}

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("granite_speech does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ISpeechTranscription::kTask)
        throw std::runtime_error("granite_speech bundle declares an unsupported task: " + task);

    auto config = trtmc::granite_speech::RuntimeConfig::from_json(
        text(require_section(context.reader, "runtime.json")));
    auto vocabulary = trtmc::granite_speech::Vocabulary::from_json(
        text(require_section(context.reader, "vocabulary.json")));

    trtmc::ModuleCreateOptions options{};
    std::unique_ptr<trtmc::ITrtModule> audio;
    {
        const auto plan = require_section(context.reader, "audio.plan");
        audio = context.backend.create_module(plan.data(), plan.size(), options);
    }
    if (!audio || !audio->ok())
        throw std::runtime_error("granite_speech audio engine failed to load");
    trtmc::BackendDualProfileModules decoder;
    {
        const auto plan = require_section(context.reader, "decoder.plan");
        decoder = context.backend.create_dual_profile_modules(plan.data(), plan.size(), options);
    }
    return new trtmc::granite_speech::Pipeline(std::move(audio), std::move(decoder.prefill),
                                               std::move(decoder.decode), std::move(config),
                                               std::move(vocabulary));
}
