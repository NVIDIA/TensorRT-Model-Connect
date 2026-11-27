/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/granite_speech5/runtime/ctc.h"
#include "families/granite_speech5/runtime/pipeline.h"
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
        throw std::runtime_error("granite_speech5 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::string_view text(const std::vector<char>& data) {
    return {data.data(), data.size()};
}

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("granite_speech5 does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ISpeechTranscription::kTask)
        throw std::runtime_error("granite_speech5 bundle declares an unsupported task: " + task);

    const auto geometry = trtmc::granite_speech5::Geometry::from_json(
        text(require_section(context.reader, "runtime.json")));
    auto vocabulary = trtmc::granite_speech5::Vocabulary::from_json(
        text(require_section(context.reader, "vocabulary.json")), geometry.vocab_size);
    std::unique_ptr<trtmc::ITrtModule> encoder;
    {
        const auto plan = require_section(context.reader, "encoder.plan");
        trtmc::ModuleCreateOptions options{};
        encoder = context.backend.create_module(plan.data(), plan.size(), options);
    }
    if (!encoder || !encoder->ok())
        throw std::runtime_error("granite_speech5 encoder engine failed to load");
    return new trtmc::granite_speech5::Pipeline(std::move(encoder), geometry,
                                                std::move(vocabulary));
}
