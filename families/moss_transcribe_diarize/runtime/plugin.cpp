/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moss_transcribe_diarize/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::moss_transcribe_diarize {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error(
            std::string("moss_transcribe_diarize bundle section is missing: ") + name);
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::moss_transcribe_diarize

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    namespace mtd = trtmc::moss_transcribe_diarize;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("moss_transcribe_diarize does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ISpeechTranscription::kTask)
        throw std::runtime_error("moss_transcribe_diarize bundle declares an unsupported task: " +
                                 task);

    auto config = mtd::parse_runtime_config(mtd::require_section(context.reader, "runtime.json"));
    auto mel_filters =
        mtd::parse_mel_filter_bank(mtd::require_section(context.reader, "mel_filters.bin"));
    mtd::TokenDecoder decoder(mtd::require_section(context.reader, "token_bytes.bin"));

    const auto text_plan = mtd::require_section(context.reader, "text.plan");
    auto text = context.backend.create_dual_profile_modules(text_plan.data(), text_plan.size(),
                                                            trtmc::ModuleCreateOptions{});
    if (!text.prefill || !text.decode)
        throw std::runtime_error("moss_transcribe_diarize text engine failed to load");
    trtmc::ModuleCreateOptions audio_options{};
    audio_options.stream = text.prefill->stream();
    const auto audio_plan = mtd::require_section(context.reader, "audio.plan");
    auto audio = context.backend.create_module(audio_plan.data(), audio_plan.size(), audio_options);

    mtd::Engines engines{std::move(audio), std::move(text.prefill), std::move(text.decode)};
    return new mtd::TranscribeDiarizeModel(std::move(engines), std::move(config),
                                           std::move(mel_filters), std::move(decoder));
}
