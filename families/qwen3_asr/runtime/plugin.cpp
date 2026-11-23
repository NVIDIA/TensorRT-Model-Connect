/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_asr/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::qwen3_asr {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("qwen3_asr bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

MelFilterbank parse_filters(const std::vector<char>& data) {
    MelFilterbank filters;
    if (data.size() < 2 * sizeof(std::int32_t))
        throw std::runtime_error("qwen3_asr mel_filters.bin is truncated");
    std::memcpy(&filters.n_freq, data.data(), sizeof(std::int32_t));
    std::memcpy(&filters.n_mel, data.data() + sizeof(std::int32_t), sizeof(std::int32_t));
    if (filters.n_freq <= 0 || filters.n_mel <= 0)
        throw std::runtime_error("qwen3_asr mel_filters.bin states an empty filterbank");
    const auto count =
        static_cast<std::size_t>(filters.n_freq) * static_cast<std::size_t>(filters.n_mel);
    if (data.size() != 2 * sizeof(std::int32_t) + count * sizeof(float))
        throw std::runtime_error("qwen3_asr mel_filters.bin size does not match its header");
    filters.weights.resize(count);
    std::memcpy(filters.weights.data(), data.data() + 2 * sizeof(std::int32_t),
                count * sizeof(float));
    return filters;
}

} // namespace
} // namespace trtmc::qwen3_asr

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc::qwen3_asr;
    // The KV cache is sized at build time (max_sequence_length).
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_asr does not support --kv-cache-size");
    const auto& task = context.reader.info().task;
    if (task != trtmc::internal::ISpeechTranscription::kTask)
        throw std::runtime_error("qwen3_asr bundle declares an unsupported task: " + task);

    auto config = parse_runtime_config(require_section(context.reader, "runtime.json"));
    auto filters = parse_filters(require_section(context.reader, "mel_filters.bin"));
    auto vocabulary = Vocabulary::parse(require_section(context.reader, "vocab.bin"));

    trtmc::ModuleCreateOptions options{};
    const auto encoder_plan = require_section(context.reader, "audio_encoder.plan");
    auto encoder = context.backend.create_module(encoder_plan.data(), encoder_plan.size(), options);
    const auto decoder_plan = require_section(context.reader, "decoder.plan");
    auto decoder = context.backend.create_dual_profile_modules(decoder_plan.data(),
                                                               decoder_plan.size(), options);
    if (!encoder || !decoder.prefill || !decoder.decode)
        throw std::runtime_error("qwen3_asr engines failed to load");
    encoder->set_timing_label("qwen3_asr.audio_encoder");
    decoder.prefill->set_timing_label("qwen3_asr.decoder_prefill");
    decoder.decode->set_timing_label("qwen3_asr.decoder_decode");
    return new Qwen3AsrPipeline(std::move(encoder), std::move(decoder.prefill),
                                std::move(decoder.decode), std::move(config), std::move(filters),
                                std::move(vocabulary));
}
