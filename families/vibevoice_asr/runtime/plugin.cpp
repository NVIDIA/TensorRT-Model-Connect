/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/pipeline.h"
#include "families/vibevoice_asr/runtime/runtime_config.h"
#include "families/vibevoice_asr/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::vibevoice_asr {
namespace {

std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (!section || section->length == 0)
        throw std::runtime_error(std::string("VibeVoice-ASR bundle lacks section ") + name);
    return reader.read_section(name);
}

void require_token(const ITokenizer& tokenizer, const char* text, std::int32_t id) {
    if (tokenizer.id_for_token(text) != id)
        throw std::invalid_argument(std::string("VibeVoice-ASR tokenizer id mismatch for ") + text);
}

} // namespace
} // namespace trtmc::vibevoice_asr

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    using namespace trtmc::vibevoice_asr;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("vibevoice_asr does not support --kv-cache-size");
    const auto& info = context.reader.info();
    if (info.family != "vibevoice_asr" || info.task != "speech_transcription" ||
        info.backend != "trt" || std::string(context.backend.name()) != "trt")
        throw std::invalid_argument("VibeVoice-ASR bundle identity does not match the runtime");

    const auto runtime = require_section(context.reader, "runtime.json");
    auto config = parse_runtime_config(std::string(runtime.begin(), runtime.end()));
    const auto tokenizer_json = require_section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer)
        throw std::invalid_argument("VibeVoice-ASR could not load its tokenizer");
    const auto& ids = config.token_ids;
    require_token(*tokenizer, "<|im_start|>", ids.im_start);
    require_token(*tokenizer, "<|im_end|>", ids.im_end);
    require_token(*tokenizer, "<|object_ref_start|>", ids.speech_start);
    require_token(*tokenizer, "<|object_ref_end|>", ids.speech_end);
    require_token(*tokenizer, "<|box_start|>", ids.speech_pad);
    require_token(*tokenizer, "<|endoftext|>", ids.eos);
    for (const auto& check : config.tokenizer_checks) {
        if (tokenizer->encode(check.text) != check.ids)
            throw std::invalid_argument(
                "VibeVoice-ASR native tokenizer disagrees with the build tokenizer on: " +
                check.text);
    }

    const auto encoder_plan = require_section(context.reader, "audio_encoder.plan");
    auto encoder = context.backend.create_module(encoder_plan.data(), encoder_plan.size(), {});
    if (!encoder || !encoder->ok())
        throw std::runtime_error("VibeVoice-ASR could not load its audio encoder");
    const auto decoder_plan = require_section(context.reader, "decoder.plan");
    auto decoder =
        context.backend.create_dual_profile_modules(decoder_plan.data(), decoder_plan.size(), {});
    return new VibeVoiceAsrModel(std::move(config), std::move(tokenizer), std::move(encoder),
                                 std::move(decoder.prefill), std::move(decoder.decode));
}
