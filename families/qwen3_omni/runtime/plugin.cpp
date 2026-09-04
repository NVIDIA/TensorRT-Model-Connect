/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/kv_cache.h"
#include "families/qwen3_omni/runtime/pipeline.h"
#include "families/qwen3_omni/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::qwen3_omni {
namespace {

template <typename T>
T require_value(const nlohmann::json& document, const char* name) {
    if (!document.contains(name))
        throw std::runtime_error(std::string("qwen3_omni runtime.json missing '") + name + "'");
    try {
        return document.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("qwen3_omni runtime.json has invalid '") + name + "'");
    }
}

Qwen3OmniRuntimeConfig parse_config(const BundleReader& bundle) {
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(require_text_section(bundle, "runtime.json"));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("qwen3_omni invalid runtime.json: " + std::string(error.what()));
    }
    if (!document.is_object())
        throw std::runtime_error("qwen3_omni runtime.json must be an object");

    Qwen3OmniRuntimeConfig config;
    config.precision = require_value<std::string>(document, "precision");
#define QWEN3_OMNI_INT(field) config.field = require_value<std::int32_t>(document, #field)
    QWEN3_OMNI_INT(sample_rate);
    QWEN3_OMNI_INT(thinker_hidden_size);
    QWEN3_OMNI_INT(thinker_num_layers);
    QWEN3_OMNI_INT(thinker_num_attention_heads);
    QWEN3_OMNI_INT(thinker_num_key_value_heads);
    QWEN3_OMNI_INT(thinker_head_dim);
    QWEN3_OMNI_INT(thinker_vocab_size);
    QWEN3_OMNI_INT(thinker_max_cache_length);
    QWEN3_OMNI_INT(thinker_eos_token_id);
    QWEN3_OMNI_INT(talker_hidden_size);
    QWEN3_OMNI_INT(talker_num_layers);
    QWEN3_OMNI_INT(talker_num_attention_heads);
    QWEN3_OMNI_INT(talker_num_key_value_heads);
    QWEN3_OMNI_INT(talker_head_dim);
    QWEN3_OMNI_INT(talker_vocab_size);
    QWEN3_OMNI_INT(talker_max_cache_length);
    QWEN3_OMNI_INT(predictor_hidden_size);
    QWEN3_OMNI_INT(predictor_num_layers);
    QWEN3_OMNI_INT(predictor_num_attention_heads);
    QWEN3_OMNI_INT(predictor_num_key_value_heads);
    QWEN3_OMNI_INT(predictor_head_dim);
    QWEN3_OMNI_INT(predictor_vocab_size);
    QWEN3_OMNI_INT(predictor_max_cache_length);
    QWEN3_OMNI_INT(num_codebooks);
    QWEN3_OMNI_INT(codebook_size);
    QWEN3_OMNI_INT(talker_max_frames);
    QWEN3_OMNI_INT(im_start_token_id);
    QWEN3_OMNI_INT(system_token_id);
    QWEN3_OMNI_INT(user_token_id);
    QWEN3_OMNI_INT(assistant_token_id);
    QWEN3_OMNI_INT(tts_bos_token_id);
    QWEN3_OMNI_INT(tts_eos_token_id);
    QWEN3_OMNI_INT(tts_pad_token_id);
    QWEN3_OMNI_INT(codec_bos_id);
    QWEN3_OMNI_INT(codec_eos_token_id);
    QWEN3_OMNI_INT(codec_nothink_id);
    QWEN3_OMNI_INT(codec_pad_id);
    QWEN3_OMNI_INT(codec_think_bos_id);
    QWEN3_OMNI_INT(codec_think_eos_id);
    QWEN3_OMNI_INT(speaker_id);
    QWEN3_OMNI_INT(code2wav_max_frames);
    QWEN3_OMNI_INT(code2wav_upsample_factor);
    QWEN3_OMNI_INT(code2wav_output_delay);
    QWEN3_OMNI_INT(code2wav_num_quantizers);
#undef QWEN3_OMNI_INT

    if (config.precision != "bf16")
        throw std::runtime_error("qwen3_omni runtime requires the qualified bf16 plans");
    const std::int32_t positive[] = {
        config.sample_rate,
        config.thinker_hidden_size,
        config.thinker_num_layers,
        config.thinker_num_attention_heads,
        config.thinker_num_key_value_heads,
        config.thinker_head_dim,
        config.thinker_vocab_size,
        config.thinker_max_cache_length,
        config.talker_hidden_size,
        config.talker_num_layers,
        config.talker_num_attention_heads,
        config.talker_num_key_value_heads,
        config.talker_head_dim,
        config.talker_vocab_size,
        config.talker_max_cache_length,
        config.predictor_hidden_size,
        config.predictor_num_layers,
        config.predictor_num_attention_heads,
        config.predictor_num_key_value_heads,
        config.predictor_head_dim,
        config.predictor_vocab_size,
        config.predictor_max_cache_length,
        config.num_codebooks,
        config.codebook_size,
        config.talker_max_frames,
        config.code2wav_max_frames,
        config.code2wav_upsample_factor,
        config.code2wav_num_quantizers,
    };
    for (const std::int32_t value : positive) {
        if (value <= 0)
            throw std::runtime_error("qwen3_omni runtime dimensions must be positive");
    }
    if (config.code2wav_output_delay < 0)
        throw std::runtime_error("qwen3_omni Code2Wav output delay must be non-negative");
    if (config.sample_rate != 24000 || config.num_codebooks != 16 ||
        config.code2wav_num_quantizers != config.num_codebooks ||
        config.talker_max_frames != config.code2wav_max_frames ||
        config.code2wav_max_frames != 32 || config.code2wav_upsample_factor != 1920 ||
        config.code2wav_output_delay != 555 ||
        config.predictor_max_cache_length < config.num_codebooks ||
        config.talker_max_cache_length <= config.talker_max_frames ||
        config.codebook_size != config.predictor_vocab_size ||
        config.codebook_size > config.talker_vocab_size ||
        config.talker_hidden_size != config.predictor_hidden_size) {
        throw std::runtime_error("qwen3_omni runtime fields violate the pinned audio contract");
    }
    const std::int32_t thinker_ids[] = {
        config.im_start_token_id,  config.system_token_id,  config.user_token_id,
        config.assistant_token_id, config.tts_bos_token_id, config.tts_eos_token_id,
        config.tts_pad_token_id,
    };
    for (const std::int32_t token : thinker_ids) {
        if (token < 0 || token >= config.thinker_vocab_size)
            throw std::runtime_error("qwen3_omni Thinker special token is out of range");
    }
    const std::int32_t talker_ids[] = {
        config.codec_bos_id, config.codec_eos_token_id, config.codec_nothink_id,
        config.codec_pad_id, config.codec_think_bos_id, config.codec_think_eos_id,
        config.speaker_id,
    };
    for (const std::int32_t token : talker_ids) {
        if (token < 0 || token >= config.talker_vocab_size)
            throw std::runtime_error("qwen3_omni Talker special token is out of range");
    }
    return config;
}

DType cache_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
}

struct DualModules {
    std::unique_ptr<ITrtModule> prefill;
    std::unique_ptr<ITrtModule> decode;
};

DualModules load_dual(IBackend& backend, const BundleReader& bundle, const char* section,
                      const char* label) {
    const auto plan = require_section(bundle, section);
    auto modules = backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (!modules.prefill || !modules.decode || !modules.prefill->ok() || !modules.decode->ok())
        throw std::runtime_error(std::string("qwen3_omni failed to load ") + label);
    modules.prefill->set_timing_label(std::string(label) + " prefill");
    modules.decode->set_timing_label(std::string(label) + " decode");
    return {std::move(modules.prefill), std::move(modules.decode)};
}

std::vector<float> floats_from_section(const BundleReader& bundle, const char* name) {
    const auto data = require_section(bundle, name);
    if (data.size() % sizeof(float) != 0)
        throw std::runtime_error(std::string("qwen3_omni section is not float32: ") + name);
    std::vector<float> result(data.size() / sizeof(float));
    std::memcpy(result.data(), data.data(), data.size());
    return result;
}

std::unique_ptr<Qwen3OmniKvCache> make_state(const Qwen3OmniRuntimeConfig& config,
                                             ITrtModule& module, std::int32_t layers,
                                             std::int32_t cache_length, std::int32_t kv_heads,
                                             std::int32_t head_dim) {
    auto state = std::make_unique<Qwen3OmniKvCache>(layers, cache_length, kv_heads * head_dim,
                                                    module.stream(), cache_dtype(config.precision));
    if (!state->ok())
        throw std::runtime_error("qwen3_omni failed to allocate a decoder KV cache");
    return state;
}

} // namespace

ITask* create(const FamilyContext& context) {
    Qwen3OmniRuntimeConfig config = parse_config(context.reader);
    DualModules thinker =
        load_dual(context.backend, context.reader, "thinker.plan", "Qwen3-Omni Thinker");
    DualModules talker =
        load_dual(context.backend, context.reader, "talker.plan", "Qwen3-Omni Talker");
    DualModules predictor = load_dual(context.backend, context.reader, "code_predictor.plan",
                                      "Qwen3-Omni CodePredictor");
    auto text_projection =
        load_engine(context.backend, require_section(context.reader, "text_projection.plan"),
                    "Qwen3-Omni text projection");
    auto code2wav = load_engine(context.backend, require_section(context.reader, "code2wav.plan"),
                                "Qwen3-Omni Code2Wav");

    auto thinker_state = make_state(config, *thinker.decode, config.thinker_num_layers,
                                    config.thinker_max_cache_length,
                                    config.thinker_num_key_value_heads, config.thinker_head_dim);
    auto talker_state =
        make_state(config, *talker.decode, config.talker_num_layers, config.talker_max_cache_length,
                   config.talker_num_key_value_heads, config.talker_head_dim);
    auto predictor_state = make_state(
        config, *predictor.decode, config.predictor_num_layers, config.predictor_max_cache_length,
        config.predictor_num_key_value_heads, config.predictor_head_dim);

    return new Qwen3OmniAudioPipeline(
        std::move(thinker.prefill), std::move(thinker.decode), std::move(thinker_state),
        std::move(text_projection), std::move(talker.prefill), std::move(talker.decode),
        std::move(talker_state), std::move(predictor.prefill), std::move(predictor.decode),
        std::move(predictor_state), std::move(code2wav),
        floats_from_section(context.reader, "talker.codec_embedding.f32"),
        floats_from_section(context.reader, "predictor.codec_embeddings.f32"), std::move(config),
        create_tokenizer(context.reader));
}

} // namespace trtmc::qwen3_omni

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::qwen3_omni::create(context);
}
