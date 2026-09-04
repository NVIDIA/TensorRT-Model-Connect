/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/pipeline.h"
#include "families/canary/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::canary_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

void require_nccl(std::int32_t tensor_parallel_size) {
    if (tensor_parallel_size <= 1)
        return;
    static void* const handle = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("tensor-parallel runtime requires NCCL: " +
                                 std::string(error == nullptr ? "unknown loader error" : error));
    }
}

std::int32_t require_rank(std::int32_t tp_size) {
    require_nccl(tp_size);
    if (tp_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("Canary TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tp_size)
        throw std::runtime_error("Canary RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

} // namespace
} // namespace trtmc::canary_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto& runtime_data = canary_factory::require_section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    const auto tp_size = config.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Canary tensor_parallel_size must be positive");
    const auto rank = canary_factory::require_rank(tp_size);
    const std::string decoder_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& encoder_plan = canary_factory::require_section(context.reader, "encoder.plan");
    const auto& decoder_plan =
        canary_factory::require_section(context.reader, decoder_section.c_str());
    ModuleCreateOptions options{};
    auto encoder =
        load_trt_module_from_plan(&context.backend, &encoder_plan, "encoder.plan", options);
    options.stream = encoder.module->stream();
    auto decoder = load_trt_module_from_plan(&context.backend, &decoder_plan,
                                             decoder_section.c_str(), options);

    CanaryConfig model;
    model.num_mel_bins = config.at("num_mel_bins").get<std::int32_t>();
    model.max_source_positions = config.at("max_source_positions").get<std::int32_t>();
    model.max_target_positions = config.at("max_target_positions").get<std::int32_t>();
    model.encoder_layers = config.at("encoder_layers").get<std::int32_t>();
    model.decoder_layers = config.at("decoder_layers").get<std::int32_t>();
    model.eot_token_id = config.at("eot_token_id").get<std::int32_t>();
    model.mel_length = config.at("mel_length").get<std::int32_t>();
    model.decoder_start_token_ids =
        config.at("decoder_start_token_ids").get<std::vector<std::int32_t>>();
    model.supported_languages = config.at("supported_languages").get<std::vector<std::string>>();
    model.language_token_ids = config.at("language_token_ids").get<std::vector<std::int32_t>>();
    model.source_language_position = config.at("source_language_position").get<std::int32_t>();
    model.target_language_position = config.at("target_language_position").get<std::int32_t>();
    model.punctuation_position = config.at("punctuation_position").get<std::int32_t>();
    model.timestamp_position = config.at("timestamp_position").get<std::int32_t>();
    model.punctuation_token_id = config.at("punctuation_token_id").get<std::int32_t>();
    model.no_punctuation_token_id = config.at("no_punctuation_token_id").get<std::int32_t>();
    model.timestamp_token_id = config.at("timestamp_token_id").get<std::int32_t>();
    model.no_timestamp_token_id = config.at("no_timestamp_token_id").get<std::int32_t>();
    model.translation_requires_english = config.at("translation_requires_english").get<bool>();
    const auto hidden_size = config.at("hidden_size").get<std::int32_t>();
    const auto max_cache_length = config.at("max_cache_length").get<std::int32_t>();
    if (model.encoder_layers <= 0 || model.decoder_layers <= 0 || hidden_size <= 0 ||
        max_cache_length <= 0 || model.decoder_start_token_ids.empty() ||
        model.supported_languages.empty() ||
        model.supported_languages.size() != model.language_token_ids.size() ||
        model.punctuation_token_id < 0 || model.no_punctuation_token_id < 0 ||
        model.timestamp_token_id < 0 || model.no_timestamp_token_id < 0) {
        throw std::runtime_error("Canary runtime.json does not match its runtime contract");
    }
    const auto cache_shape = decoder.module->tensor_shape("cache_k_0");
    if (cache_shape.empty() || cache_shape.back() <= 0)
        throw std::runtime_error("Canary decoder cache shape is invalid");
    std::int32_t batch_capacity = 1;
    if (decoder.module->input_is_dynamic("token_id")) {
        const auto shape = decoder.module->input_profile_shape(
            "token_id", decoder.module->profile_idx(), ProfileShapeSelector::kMax);
        if (shape.empty() || shape.front() <= 0)
            throw std::runtime_error("Canary decoder batch profile is invalid");
        batch_capacity = static_cast<std::int32_t>(shape.front());
    }
    auto state = std::make_unique<CanaryKvCache>(
        model.decoder_layers, max_cache_length, static_cast<std::int32_t>(cache_shape.back()),
        decoder.module->stream(), decoder.module->tensor_dtype("cache_k_0"), batch_capacity);
    if (!state->ok())
        throw std::runtime_error("Canary failed to create its KV cache");
    auto mel = load_mel_filterbank(context.reader);
    if (mel.data.empty())
        throw std::runtime_error("Canary bundle has an invalid mel_filterbank section");
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Canary bundle does not contain its required tokenizer");

    return new CanaryPipeline(
        std::move(encoder.module), std::move(decoder.module), std::move(state), std::move(model),
        hidden_size, config.at("decoder_layers").get<std::int32_t>(), std::move(mel),
        config.at("mel_n_fft").get<std::int32_t>(), config.at("mel_win_length").get<std::int32_t>(),
        config.at("mel_hop_length").get<std::int32_t>(),
        config.at("mel_chunk_length").get<std::int32_t>(),
        config.at("mel_sampling_rate").get<std::int32_t>(), config.at("mel_preemph").get<float>(),
        config.at("mel_normalize_per_feature").get<bool>(), options.stream, std::move(tokenizer));
}
