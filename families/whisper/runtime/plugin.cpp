/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/whisper/runtime/distributed_runtime.h"
#include "families/whisper/runtime/pipeline.h"
#include "families/whisper/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::whisper_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::whisper_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("whisper does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime_data = whisper_factory::require_section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    const auto tp_size = config.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Whisper tensor_parallel_size must be positive");
    const auto group = whisper::initialize_tensor_parallel_group(tp_size);
    const std::string decoder_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";
    const auto& encoder_plan = whisper_factory::require_section(context.reader, "encoder.plan");
    const auto& decoder_plan =
        whisper_factory::require_section(context.reader, decoder_section.c_str());
    ModuleCreateOptions options{};
    auto encoder =
        load_trt_module_from_plan(&context.backend, &encoder_plan, "encoder.plan", options);
    ModuleCreateOptions decoder_options = options;
    decoder_options.stream = encoder.module->stream();
    if (tp_size > 1) {
        decoder_options.distributed_communicator = group.communicator;
        decoder_options.distributed_owner = group.owner;
    }
    auto decoder = load_trt_module_from_plan(&context.backend, &decoder_plan,
                                             decoder_section.c_str(), decoder_options);

    WhisperConfig model;
    model.num_mel_bins = config.at("num_mel_bins").get<std::int32_t>();
    model.max_source_positions = config.at("max_source_positions").get<std::int32_t>();
    model.max_target_positions = config.at("max_target_positions").get<std::int32_t>();
    model.encoder_layers = config.at("encoder_layers").get<std::int32_t>();
    model.decoder_layers = config.at("decoder_layers").get<std::int32_t>();
    model.eot_token_id = config.at("eot_token_id").get<std::int32_t>();
    model.mel_length = config.at("mel_length").get<std::int32_t>();
    model.decoder_start_token_ids =
        config.at("decoder_start_token_ids").get<std::vector<std::int32_t>>();
    const auto hidden_size = config.at("hidden_size").get<std::int32_t>();
    const auto max_cache_length = config.at("max_cache_length").get<std::int32_t>();
    if (model.num_mel_bins <= 0 || model.encoder_layers <= 0 || model.decoder_layers <= 0 ||
        hidden_size <= 0 || max_cache_length <= 0 || model.decoder_start_token_ids.empty()) {
        throw std::runtime_error("Whisper runtime.json does not match its runtime contract");
    }
    const auto cache_shape = decoder.module->tensor_shape("cache_k_0");
    if (cache_shape.size() < 2 || cache_shape.back() <= 0)
        throw std::runtime_error("Whisper decoder cache shape is invalid");
    auto state = std::make_unique<WhisperKvCache>(
        model.decoder_layers, max_cache_length, static_cast<std::int32_t>(cache_shape.back()),
        decoder.module->stream(), decoder.module->tensor_dtype("cache_k_0"));
    if (!state->ok())
        throw std::runtime_error("Whisper failed to create its KV cache");
    auto mel = load_mel_filterbank(context.reader);
    if (mel.data.empty())
        throw std::runtime_error("Whisper bundle has an invalid mel_filterbank section");
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Whisper bundle does not contain its required tokenizer");

    return new WhisperPipeline(
        std::move(encoder.module), std::move(decoder.module), std::move(state), std::move(model),
        hidden_size, config.at("decoder_layers").get<std::int32_t>(), std::move(mel),
        config.at("mel_n_fft").get<std::int32_t>(), config.at("mel_hop_length").get<std::int32_t>(),
        config.at("mel_chunk_length").get<std::int32_t>(),
        config.at("mel_sampling_rate").get<std::int32_t>(),
        config.at("mel_win_length").get<std::int32_t>(), config.at("mel_preemph").get<float>(),
        config.at("mel_normalize_per_feature").get<bool>(),
        config.at("mel_frontend").get<std::string>(), decoder_options.stream, std::move(tokenizer));
}
