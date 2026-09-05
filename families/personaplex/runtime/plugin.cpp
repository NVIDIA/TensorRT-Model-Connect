/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/personaplex/runtime/distributed_runtime.h"
#include "families/personaplex/runtime/pipeline.h"
#include "families/personaplex/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::personaplex_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

SpeechConfig parse_config(const BundleReader& bundle, const nlohmann::json& json) {
    SpeechConfig config;
#define SPEECH_INT(field) config.field = json.at(#field).get<std::int32_t>()
    SPEECH_INT(sample_rate);
    SPEECH_INT(num_codebooks);
    SPEECH_INT(codebook_size);
    SPEECH_INT(mimi_max_frames);
    SPEECH_INT(temporal_hidden_size);
    SPEECH_INT(temporal_num_layers);
    SPEECH_INT(depth_hidden_size);
    SPEECH_INT(depth_num_layers);
    SPEECH_INT(depth_num_heads);
    SPEECH_INT(depth_num_kv_heads);
    SPEECH_INT(depth_max_cache_length);
    SPEECH_INT(text_padding_id);
    SPEECH_INT(mimi_decode_codebooks);
    SPEECH_INT(text_initial_token_id);
    SPEECH_INT(audio_initial_token_id);
    SPEECH_INT(depth_top_k);
    SPEECH_INT(text_eos_token_id);
#undef SPEECH_INT
    config.frame_rate = json.at("frame_rate").get<float>();
    config.depth_temperature = json.at("depth_temperature").get<float>();
    config.delays = json.at("delays").get<std::vector<std::int32_t>>();
    config.text_prompt_ids = json.at("text_prompt_ids").get<std::vector<std::int32_t>>();
    const auto floats = [&bundle](const char* name) {
        const auto section = require_section(bundle, name);
        return section_to_floats(&section);
    };
    config.depth_projection = floats("depth.projection");
    config.audio_embeddings = floats("audio.embeddings");
    config.temporal_text_embedding = floats("temporal_text.embeddings");
    config.depth_text_embedding = floats("depth_text.embeddings");
    config.depth_audio_embeddings = floats("depth_audio.embeddings");
    if (config.sample_rate <= 0 || config.num_codebooks <= 0 || config.codebook_size <= 0 ||
        config.temporal_hidden_size <= 0 || config.temporal_num_layers <= 0 ||
        config.depth_hidden_size <= 0 || config.depth_num_layers <= 0 ||
        config.depth_max_cache_length <= 0 || config.delays.empty())
        throw std::runtime_error("PersonaPlex runtime.json does not match its runtime contract");
    const auto checked_divide = [](std::size_t size, std::size_t divisor, const char* label) {
        if (divisor == 0 || size == 0 || size % divisor != 0)
            throw std::runtime_error(std::string("PersonaPlex invalid ") + label);
        return static_cast<std::int32_t>(size / divisor);
    };
    config.audio_vocab_size =
        checked_divide(config.audio_embeddings.size(),
                       static_cast<std::size_t>(config.num_codebooks) * config.temporal_hidden_size,
                       "audio embeddings");
    config.temporal_text_vocab =
        checked_divide(config.temporal_text_embedding.size(), config.temporal_hidden_size,
                       "temporal text embeddings");
    config.depth_text_vocab = checked_divide(config.depth_text_embedding.size(),
                                             config.depth_hidden_size, "depth text embeddings");
    config.num_depformer_emb =
        checked_divide(config.depth_audio_embeddings.size(),
                       static_cast<std::size_t>(config.audio_vocab_size) * config.depth_hidden_size,
                       "depth audio embeddings");
    config.temporal_hidden_for_proj = config.temporal_hidden_size;
    return config;
}

} // namespace
} // namespace trtmc::personaplex_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("personaplex does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime = personaplex_factory::require_section(context.reader, "runtime.json");
    const auto document = nlohmann::json::parse(runtime.begin(), runtime.end());
    auto config = personaplex_factory::parse_config(context.reader, document);
    const auto tp_size = document.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("PersonaPlex tensor_parallel_size must be positive");
    const auto group = personaplex::initialize_tensor_parallel_group(tp_size);
    const std::string temporal_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";
    ModuleCreateOptions options{};
    ModuleCreateOptions temporal_options{};
    if (tp_size > 1) {
        temporal_options.distributed_communicator = group.communicator;
        temporal_options.distributed_owner = group.owner;
    }
    const auto& temporal_plan =
        personaplex_factory::require_section(context.reader, temporal_section.c_str());
    auto temporal = load_trt_module_from_plan(&context.backend, &temporal_plan,
                                              temporal_section.c_str(), temporal_options);
    options.stream = temporal.module->stream();
    const auto load = [&](const char* name) {
        const auto& plan = personaplex_factory::require_section(context.reader, name);
        return load_trt_module_from_plan(&context.backend, &plan, name, options).module;
    };
    auto mimi_encoder = load("mimi.encoder.plan");
    auto mimi_decoder = load("mimi.decoder.plan");
    std::vector<std::unique_ptr<ITrtModule>> depth;
    depth.reserve(static_cast<std::size_t>(config.num_codebooks));
    for (std::int32_t codebook = 0; codebook < config.num_codebooks; ++codebook)
        depth.push_back(load(("depth." + std::to_string(codebook) + ".plan").c_str()));

    const auto temporal_shape = temporal.module->tensor_shape("cache_k_0");
    const auto depth_shape = depth.front()->tensor_shape("cache_k_0");
    if (temporal_shape.size() < 2 || temporal_shape[1] <= 0 || depth_shape.size() < 2 ||
        depth_shape[1] <= 0)
        throw std::runtime_error("PersonaPlex engine cache geometry is invalid");
    auto temporal_state = std::make_unique<PersonaplexKvCache>(
        config.temporal_num_layers, config.mimi_max_frames,
        static_cast<std::int32_t>(temporal_shape[1]), options.stream,
        temporal.module->tensor_dtype("cache_k_0"));
    auto depth_state = std::make_unique<PersonaplexKvCache>(
        config.depth_num_layers, config.depth_max_cache_length,
        static_cast<std::int32_t>(depth_shape[1]), options.stream,
        depth.front()->tensor_dtype("cache_k_0"));
    return new SpeechPipeline(std::move(mimi_encoder), std::move(temporal.module),
                              std::move(temporal_state), std::move(depth), std::move(depth_state),
                              std::move(mimi_decoder), std::move(config), options.stream);
}
