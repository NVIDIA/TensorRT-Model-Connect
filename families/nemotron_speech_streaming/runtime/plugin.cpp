/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_speech_streaming/runtime/distributed_runtime.h"
#include "families/nemotron_speech_streaming/runtime/pipeline.h"
#include "families/nemotron_speech_streaming/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::nemotron_streaming_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

RnntConfig parse_config(const nlohmann::json& json) {
    RnntConfig config;
#define RNNT_INT(field) config.field = json.at(#field).get<std::int32_t>()
    RNNT_INT(sample_rate);
    RNNT_INT(num_mel_bins);
    RNNT_INT(mel_n_fft);
    RNNT_INT(mel_win_length);
    RNNT_INT(mel_hop_length);
    RNNT_INT(mel_chunk_length);
    RNNT_INT(mel_length);
    RNNT_INT(encoder_hidden_size);
    RNNT_INT(pred_hidden_size);
    RNNT_INT(pred_num_layers);
    RNNT_INT(encoder_layers);
    RNNT_INT(vocab_size);
    RNNT_INT(blank_id);
    RNNT_INT(max_symbols_per_step);
    RNNT_INT(encoder_seq_len);
    RNNT_INT(att_context_left);
    RNNT_INT(att_context_right);
    RNNT_INT(subsampling_factor);
    RNNT_INT(streaming_cache_left);
    RNNT_INT(streaming_time_cache);
    RNNT_INT(streaming_pre_encode_cache);
    RNNT_INT(streaming_drop_pre_encoded);
    RNNT_INT(num_prompts);
#undef RNNT_INT
    config.mel_preemph = json.at("mel_preemph").get<float>();
    config.causal_downsampling = json.at("causal_downsampling").get<bool>();
    config.has_prompt_kernel = json.at("has_prompt_kernel").get<bool>();
    config.prompt_dictionary =
        json.at("prompt_dictionary").get<std::unordered_map<std::string, std::int32_t>>();
    config.supported_right_contexts =
        json.at("supported_right_contexts").get<std::vector<std::int32_t>>();
    if (config.sample_rate <= 0 || config.encoder_hidden_size <= 0 ||
        config.pred_hidden_size <= 0 || config.pred_num_layers <= 0 || config.vocab_size <= 0 ||
        config.supported_right_contexts.empty())
        throw std::runtime_error("Nemotron streaming runtime.json is invalid");
    if (config.has_prompt_kernel && (config.num_prompts <= 0 || config.prompt_dictionary.empty()))
        throw std::runtime_error("Nemotron streaming prompt configuration is invalid");
    return config;
}

} // namespace
} // namespace trtmc::nemotron_streaming_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron_speech_streaming does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime_data =
        nemotron_streaming_factory::require_section(context.reader, "runtime.json");
    const auto document = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    auto config = nemotron_streaming_factory::parse_config(document);
    const auto tp_size = document.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Nemotron streaming tensor_parallel_size must be positive");
    const auto group = nemotron_speech_streaming::initialize_tensor_parallel_group(tp_size);
    const std::string predictor_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";

    ModuleCreateOptions options{};
    const auto& encoder_plan =
        nemotron_streaming_factory::require_section(context.reader, "encoder.plan");
    auto encoder =
        load_trt_module_from_plan(&context.backend, &encoder_plan, "encoder.plan", options);
    options.stream = encoder.module->stream();
    ModuleCreateOptions predictor_options = options;
    if (tp_size > 1) {
        predictor_options.distributed_communicator = group.communicator;
        predictor_options.distributed_owner = group.owner;
    }
    const auto& predictor_plan =
        nemotron_streaming_factory::require_section(context.reader, predictor_section.c_str());
    auto predictor = load_trt_module_from_plan(&context.backend, &predictor_plan,
                                               predictor_section.c_str(), predictor_options);
    const auto& joint_plan =
        nemotron_streaming_factory::require_section(context.reader, "joint.plan");
    auto joint = load_trt_module_from_plan(&context.backend, &joint_plan, "joint.plan", options);
    std::unique_ptr<ITrtModule> prompt;
    if (config.has_prompt_kernel) {
        const auto& prompt_plan =
            nemotron_streaming_factory::require_section(context.reader, "prompt.plan");
        prompt = load_trt_module_from_plan(&context.backend, &prompt_plan, "prompt.plan", options)
                     .module;
    }

    std::map<std::int32_t, std::vector<char>> streaming;
    std::map<std::int32_t, std::vector<char>> first;
    for (const auto right : config.supported_right_contexts) {
        const auto section = "streaming." + std::to_string(right) + ".plan";
        const auto first_section = "streaming." + std::to_string(right) + ".first.plan";
        streaming.emplace(
            right, nemotron_streaming_factory::require_section(context.reader, section.c_str()));
        first.emplace(right, nemotron_streaming_factory::require_section(context.reader,
                                                                         first_section.c_str()));
    }
    auto mel = load_mel_filterbank(context.reader);
    if (mel.data.empty())
        throw std::runtime_error("Nemotron streaming mel_filterbank is invalid");
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Nemotron streaming tokenizer is missing");
    return new RnntPipeline(std::move(encoder.module), std::move(predictor.module),
                            std::move(joint.module), std::move(prompt), std::move(streaming),
                            &context.backend, options, std::move(first), std::move(config),
                            std::move(mel), options.stream, std::move(tokenizer));
}
