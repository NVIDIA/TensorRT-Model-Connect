/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/magpie_tts/runtime/audio_helpers.h"
#include "families/magpie_tts/runtime/distributed_runtime.h"
#include "families/magpie_tts/runtime/pipeline.h"
#include "families/magpie_tts/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::magpie_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::magpie_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("magpie_tts does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime_data = magpie_factory::require_section(context.reader, "runtime.json");
    const std::string runtime(runtime_data.begin(), runtime_data.end());
    const auto document = nlohmann::json::parse(runtime);
    const auto tp_size = document.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Magpie tensor_parallel_size must be positive");
    const auto group = magpie_tts::initialize_tensor_parallel_group(tp_size);
    const std::string decoder_section =
        tp_size == 1 ? "decoder.plan" : "decoder.rank" + std::to_string(group.rank) + ".plan";

    auto stream_owner = std::make_shared<MagpieCudaStream>();
    if (!stream_owner->ok())
        throw std::runtime_error("Magpie failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream_owner->get();
    const auto load = [&](const char* name) {
        const auto& plan = magpie_factory::require_section(context.reader, name);
        auto module = load_trt_module_from_plan(&context.backend, &plan, name, options);
        module.module->keep_alive(stream_owner);
        return std::move(module.module);
    };
    auto encoder = load("encoder.plan");
    ModuleCreateOptions decoder_options = options;
    if (tp_size > 1) {
        decoder_options.distributed_communicator = group.communicator;
        decoder_options.distributed_owner = group.owner;
    }
    const auto& decoder_plan =
        magpie_factory::require_section(context.reader, decoder_section.c_str());
    auto decoder_profiles = context.backend.create_dual_profile_modules(
        decoder_plan.data(), decoder_plan.size(), decoder_options);
    if (!decoder_profiles.prefill || !decoder_profiles.prefill->ok() || !decoder_profiles.decode ||
        !decoder_profiles.decode->ok())
        throw std::runtime_error("Magpie decoder plan must contain prefill and decode profiles");
    decoder_profiles.decode->keep_alive(stream_owner);
    auto codec = load("codec.plan");
    auto local_transformer = load("local_transformer.plan");

    auto config = build_magpie_config(runtime);
    const auto max_cache = document.at("max_cache_length").get<std::int32_t>();
    const auto shape = decoder_profiles.decode->tensor_shape("cache_k_0");
    if (max_cache <= 0 || shape.size() < 2 || shape[1] <= 0)
        throw std::runtime_error("Magpie decoder cache geometry is invalid");
    const auto kv_dim = static_cast<std::int32_t>(shape[1]);
    const auto cache_dtype = decoder_profiles.decode->tensor_dtype("cache_k_0");
    auto state = std::make_unique<MagpieKvCache>(config.decoder_layers, max_cache, kv_dim,
                                                 options.stream, cache_dtype);
    std::unique_ptr<MagpieInferenceState> unconditional_state;
    if (config.cfg_scale > 1.0F) {
        unconditional_state = std::make_unique<MagpieKvCache>(config.decoder_layers, max_cache,
                                                              kv_dim, options.stream, cache_dtype);
    }

    const auto encoder_bytes = static_cast<std::size_t>(config.max_source_positions) *
                               static_cast<std::size_t>(config.hidden_size) * sizeof(float);
    std::vector<MagpieCudaBuffer> cross_k;
    std::vector<MagpieCudaBuffer> cross_v;
    allocate_cross_kv_buffers(config.decoder_layers, encoder_bytes, cross_k, cross_v);
    std::vector<MagpieCudaBuffer> unconditional_k;
    std::vector<MagpieCudaBuffer> unconditional_v;
    if (config.cfg_scale > 1.0F)
        allocate_cross_kv_buffers(config.decoder_layers, encoder_bytes, unconditional_k,
                                  unconditional_v);
    MagpieCudaBuffer encoder_output(encoder_bytes);
    MagpieCudaBuffer unconditional_output(config.cfg_scale > 1.0F ? encoder_bytes : 0);

    const auto floats = [&context](const char* name) {
        const auto section = magpie_factory::require_section(context.reader, name);
        return section_to_floats(&section);
    };
    const auto ints = [&context](const char* name) {
        const auto section = magpie_factory::require_section(context.reader, name);
        return section_to_int32s(&section);
    };
    auto audio_embed = floats("audio.embed");
    auto text_embed = floats("text.embed");
    auto context_embed = floats("context.embed");
    auto context_lengths = ints("context.lengths");
    if (audio_embed.empty() || text_embed.empty() || context_embed.empty() ||
        context_lengths.empty())
        throw std::runtime_error("Magpie embedding sections are invalid");

    auto in_projection = floats("local_transformer.in_projection");
    auto out_projections = floats("local_transformer.out_projections");
    auto position_embedding = floats("local_transformer.position_embedding");
    const auto lt_shape = local_transformer->tensor_shape("input_embed");
    if (lt_shape.empty() || lt_shape.back() <= 0)
        throw std::runtime_error("Magpie local transformer input shape is invalid");
    const auto lt_hidden = static_cast<std::int32_t>(lt_shape.back());
    const auto in_weight_count = static_cast<std::size_t>(config.hidden_size) * lt_hidden;
    if (in_projection.size() != in_weight_count + static_cast<std::size_t>(lt_hidden))
        throw std::runtime_error("Magpie local transformer in_projection size is invalid");
    std::vector<float> in_weight(in_projection.begin(),
                                 in_projection.begin() +
                                     static_cast<std::ptrdiff_t>(in_weight_count));
    std::vector<float> in_bias(in_projection.begin() + static_cast<std::ptrdiff_t>(in_weight_count),
                               in_projection.end());
    const auto out_count = static_cast<std::size_t>(config.num_codebooks) *
                           static_cast<std::size_t>(lt_hidden + 1) * config.codebook_size;
    const auto position_count =
        static_cast<std::size_t>(config.num_codebooks) * static_cast<std::size_t>(lt_hidden);
    if (out_projections.size() != out_count || position_embedding.size() != position_count)
        throw std::runtime_error("Magpie local transformer projection assets have invalid sizes");

    return new MagpiePipeline(
        std::move(encoder), std::move(decoder_profiles.decode), std::move(state), std::move(codec),
        std::move(local_transformer), std::move(unconditional_state), std::move(cross_k),
        std::move(cross_v), std::move(unconditional_k), std::move(unconditional_v),
        std::move(encoder_output), std::move(unconditional_output), std::move(audio_embed),
        std::move(text_embed), std::move(context_embed), std::move(context_lengths),
        std::move(in_weight), std::move(in_bias), std::move(out_projections),
        std::move(position_embedding), lt_hidden, std::move(config), options.stream,
        make_ipa_tok(context.reader));
}
