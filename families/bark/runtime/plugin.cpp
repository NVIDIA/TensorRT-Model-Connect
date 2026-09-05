/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bark/runtime/distributed_runtime.h"
#include "families/bark/runtime/pipeline.h"
#include "families/bark/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::bark_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

BarkConfig parse_config(const nlohmann::json& json) {
    BarkConfig config;
#define BARK_INT(field) config.field = json.at(#field).get<std::int32_t>()
    BARK_INT(sample_rate);
    BARK_INT(hidden_size);
    BARK_INT(semantic_input_vocab);
    BARK_INT(semantic_output_vocab);
    BARK_INT(text_encoding_offset);
    BARK_INT(text_pad_token);
    BARK_INT(semantic_pad_token);
    BARK_INT(semantic_infer_token);
    BARK_INT(semantic_vocab_size);
    BARK_INT(coarse_input_vocab);
    BARK_INT(coarse_semantic_pad_token);
    BARK_INT(coarse_infer_token);
    BARK_INT(n_coarse_codebooks);
    BARK_INT(codebook_size);
    BARK_INT(coarse_rate_hz);
    BARK_INT(max_coarse_history);
    BARK_INT(max_coarse_input_length);
    BARK_INT(sliding_window_len);
    BARK_INT(codec_seq_length);
    BARK_INT(codec_upsample_factor);
    BARK_INT(codec_n_codebooks);
    BARK_INT(fine_hidden_size);
    BARK_INT(fine_n_lm_heads);
    BARK_INT(fine_codebook_size);
    BARK_INT(fine_seq_length);
    BARK_INT(top_k);
#undef BARK_INT
    config.semantic_rate_hz = json.at("semantic_rate_hz").get<float>();
    config.semantic_temperature = json.at("semantic_temperature").get<float>();
    config.coarse_temperature = json.at("coarse_temperature").get<float>();
    config.fine_temperature = json.at("fine_temperature").get<float>();
    config.min_eos_p = json.at("min_eos_p").get<float>();
    config.greedy = json.at("greedy").get<bool>();
    config.seed = json.at("seed").get<std::int64_t>();
    if (config.sample_rate <= 0 || config.hidden_size <= 0 || config.semantic_vocab_size <= 0 ||
        config.n_coarse_codebooks <= 0 || config.codebook_size <= 0 ||
        config.codec_seq_length <= 0 || config.fine_seq_length <= 0)
        throw std::runtime_error("Bark runtime.json does not match its runtime contract");
    return config;
}

} // namespace
} // namespace trtmc::bark_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("bark does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime_data = bark_factory::require_section(context.reader, "runtime.json");
    const auto document = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    const auto tp_size = document.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Bark tensor_parallel_size must be positive");
    const auto group = bark::initialize_tensor_parallel_group(tp_size);
    const std::string semantic_section =
        tp_size == 1 ? "semantic.decode.plan"
                     : "semantic.decode.rank" + std::to_string(group.rank) + ".plan";
    const std::string semantic_prefill_section =
        tp_size == 1 ? "semantic.prefill.plan"
                     : "semantic.prefill.rank" + std::to_string(group.rank) + ".plan";
    const std::string coarse_section =
        tp_size == 1 ? "coarse.decode.plan"
                     : "coarse.decode.rank" + std::to_string(group.rank) + ".plan";
    const std::string coarse_prefill_section =
        tp_size == 1 ? "coarse.prefill.plan"
                     : "coarse.prefill.rank" + std::to_string(group.rank) + ".plan";
    ModuleCreateOptions options{};
    ModuleCreateOptions decoder_options{};
    if (tp_size > 1) {
        decoder_options.distributed_communicator = group.communicator;
        decoder_options.distributed_owner = group.owner;
    }
    const auto load = [&](const char* name, const ModuleCreateOptions& base_options,
                          cudaStream_t stream = nullptr) {
        auto module_options = base_options;
        module_options.stream = stream;
        const auto& plan = bark_factory::require_section(context.reader, name);
        return load_trt_module_from_plan(&context.backend, &plan, name, module_options).module;
    };
    std::unique_ptr<ITrtModule> semantic;
    std::unique_ptr<ITrtModule> coarse;
    std::unique_ptr<ITrtModule> semantic_prefill;
    std::unique_ptr<ITrtModule> coarse_prefill;
    cudaStream_t stream = nullptr;
    if (tp_size == 1) {
        const auto& semantic_plan =
            bark_factory::require_section(context.reader, semantic_section.c_str());
        auto semantic_modules = context.backend.create_dual_profile_modules(
            semantic_plan.data(), semantic_plan.size(), options);
        if (!semantic_modules.prefill || !semantic_modules.decode ||
            !semantic_modules.prefill->ok() || !semantic_modules.decode->ok())
            throw std::runtime_error(
                "Bark semantic engine does not provide prefill and decode profiles");
        semantic_prefill = std::move(semantic_modules.prefill);
        semantic = std::move(semantic_modules.decode);
        semantic_prefill->set_timing_label(semantic_prefill_section);
        semantic->set_timing_label(semantic_section);
        stream = semantic->stream();

        auto coarse_options = options;
        coarse_options.stream = stream;
        const auto& coarse_plan =
            bark_factory::require_section(context.reader, coarse_section.c_str());
        auto coarse_modules = context.backend.create_dual_profile_modules(
            coarse_plan.data(), coarse_plan.size(), coarse_options);
        if (!coarse_modules.prefill || !coarse_modules.decode || !coarse_modules.prefill->ok() ||
            !coarse_modules.decode->ok())
            throw std::runtime_error(
                "Bark coarse engine does not provide prefill and decode profiles");
        coarse_prefill = std::move(coarse_modules.prefill);
        coarse = std::move(coarse_modules.decode);
        coarse_prefill->set_timing_label(coarse_prefill_section);
        coarse->set_timing_label(coarse_section);
    } else {
        semantic = load(semantic_section.c_str(), decoder_options);
        stream = semantic->stream();
        coarse = load(coarse_section.c_str(), decoder_options, stream);
        semantic_prefill = load(semantic_prefill_section.c_str(), decoder_options, stream);
        coarse_prefill = load(coarse_prefill_section.c_str(), decoder_options, stream);
    }
    auto codec = load("codec.plan", options, stream);
    auto fine = load("fine.plan", options, stream);

    auto config = bark_factory::parse_config(document);
    const auto semantic_layers = document.at("semantic_num_layers").get<std::int32_t>();
    const auto semantic_cache = document.at("semantic_max_cache_length").get<std::int32_t>();
    const auto coarse_layers = document.at("coarse_num_layers").get<std::int32_t>();
    const auto coarse_cache = document.at("coarse_max_cache_length").get<std::int32_t>();
    const auto semantic_shape = semantic->tensor_shape("cache_k_0");
    const auto coarse_shape = coarse->tensor_shape("cache_k_0");
    if (semantic_layers <= 0 || semantic_cache <= 0 || coarse_layers <= 0 || coarse_cache <= 0 ||
        semantic_shape.size() < 2 || semantic_shape[1] <= 0 || coarse_shape.size() < 2 ||
        coarse_shape[1] <= 0)
        throw std::runtime_error("Bark engine cache geometry is invalid");
    auto semantic_state = std::make_unique<BarkKvCache>(
        semantic_layers, semantic_cache, static_cast<std::int32_t>(semantic_shape[1]), stream,
        semantic->tensor_dtype("cache_k_0"));
    auto coarse_state = std::make_unique<BarkKvCache>(coarse_layers, coarse_cache,
                                                      static_cast<std::int32_t>(coarse_shape[1]),
                                                      stream, coarse->tensor_dtype("cache_k_0"));
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Bark bundle does not contain its required tokenizer");
    const auto semantic_embed_section =
        bark_factory::require_section(context.reader, "semantic.embed");
    const auto coarse_embed_section = bark_factory::require_section(context.reader, "coarse.embed");
    const auto fine_embed_section = bark_factory::require_section(context.reader, "fine.embed");
    const auto fine_position_section =
        bark_factory::require_section(context.reader, "fine.position_embed");
    auto semantic_embed = section_to_floats(&semantic_embed_section);
    auto coarse_embed = section_to_floats(&coarse_embed_section);
    auto fine_embed = section_to_floats(&fine_embed_section);
    auto fine_position = section_to_floats(&fine_position_section);
    if (semantic_embed.empty() || coarse_embed.empty() || fine_embed.empty() ||
        fine_position.empty())
        throw std::runtime_error("Bark embedding sections are invalid");

    auto pipeline = std::make_unique<BarkPipeline>(
        std::move(semantic), std::move(coarse), std::move(semantic_state), std::move(coarse_state),
        std::move(semantic_embed), std::move(coarse_embed), std::move(config), stream,
        std::move(tokenizer));
    pipeline->set_prefill_modules(std::move(semantic_prefill), std::move(coarse_prefill));
    pipeline->set_codec_module(std::move(codec));
    pipeline->set_fine_module(std::move(fine));
    pipeline->set_fine_embeddings(std::move(fine_embed), std::move(fine_position));
    return pipeline.release();
}
