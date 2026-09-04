/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bark/runtime/pipeline.h"
#include "families/bark/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdlib>
#include <dlfcn.h>
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

std::int32_t require_rank(std::int32_t size) {
    require_nccl(size);
    if (size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("Bark TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= size)
        throw std::runtime_error("Bark RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
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
    using namespace trtmc;
    const auto& runtime_data = bark_factory::require_section(context.reader, "runtime.json");
    const auto document = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    const auto tp_size = document.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Bark tensor_parallel_size must be positive");
    const auto rank = bark_factory::require_rank(tp_size);
    const std::string semantic_section =
        tp_size == 1 ? "semantic.decode.plan"
                     : "semantic.decode.rank" + std::to_string(rank) + ".plan";
    const std::string semantic_prefill_section =
        tp_size == 1 ? "semantic.prefill.plan"
                     : "semantic.prefill.rank" + std::to_string(rank) + ".plan";
    const std::string coarse_section =
        tp_size == 1 ? "coarse.decode.plan" : "coarse.decode.rank" + std::to_string(rank) + ".plan";
    const std::string coarse_prefill_section =
        tp_size == 1 ? "coarse.prefill.plan"
                     : "coarse.prefill.rank" + std::to_string(rank) + ".plan";
    ModuleCreateOptions options{};
    const auto load = [&](const char* name, cudaStream_t stream = nullptr) {
        auto module_options = options;
        module_options.stream = stream;
        const auto& plan = bark_factory::require_section(context.reader, name);
        return load_trt_module_from_plan(&context.backend, &plan, name, module_options).module;
    };
    auto semantic = load(semantic_section.c_str());
    const auto stream = semantic->stream();
    auto coarse = load(coarse_section.c_str(), stream);
    auto semantic_prefill = load(semantic_prefill_section.c_str(), stream);
    auto coarse_prefill = load(coarse_prefill_section.c_str(), stream);
    auto codec = load("codec.plan", stream);
    auto fine = load("fine.plan", stream);

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
