/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::diffusion_gemma {
namespace {

std::vector<char> require_section(const BundleReader& reader, const std::string& name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("diffusion_gemma bundle is missing " + name);
    return reader.read_section(name);
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto document = nlohmann::json::parse(data.begin(), data.end());
    if (document.at("format_version").get<std::int32_t>() != 2 ||
        document.at("engine_layout").get<std::string>() != "qkv_layer")
        throw std::runtime_error("diffusion_gemma runtime configuration version is unsupported");
    const auto& text = document.at("text_config");
    RuntimeConfig config;
    config.hidden_size = text.at("hidden_size").get<std::int32_t>();
    config.vocab_size = text.at("vocab_size").get<std::int32_t>();
    config.num_layers = text.at("num_hidden_layers").get<std::int32_t>();
    config.num_attention_heads = text.at("num_attention_heads").get<std::int32_t>();
    config.num_key_value_heads = text.at("num_key_value_heads").get<std::int32_t>();
    config.num_global_key_value_heads = text.at("num_global_key_value_heads").get<std::int32_t>();
    config.head_dim = text.at("head_dim").get<std::int32_t>();
    config.global_head_dim = text.at("global_head_dim").get<std::int32_t>();
    config.sliding_window = text.at("sliding_window").get<std::int32_t>();
    config.layer_types = text.at("layer_types").get<std::vector<std::string>>();
    config.bos_token_id = text.at("bos_token_id").get<std::int32_t>();
    config.max_sequence_length = document.at("max_sequence_length").get<std::int32_t>();
    config.canvas_length = document.at("canvas_length").get<std::int32_t>();
    config.unsupported_token_ids =
        document.value("unsupported_token_ids", std::vector<std::int32_t>{});
    const auto precision = document.at("precision").get<std::string>();
    if (precision == "bf16")
        config.dtype = DType::kBFloat16;
    else if (precision == "fp32")
        config.dtype = DType::kFloat32;
    else
        throw std::runtime_error("diffusion_gemma runtime requires BF16 or FP32 precision");
    const auto& generation = document.at("generation_config");
    config.max_new_tokens = generation.at("max_new_tokens").get<std::int32_t>();
    config.max_denoising_steps = generation.at("max_denoising_steps").get<std::int32_t>();
    config.stability_threshold = generation.at("stability_threshold").get<std::int32_t>();
    config.confidence_threshold = generation.at("confidence_threshold").get<float>();
    config.t_min = generation.at("t_min").get<float>();
    config.t_max = generation.at("t_max").get<float>();
    config.entropy_bound = generation.at("sampler_config").at("entropy_bound").get<float>();
    config.eos_token_ids = generation.at("eos_token_id").get<std::vector<std::int32_t>>();
    if (config.hidden_size <= 0 || config.vocab_size <= 0 || config.num_layers <= 0 ||
        config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
        config.num_global_key_value_heads <= 0 || config.head_dim <= 0 ||
        config.global_head_dim <= 0 || config.sliding_window < 2 ||
        config.max_sequence_length <= config.canvas_length || config.canvas_length <= 0 ||
        config.max_new_tokens <= 0 || config.max_denoising_steps <= 0 ||
        config.stability_threshold < 0 || config.eos_token_ids.empty() ||
        config.layer_types.size() != static_cast<std::size_t>(config.num_layers))
        throw std::runtime_error(
            "diffusion_gemma runtime dimensions or generation controls are invalid");
    if (!(config.t_min > 0.0F) || !(config.t_max >= config.t_min) || !std::isfinite(config.t_min) ||
        !std::isfinite(config.t_max) || !(config.entropy_bound >= 0.0F) ||
        !std::isfinite(config.entropy_bound) || !(config.confidence_threshold >= 0.0F) ||
        !std::isfinite(config.confidence_threshold))
        throw std::runtime_error("diffusion_gemma sampler controls are invalid");
    for (const auto& type : config.layer_types)
        if (type != "sliding_attention" && type != "full_attention")
            throw std::runtime_error("diffusion_gemma runtime has an unsupported attention type");
    for (const auto id : config.eos_token_ids)
        if (id < 0 || id >= config.vocab_size)
            throw std::runtime_error("diffusion_gemma stop token is outside its vocabulary");
    return config;
}

} // namespace
} // namespace trtmc::diffusion_gemma

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("diffusion_gemma does not support --kv-cache-size");
    if (context.reader.info().task != "text_continuation")
        throw std::runtime_error("diffusion_gemma bundle must declare text_continuation");
    auto config = trtmc::diffusion_gemma::parse_config(
        trtmc::diffusion_gemma::require_section(context.reader, "runtime.json"));
    const auto tokenizer_data =
        trtmc::diffusion_gemma::require_section(context.reader, "tokenizer.json");
    auto tokenizer = trtmc::diffusion_gemma::CreateBpeTokenizer(tokenizer_data.data(),
                                                                tokenizer_data.size(), false);
    return new trtmc::diffusion_gemma::Pipeline(context.reader, context.backend, std::move(config),
                                                std::move(tokenizer));
}
