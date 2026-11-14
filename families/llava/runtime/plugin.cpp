/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llava/runtime/cuda_stream.h"
#include "families/llava/runtime/pipeline.h"
#include "families/llava/runtime/plugin_helpers.h"
#include "families/llava/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cstdlib>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::llava_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string section_text(const BundleReader& bundle, const char* name) {
    const auto& data = require_section(bundle, name);
    return std::string(data.begin(), data.end());
}

std::int32_t require_int(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || !json.at(key).is_number_integer())
        throw std::runtime_error(std::string("LLaVA runtime.json requires integer ") + key);
    return json.at(key).get<std::int32_t>();
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    const auto value = require_int(json, key);
    if (value <= 0)
        throw std::runtime_error(std::string("LLaVA runtime.json has invalid ") + key);
    return value;
}

std::int32_t require_rank(std::int32_t tp_size) {
    if (tp_size == 1)
        return 0;
    const char* text = std::getenv("OMPI_COMM_WORLD_RANK");
    if (text == nullptr || *text == '\0')
        throw std::runtime_error("LLaVA TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= tp_size)
        throw std::runtime_error("LLaVA RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

LlavaKvCacheNames build_kv_names(const nlohmann::json& config, std::int32_t num_layers) {
    const auto& io = config.at("io_map");
    const auto cache_k = io.at("cache_k_pattern").get<std::string>();
    const auto cache_v = io.at("cache_v_pattern").get<std::string>();
    const auto present_k = io.at("present_k_pattern").get<std::string>();
    const auto present_v = io.at("present_v_pattern").get<std::string>();
    LlavaKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(llava_expand_layer_name(cache_k, layer));
        names.cache_v.push_back(llava_expand_layer_name(cache_v, layer));
        names.present_k.push_back(llava_expand_layer_name(present_k, layer));
        names.present_v.push_back(llava_expand_layer_name(present_v, layer));
    }
    return names;
}

} // namespace
} // namespace trtmc::llava_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("llava does not support --kv-cache-size");
    using namespace trtmc;
    const std::string runtime_text = llava_factory::section_text(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime_text);
    if (!config.is_object())
        throw std::runtime_error("LLaVA runtime.json must be an object");

    const auto tp_size = llava_factory::require_positive_int(config, "tensor_parallel_size");
    const auto rank = llava_factory::require_rank(tp_size);
    const std::string decode_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& decode_plan =
        llava_factory::require_section(context.reader, decode_section.c_str());
    const auto& prefill_plan = llava_factory::require_section(context.reader, "prefill.plan");
    const auto& vision_plan = llava_factory::require_section(context.reader, "vision.plan");

    auto stream_owner = std::make_shared<LlavaCudaStream>();
    if (!stream_owner->ok())
        throw std::runtime_error("LLaVA failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream_owner->get();
    auto decode =
        load_trt_module_from_plan(&context.backend, &decode_plan, decode_section.c_str(), options);
    auto prefill =
        load_trt_module_from_plan(&context.backend, &prefill_plan, "prefill.plan", options);
    auto vision = load_trt_module_from_plan(&context.backend, &vision_plan, "vision.plan", options);
    decode.module->keep_alive(stream_owner);
    prefill.module->keep_alive(stream_owner);
    vision.module->keep_alive(stream_owner);

    const auto num_layers = llava_factory::require_positive_int(config, "num_layers");
    const auto max_cache_length = llava_factory::require_positive_int(config, "max_cache_length");
    auto kv_names = llava_factory::build_kv_names(config, num_layers);
    const auto cache_k_name = kv_names.cache_k.front();
    const auto cache_shape = decode.module->tensor_shape(cache_k_name);
    if (cache_shape.size() < 2 || cache_shape[1] <= 0)
        throw std::runtime_error("LLaVA decode engine has invalid KV-cache shape");
    const auto kv_dim = static_cast<std::int32_t>(cache_shape[1]);
    const auto cache_dtype = decode.module->tensor_dtype(cache_k_name);
    std::unique_ptr<LlavaInferenceState> state =
        std::make_unique<LlavaKvCache>(num_layers, max_cache_length, kv_dim,
                                       decode.module->stream(), cache_dtype, std::move(kv_names));

    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("LLaVA bundle does not contain its required tokenizer");

    LlavaConfig model;
    model.vocab_size = llava_factory::require_positive_int(config, "vocab_size");
    model.id_bos = llava_factory::require_int(config, "id_bos");
    model.id_eos = llava_factory::require_int(config, "id_eos");
    model.eos_token_ids =
        config.value("eos_token_ids", model.id_eos >= 0 ? std::vector<int32_t>{model.id_eos}
                                                        : std::vector<int32_t>{});
    for (const auto token : model.eos_token_ids)
        if (token < 0 || token >= model.vocab_size)
            throw std::runtime_error("LLaVA runtime.json has invalid EOS token");
    model.image_token_id = llava_factory::require_int(config, "image_token_id");
    model.vision_output_dim = llava_factory::require_positive_int(config, "vision_output_dim");
    model.has_position_input = decode.module->has_input("position_id");
    model.num_layers = num_layers;
    model.prefill_max_length = llava_factory::require_positive_int(config, "prefill_max_length");
    if (config.contains("rope_long_factor_threshold"))
        model.rope_long_factor_threshold =
            llava_factory::require_int(config, "rope_long_factor_threshold");
    const auto& io = config.at("io_map");
    model.present_k_pattern = io.at("present_k_pattern").get<std::string>();
    model.present_v_pattern = io.at("present_v_pattern").get<std::string>();
    auto preprocess = llava_parse_preprocess_config(runtime_text);
    return new LlavaPipeline(std::move(decode.module), std::move(vision.module), std::move(state),
                             std::move(model), std::move(preprocess), options.stream,
                             std::move(tokenizer), "", std::move(prefill.module));
}
