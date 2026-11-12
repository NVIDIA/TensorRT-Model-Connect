/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_vl_moe/runtime/cuda_stream.h"
#include "families/qwen3_vl_moe/runtime/pipeline.h"
#include "families/qwen3_vl_moe/runtime/plugin_helpers.h"
#include "families/qwen3_vl_moe/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::qwen3_vl_moe_factory {
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

Qwen3VlMoeKvCacheNames kv_names(const nlohmann::json& config, std::int32_t layers) {
    const auto& io = config.at("io_map");
    const auto cache_k = io.at("cache_k_pattern").get<std::string>();
    const auto cache_v = io.at("cache_v_pattern").get<std::string>();
    const auto present_k = io.at("present_k_pattern").get<std::string>();
    const auto present_v = io.at("present_v_pattern").get<std::string>();
    Qwen3VlMoeKvCacheNames names;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        names.cache_k.push_back(qwen3_vl_moe_expand_layer_name(cache_k, layer));
        names.cache_v.push_back(qwen3_vl_moe_expand_layer_name(cache_v, layer));
        names.present_k.push_back(qwen3_vl_moe_expand_layer_name(present_k, layer));
        names.present_v.push_back(qwen3_vl_moe_expand_layer_name(present_v, layer));
    }
    return names;
}

bool is_lora_input(const std::string& name) {
    return name.rfind("lora_a_", 0) == 0 || name.rfind("lora_b_", 0) == 0;
}

std::vector<TensorInfo> lora_contract(const ITrtModule& module) {
    std::vector<TensorInfo> contract;
    for (const auto& info : module.input_info())
        if (is_lora_input(info.name))
            contract.push_back(info);
    return contract;
}

} // namespace
} // namespace trtmc::qwen3_vl_moe_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_vl_moe does not support --kv-cache-size");
    using namespace trtmc;
    const auto runtime = qwen3_vl_moe_factory::section_text(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime);
    const auto tp_size = config.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size != 1)
        throw std::invalid_argument("qwen3_vl_moe supports one GPU");

    auto stream = std::make_shared<Qwen3VlMoeCudaStream>();
    if (!stream->ok())
        throw std::runtime_error("Qwen3-VL-MoE failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream->get();
    std::unique_ptr<ITrtModule> decode;
    std::unique_ptr<ITrtModule> prefill;
    const auto decoder_plan = qwen3_vl_moe_factory::require_section(context.reader, "engine.plan");
    auto modules = context.backend.create_dual_profile_modules(decoder_plan.data(),
                                                               decoder_plan.size(), options);
    if (!modules.decode || !modules.decode->ok() || !modules.prefill || !modules.prefill->ok())
        throw std::runtime_error("Qwen3-VL-MoE could not load prefill and decode contexts");
    decode = std::move(modules.decode);
    prefill = std::move(modules.prefill);
    const auto vision_plan = qwen3_vl_moe_factory::require_section(context.reader, "vision.plan");
    ModuleCreateOptions vision_options{};
    vision_options.stream = stream->get();
    auto vision =
        load_trt_module_from_plan(&context.backend, &vision_plan, "vision.plan", vision_options);
    decode->keep_alive(stream);
    prefill->keep_alive(stream);
    vision.module->keep_alive(stream);

    const auto layers = config.at("num_layers").get<std::int32_t>();
    const auto max_cache = config.at("max_cache_length").get<std::int32_t>();
    if (layers <= 0 || max_cache <= 0)
        throw std::runtime_error("Qwen3-VL-MoE runtime.json has invalid cache geometry");
    auto names = qwen3_vl_moe_factory::kv_names(config, layers);
    const auto cache_name = names.cache_k.front();
    const auto shape = decode->tensor_shape(cache_name);
    if (shape.size() < 2 || shape[1] <= 0)
        throw std::runtime_error("Qwen3-VL-MoE decoder cache shape is invalid");
    auto state = std::make_unique<Qwen3VlMoeKvCache>(
        layers, max_cache, static_cast<std::int32_t>(shape[1]), options.stream,
        decode->tensor_dtype(cache_name), std::move(names));

    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Qwen3-VL-MoE bundle does not contain its required tokenizer");
    Qwen3VlMoeConfig model;
    model.vocab_size = config.at("vocab_size").get<std::int32_t>();
    model.id_bos = config.at("id_bos").get<std::int32_t>();
    model.id_eos = config.at("id_eos").get<std::int32_t>();
    model.id_eos_ids = config.at("id_eos_ids").get<std::vector<std::int32_t>>();
    model.image_token_id = config.at("image_token_id").get<std::int32_t>();
    model.vision_output_dim = config.at("vision_output_dim").get<std::int32_t>();
    model.has_position_input = decode->has_input("position_id");
    model.num_layers = layers;
    model.prefill_max_length = config.at("prefill_max_length").get<std::int32_t>();
    model.present_k_pattern = config.at("io_map").at("present_k_pattern").get<std::string>();
    model.present_v_pattern = config.at("io_map").at("present_v_pattern").get<std::string>();
    auto preprocess = qwen3_vl_moe_parse_preprocess_config(runtime);
    auto adapters = std::make_shared<qwen3_vl_moe::LoraAdapterCache>(
        qwen3_vl_moe_factory::lora_contract(*decode), options.stream);
    return new Qwen3VlMoePipeline(std::move(decode), std::move(vision.module), std::move(state),
                                  std::move(model), std::move(preprocess), options.stream,
                                  std::move(tokenizer), "", std::move(prefill),
                                  std::move(adapters));
}
