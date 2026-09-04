/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_vl/runtime/cuda_stream.h"
#include "families/qwen_vl/runtime/pipeline.h"
#include "families/qwen_vl/runtime/plugin_helpers.h"
#include "families/qwen_vl/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"

#include <cstdlib>
#include <dlfcn.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::qwen_vl_factory {
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
        throw std::runtime_error("Qwen-VL TP runtime requires OMPI_COMM_WORLD_RANK");
    char* end = nullptr;
    const long rank = std::strtol(text, &end, 10);
    if (*end != '\0' || rank < 0 || rank >= size)
        throw std::runtime_error("Qwen-VL RANK is outside tensor_parallel_size");
    return static_cast<std::int32_t>(rank);
}

QwenVlKvCacheNames kv_names(const nlohmann::json& config, std::int32_t layers) {
    const auto& io = config.at("io_map");
    const auto cache_k = io.at("cache_k_pattern").get<std::string>();
    const auto cache_v = io.at("cache_v_pattern").get<std::string>();
    const auto present_k = io.at("present_k_pattern").get<std::string>();
    const auto present_v = io.at("present_v_pattern").get<std::string>();
    QwenVlKvCacheNames names;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        names.cache_k.push_back(qwen_vl_expand_layer_name(cache_k, layer));
        names.cache_v.push_back(qwen_vl_expand_layer_name(cache_v, layer));
        names.present_k.push_back(qwen_vl_expand_layer_name(present_k, layer));
        names.present_v.push_back(qwen_vl_expand_layer_name(present_v, layer));
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
} // namespace trtmc::qwen_vl_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    const auto runtime = qwen_vl_factory::section_text(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime);
    const auto tp_size = config.at("tensor_parallel_size").get<std::int32_t>();
    if (tp_size <= 0)
        throw std::runtime_error("Qwen-VL tensor_parallel_size must be positive");
    const auto rank = qwen_vl_factory::require_rank(tp_size);
    const std::string decode_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& decode_plan =
        qwen_vl_factory::require_section(context.reader, decode_section.c_str());
    const auto& prefill_plan = qwen_vl_factory::require_section(context.reader, "prefill.plan");
    const auto& vision_plan = qwen_vl_factory::require_section(context.reader, "vision.plan");

    auto stream = std::make_shared<QwenVlCudaStream>();
    if (!stream->ok())
        throw std::runtime_error("Qwen-VL failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream->get();
    auto decode =
        load_trt_module_from_plan(&context.backend, &decode_plan, decode_section.c_str(), options);
    auto prefill =
        load_trt_module_from_plan(&context.backend, &prefill_plan, "prefill.plan", options);
    auto vision = load_trt_module_from_plan(&context.backend, &vision_plan, "vision.plan", options);
    decode.module->keep_alive(stream);
    prefill.module->keep_alive(stream);
    vision.module->keep_alive(stream);

    const auto layers = config.at("num_layers").get<std::int32_t>();
    const auto max_cache = config.at("max_cache_length").get<std::int32_t>();
    if (layers <= 0 || max_cache <= 0)
        throw std::runtime_error("Qwen-VL runtime.json has invalid cache geometry");
    auto names = qwen_vl_factory::kv_names(config, layers);
    const auto cache_name = names.cache_k.front();
    const auto shape = decode.module->tensor_shape(cache_name);
    if (shape.size() < 2 || shape[1] <= 0)
        throw std::runtime_error("Qwen-VL decoder cache shape is invalid");
    auto state = std::make_unique<QwenVlKvCache>(
        layers, max_cache, static_cast<std::int32_t>(shape[1]), options.stream,
        decode.module->tensor_dtype(cache_name), std::move(names));

    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("Qwen-VL bundle does not contain its required tokenizer");
    QwenVlConfig model;
    model.vocab_size = config.at("vocab_size").get<std::int32_t>();
    model.id_bos = config.at("id_bos").get<std::int32_t>();
    model.id_eos = config.at("id_eos").get<std::int32_t>();
    model.id_eos_ids = config.at("id_eos_ids").get<std::vector<std::int32_t>>();
    model.image_token_id = config.at("image_token_id").get<std::int32_t>();
    model.vision_output_dim = config.at("vision_output_dim").get<std::int32_t>();
    model.has_position_input = decode.module->has_input("position_id");
    model.num_layers = layers;
    model.prefill_max_length = config.at("prefill_max_length").get<std::int32_t>();
    model.present_k_pattern = config.at("io_map").at("present_k_pattern").get<std::string>();
    model.present_v_pattern = config.at("io_map").at("present_v_pattern").get<std::string>();
    auto preprocess = qwen_vl_parse_preprocess_config(runtime);
    auto adapters = std::make_shared<qwen_vl::LoraAdapterCache>(
        qwen_vl_factory::lora_contract(*decode.module), options.stream);
    return new QwenVlPipeline(std::move(decode.module), std::move(vision.module), std::move(state),
                              std::move(model), std::move(preprocess), options.stream,
                              std::move(tokenizer), "", std::move(prefill.module),
                              std::move(adapters));
}
