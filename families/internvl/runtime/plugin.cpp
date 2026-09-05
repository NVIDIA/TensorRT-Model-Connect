/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/internvl/runtime/cuda_stream.h"
#include "families/internvl/runtime/distributed_runtime.h"
#include "families/internvl/runtime/pipeline.h"
#include "families/internvl/runtime/plugin_helpers.h"
#include "families/internvl/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::internvl_factory {
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
        throw std::runtime_error(std::string("InternVL runtime.json requires integer ") + key);
    return json.at(key).get<std::int32_t>();
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    const auto value = require_int(json, key);
    if (value <= 0)
        throw std::runtime_error(std::string("InternVL runtime.json has invalid ") + key);
    return value;
}

InternvlKvCacheNames build_kv_names(const nlohmann::json& config, std::int32_t num_layers) {
    const auto& io = config.at("io_map");
    const auto cache_k = io.at("cache_k_pattern").get<std::string>();
    const auto cache_v = io.at("cache_v_pattern").get<std::string>();
    const auto present_k = io.at("present_k_pattern").get<std::string>();
    const auto present_v = io.at("present_v_pattern").get<std::string>();
    InternvlKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(internvl_expand_layer_name(cache_k, layer));
        names.cache_v.push_back(internvl_expand_layer_name(cache_v, layer));
        names.present_k.push_back(internvl_expand_layer_name(present_k, layer));
        names.present_v.push_back(internvl_expand_layer_name(present_v, layer));
    }
    return names;
}

} // namespace
} // namespace trtmc::internvl_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("internvl does not support --kv-cache-size");
    using namespace trtmc;
    const std::string runtime_text = internvl_factory::section_text(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime_text);
    if (!config.is_object())
        throw std::runtime_error("InternVL runtime.json must be an object");

    const auto tp_size = internvl_factory::require_positive_int(config, "tensor_parallel_size");
    auto group = internvl::initialize_tensor_parallel_group(tp_size);

    auto stream_owner = std::make_shared<InternVlCudaStream>();
    if (!stream_owner->ok())
        throw std::runtime_error("InternVL failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream_owner->get();
    std::unique_ptr<ITrtModule> decode;
    std::unique_ptr<ITrtModule> prefill;
    if (tp_size > 1) {
        const std::string section = "engine.rank" + std::to_string(group.rank) + ".plan";
        const auto& plan = internvl_factory::require_section(context.reader, section.c_str());
        options.distributed_communicator = group.communicator;
        options.distributed_owner = group.owner;
        auto dual = context.backend.create_dual_profile_modules(plan.data(), plan.size(), options);
        if (dual.decode == nullptr || !dual.decode->ok() || dual.prefill == nullptr ||
            !dual.prefill->ok())
            throw std::runtime_error("InternVL TP engine did not create both profiles");
        decode = std::move(dual.decode);
        prefill = std::move(dual.prefill);
    } else {
        const auto& decode_plan = internvl_factory::require_section(context.reader, "engine.plan");
        const auto& prefill_plan =
            internvl_factory::require_section(context.reader, "prefill.plan");
        decode = load_trt_module_from_plan(&context.backend, &decode_plan, "engine.plan", options)
                     .module;
        prefill =
            load_trt_module_from_plan(&context.backend, &prefill_plan, "prefill.plan", options)
                .module;
    }
    const auto& vision_plan = internvl_factory::require_section(context.reader, "vision.plan");
    options.distributed_communicator = nullptr;
    options.distributed_owner.reset();
    auto vision = load_trt_module_from_plan(&context.backend, &vision_plan, "vision.plan", options);
    decode->keep_alive(stream_owner);
    prefill->keep_alive(stream_owner);
    vision.module->keep_alive(stream_owner);

    const auto num_layers = internvl_factory::require_positive_int(config, "num_layers");
    const auto max_cache_length =
        internvl_factory::require_positive_int(config, "max_cache_length");
    auto kv_names = internvl_factory::build_kv_names(config, num_layers);
    const auto cache_k_name = kv_names.cache_k.front();
    const auto cache_shape = decode->tensor_shape(cache_k_name);
    if (cache_shape.size() < 2 || cache_shape[1] <= 0)
        throw std::runtime_error("InternVL decode engine has invalid KV-cache shape");
    const auto kv_dim = static_cast<std::int32_t>(cache_shape[1]);
    const auto cache_dtype = decode->tensor_dtype(cache_k_name);
    std::unique_ptr<InternvlInferenceState> state = std::make_unique<InternvlKvCache>(
        num_layers, max_cache_length, kv_dim, decode->stream(), cache_dtype, std::move(kv_names));

    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("InternVL bundle does not contain its required tokenizer");

    InternVlConfig model;
    model.vocab_size = internvl_factory::require_positive_int(config, "vocab_size");
    model.id_bos = internvl_factory::require_int(config, "id_bos");
    model.id_eos = internvl_factory::require_int(config, "id_eos");
    model.image_token_id = internvl_factory::require_int(config, "image_token_id");
    model.vision_output_dim = internvl_factory::require_positive_int(config, "vision_output_dim");
    model.has_position_input = decode->has_input("position_id");
    model.num_layers = num_layers;
    model.prefill_max_length = internvl_factory::require_positive_int(config, "prefill_max_length");
    const auto& io = config.at("io_map");
    model.present_k_pattern = io.at("present_k_pattern").get<std::string>();
    model.present_v_pattern = io.at("present_v_pattern").get<std::string>();
    auto preprocess = internvl_parse_preprocess_config(runtime_text);
    return new InternVlPipeline(std::move(decode), std::move(vision.module), std::move(state),
                                std::move(model), std::move(preprocess), options.stream,
                                std::move(tokenizer), "", std::move(prefill));
}
