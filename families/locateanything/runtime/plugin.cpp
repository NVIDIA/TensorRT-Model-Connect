/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/locateanything/runtime/cuda_stream.h"
#include "families/locateanything/runtime/distributed_runtime.h"
#include "families/locateanything/runtime/pipeline.h"
#include "families/locateanything/runtime/plugin_helpers.h"
#include "families/locateanything/runtime/tensor_names.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::locateanything_factory {
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
        throw std::runtime_error(std::string("LocateAnything runtime.json requires integer ") +
                                 key);
    return json.at(key).get<std::int32_t>();
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    const auto value = require_int(json, key);
    if (value <= 0)
        throw std::runtime_error(std::string("LocateAnything runtime.json has invalid ") + key);
    return value;
}

LocateanythingKvCacheNames build_kv_names(const nlohmann::json& config, std::int32_t num_layers) {
    const auto& io = config.at("io_map");
    const auto cache_k = io.at("cache_k_pattern").get<std::string>();
    const auto cache_v = io.at("cache_v_pattern").get<std::string>();
    const auto present_k = io.at("present_k_pattern").get<std::string>();
    const auto present_v = io.at("present_v_pattern").get<std::string>();
    LocateanythingKvCacheNames names;
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        names.cache_k.push_back(locateanything_expand_layer_name(cache_k, layer));
        names.cache_v.push_back(locateanything_expand_layer_name(cache_v, layer));
        names.present_k.push_back(locateanything_expand_layer_name(present_k, layer));
        names.present_v.push_back(locateanything_expand_layer_name(present_v, layer));
    }
    return names;
}

} // namespace
} // namespace trtmc::locateanything_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("locateanything does not support --kv-cache-size");
    using namespace trtmc;
    const std::string runtime_text =
        locateanything_factory::section_text(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(runtime_text);
    if (!config.is_object())
        throw std::runtime_error("LocateAnything runtime.json must be an object");

    const auto tp_size =
        locateanything_factory::require_positive_int(config, "tensor_parallel_size");
    const auto group = locateanything::initialize_tensor_parallel_group(tp_size);
    const std::string decode_section =
        tp_size == 1 ? "engine.plan" : "engine.rank" + std::to_string(group.rank) + ".plan";
    const auto& decode_plan =
        locateanything_factory::require_section(context.reader, decode_section.c_str());
    const auto& prefill_plan =
        locateanything_factory::require_section(context.reader, "prefill.plan");
    const auto& vision_plan =
        locateanything_factory::require_section(context.reader, "vision.plan");

    auto stream_owner = std::make_shared<LocateAnythingCudaStream>();
    if (!stream_owner->ok())
        throw std::runtime_error("LocateAnything failed to create its CUDA stream");
    ModuleCreateOptions options{};
    options.stream = stream_owner->get();
    ModuleCreateOptions text_options = options;
    if (tp_size > 1) {
        text_options.distributed_communicator = group.communicator;
        text_options.distributed_owner = group.owner;
    }
    auto decode = load_trt_module_from_plan(&context.backend, &decode_plan, decode_section.c_str(),
                                            text_options);
    auto prefill =
        load_trt_module_from_plan(&context.backend, &prefill_plan, "prefill.plan", text_options);
    auto vision = load_trt_module_from_plan(&context.backend, &vision_plan, "vision.plan", options);
    decode.module->keep_alive(stream_owner);
    prefill.module->keep_alive(stream_owner);
    vision.module->keep_alive(stream_owner);

    const auto num_layers = locateanything_factory::require_positive_int(config, "num_layers");
    const auto max_cache_length =
        locateanything_factory::require_positive_int(config, "max_cache_length");
    auto kv_names = locateanything_factory::build_kv_names(config, num_layers);
    const auto cache_k_name = kv_names.cache_k.front();
    const auto cache_shape = decode.module->tensor_shape(cache_k_name);
    if (cache_shape.size() < 2 || cache_shape[1] <= 0)
        throw std::runtime_error("LocateAnything decode engine has invalid KV-cache shape");
    const auto kv_dim = static_cast<std::int32_t>(cache_shape[1]);
    const auto cache_dtype = decode.module->tensor_dtype(cache_k_name);
    std::unique_ptr<LocateanythingInferenceState> state = std::make_unique<LocateanythingKvCache>(
        num_layers, max_cache_length, kv_dim, decode.module->stream(), cache_dtype,
        std::move(kv_names));

    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("LocateAnything bundle does not contain its required tokenizer");

    LocateAnythingConfig model;
    model.vocab_size = locateanything_factory::require_positive_int(config, "vocab_size");
    model.id_bos = locateanything_factory::require_int(config, "id_bos");
    model.id_eos = locateanything_factory::require_int(config, "id_eos");
    model.image_token_id = locateanything_factory::require_int(config, "image_token_id");
    model.vision_output_dim =
        locateanything_factory::require_positive_int(config, "vision_output_dim");
    model.has_position_input = decode.module->has_input("position_id");
    model.num_layers = num_layers;
    model.prefill_max_length =
        locateanything_factory::require_positive_int(config, "prefill_max_length");
    const auto& io = config.at("io_map");
    model.present_k_pattern = io.at("present_k_pattern").get<std::string>();
    model.present_v_pattern = io.at("present_v_pattern").get<std::string>();
    auto preprocess = locateanything_parse_preprocess_config(runtime_text);
    return new LocateAnythingPipeline(
        std::move(decode.module), std::move(vision.module), std::move(state), std::move(model),
        std::move(preprocess), options.stream, std::move(tokenizer), "", std::move(prefill.module));
}
