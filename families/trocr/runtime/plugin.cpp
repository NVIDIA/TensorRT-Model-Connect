/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/trocr/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::trocr {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("trocr bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::int32_t positive(const nlohmann::json& json, const char* name) {
    const auto value = json.at(name).get<std::int32_t>();
    if (value <= 0)
        throw std::runtime_error(std::string("trocr runtime.json has invalid ") + name);
    return value;
}

std::array<float, 3> triple(const nlohmann::json& json, const char* name) {
    const auto values = json.at(name).get<std::vector<float>>();
    if (values.size() != 3)
        throw std::runtime_error(std::string("trocr runtime.json has invalid ") + name);
    return {values[0], values[1], values[2]};
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    RuntimeConfig config;
    config.preprocess.height = positive(json, "image_height");
    config.preprocess.width = positive(json, "image_width");
    if (json.at("resample").get<std::int32_t>() != 2)
        throw std::runtime_error("trocr runtime.json names a resize filter other than bilinear");
    config.preprocess.rescale_factor = json.at("rescale_factor").get<double>();
    config.preprocess.image_mean = triple(json, "image_mean");
    config.preprocess.image_std = triple(json, "image_std");
    config.decoder_start_token_id = json.at("decoder_start_token_id").get<std::int32_t>();
    config.eos_token_id = json.at("eos_token_id").get<std::int32_t>();
    config.special_token_ids = json.at("special_token_ids").get<std::vector<std::int32_t>>();
    config.vocab_size = positive(json, "vocab_size");
    config.layers = positive(json, "decoder_layers");
    config.heads = positive(json, "decoder_heads");
    config.head_dim = positive(json, "head_dim");
    config.encoder_sequence_length = positive(json, "encoder_sequence_length");
    config.max_sequence_length = positive(json, "max_sequence_length");
    const auto precision = json.at("precision").get<std::string>();
    if (precision != "fp32")
        throw std::runtime_error("trocr runtime.json names an unsupported precision: " + precision);
    config.work_dtype = DType::kFloat32;
    for (const auto id : {config.decoder_start_token_id, config.eos_token_id}) {
        if (id < 0 || id >= config.vocab_size)
            throw std::runtime_error("trocr runtime.json names a token outside the vocabulary");
    }
    return config;
}

std::vector<std::string> parse_vocabulary(const std::vector<char>& data) {
    return nlohmann::json::parse(data.begin(), data.end()).get<std::vector<std::string>>();
}

ModuleExternalBinding binding(const char* name, DeviceTensor& tensor) {
    return {name, tensor.data(), tensor.nbytes()};
}

std::unique_ptr<ITrtModule> load(IBackend& backend, const std::vector<char>& plan,
                                 const std::vector<ModuleExternalBinding>& bindings,
                                 const char* name) {
    auto module =
        backend.create_module_prebound(plan.data(), plan.size(), ModuleCreateOptions{}, bindings);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("trocr ") + name + " engine failed to load");
    return module;
}

// The engines must agree with runtime.json, which sizes every shared buffer
// and the cache rows the runtime writes.
void check_tensor(const ITrtModule& module, const std::string& name,
                  const std::vector<int64_t>& shape, DType dtype) {
    if (module.tensor_shape(name) != shape || module.tensor_dtype(name) != dtype)
        throw std::runtime_error("trocr engine tensor " + name +
                                 " does not match the bundle's runtime.json");
}

void check_engines(const ITrtModule& encoder, const ITrtModule& decoder,
                   const RuntimeConfig& config) {
    const std::int64_t layers = config.layers, heads = config.heads, dim = config.head_dim;
    const std::vector<int64_t> cross{layers, heads, config.encoder_sequence_length, dim};
    const std::vector<int64_t> cache{layers, heads, config.max_sequence_length, dim};
    const std::vector<int64_t> row{layers, heads, 1, dim};
    check_tensor(encoder, "pixel_values", {1, 3, config.preprocess.height, config.preprocess.width},
                 DType::kFloat32);
    for (const auto* name : {"cross_key", "cross_value"}) {
        check_tensor(encoder, name, cross, config.work_dtype);
        check_tensor(decoder, name, cross, config.work_dtype);
    }
    for (const auto* name : {"cache_key", "cache_value"})
        check_tensor(decoder, name, cache, config.work_dtype);
    for (const auto* name : {"new_key", "new_value"})
        check_tensor(decoder, name, row, config.work_dtype);
    check_tensor(decoder, "token_id", {1}, DType::kInt32);
    check_tensor(decoder, "position", {1}, DType::kInt32);
    check_tensor(decoder, "logits", {1, config.vocab_size}, DType::kFloat32);
}

} // namespace
} // namespace trtmc::trocr

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc::trocr;
    // The decode cache is sized when the bundle is built (max_sequence_length).
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("trocr does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::internal::IImagesTextToText::kTask)
        throw std::runtime_error("trocr bundle declares an unsupported task: " +
                                 context.reader.info().task);
    auto config = parse_config(require_section(context.reader, "runtime.json"));
    ByteLevelDetokenizer detokenizer(
        parse_vocabulary(require_section(context.reader, "id_to_token.json")),
        config.special_token_ids);
    const auto encoder_plan = require_section(context.reader, "encoder.plan");
    const auto decoder_plan = require_section(context.reader, "decoder.plan");

    auto buffers = std::make_unique<DecodeBuffers>(allocate_buffers(config));
    auto encoder = load(
        context.backend, encoder_plan,
        {binding("cross_key", buffers->cross_key), binding("cross_value", buffers->cross_value)},
        "encoder");
    auto decoder = load(
        context.backend, decoder_plan,
        {binding("cache_key", buffers->cache_key), binding("cache_value", buffers->cache_value),
         binding("cross_key", buffers->cross_key), binding("cross_value", buffers->cross_value),
         binding("new_key", buffers->new_key), binding("new_value", buffers->new_value)},
        "decoder");
    check_engines(*encoder, *decoder, config);
    return new TrocrPipeline(std::move(config), std::move(buffers), std::move(encoder),
                             std::move(decoder), std::move(detokenizer));
}
