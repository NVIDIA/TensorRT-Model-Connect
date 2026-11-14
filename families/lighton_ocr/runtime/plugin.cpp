/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::lighton_ocr {
namespace {

std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* item = reader.find_section(name);
    if (!item || !item->length)
        throw std::runtime_error(std::string("LightOnOCR bundle section is missing: ") + name);
    return reader.read_section(name);
}

RuntimeConfig runtime_config(const std::vector<char>& bytes) {
    const auto json = nlohmann::json::parse(bytes.begin(), bytes.end());
    if (json.at("format_version") != 1 || json.at("precision") != "fp32")
        throw std::runtime_error("LightOnOCR requires its FP32 runtime format version 1");
    RuntimeConfig config;
    config.max_sequence_length = json.at("max_sequence_length");
    config.max_new_tokens = json.at("max_new_tokens");
    config.hidden_size = json.at("hidden_size");
    config.vocab_size = json.at("vocab_size");
    config.num_hidden_layers = json.at("num_hidden_layers");
    config.num_key_value_heads = json.at("num_key_value_heads");
    config.head_dim = json.at("head_dim");
    config.image_token_id = json.at("image_token_id");
    config.eos_token_ids = json.at("eos_token_ids").get<std::vector<std::int32_t>>();
    config.prompt_prefix_ids = json.at("prompt_prefix_ids").get<std::vector<std::int32_t>>();
    config.prompt_suffix_ids = json.at("prompt_suffix_ids").get<std::vector<std::int32_t>>();
    if (config.max_sequence_length <= 0 || config.max_new_tokens <= 0 ||
        config.max_new_tokens > config.max_sequence_length || config.hidden_size <= 0 ||
        config.vocab_size <= 0 || config.num_hidden_layers <= 0 ||
        config.num_key_value_heads <= 0 || config.head_dim <= 0 || config.image_token_id < 0 ||
        config.image_token_id >= config.vocab_size)
        throw std::runtime_error("LightOnOCR runtime dimensions or token bounds are invalid");
    for (const auto* ids :
         {&config.eos_token_ids, &config.prompt_prefix_ids, &config.prompt_suffix_ids})
        if (ids->empty() || !std::all_of(ids->begin(), ids->end(), [&](auto id) {
                return id >= 0 && id < config.vocab_size;
            }))
            throw std::runtime_error("LightOnOCR runtime prompt or EOS tokens are invalid");
    const auto& image = json.at("image_processor");
    config.image.patch_size = image.at("patch_size");
    config.image.spatial_merge_size = image.at("spatial_merge_size");
    config.image.max_image_size = image.at("max_image_size");
    config.image.rescale_factor = image.at("rescale_factor");
    config.image.mean = image.at("image_mean").get<std::array<float, 3>>();
    config.image.standard_deviation = image.at("image_std").get<std::array<float, 3>>();
    return config;
}

std::unique_ptr<ITrtModule> engine(IBackend& backend, const BundleReader& reader,
                                   const char* name) {
    const auto plan = section(reader, name);
    auto module = backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error(std::string("LightOnOCR engine failed to load: ") + name);
    module->set_timing_label(std::string("lighton_ocr/") + name);
    return module;
}

} // namespace
} // namespace trtmc::lighton_ocr

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc::lighton_ocr;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lighton_ocr does not support --kv-cache-size");
    if (context.reader.info().family != "lighton_ocr" ||
        context.reader.info().task != trtmc::internal::IImagesTextToText::kTask)
        throw std::runtime_error("LightOnOCR bundle family or primary task is invalid");
    auto config = runtime_config(section(context.reader, "runtime.json"));
    const auto tokenizer_data = section(context.reader, "tokenizer.json");
    Tokenizer tokenizer(std::string_view(tokenizer_data.data(), tokenizer_data.size()),
                        config.vocab_size);
    const auto bytes = section(context.reader, "embedding.bin");
    const auto count = static_cast<std::size_t>(config.hidden_size) * config.vocab_size;
    if (count > SIZE_MAX / sizeof(float) || bytes.size() != count * sizeof(float))
        throw std::runtime_error("LightOnOCR embedding bytes do not match the runtime dimensions");
    std::vector<float> embedding(count);
    std::memcpy(embedding.data(), bytes.data(), bytes.size());
    auto vision = engine(context.backend, context.reader, "vision.plan");
    auto decoder = engine(context.backend, context.reader, "decoder.plan");
    return new Pipeline(std::move(vision), std::move(decoder), std::move(embedding),
                        std::move(tokenizer), std::move(config));
}
