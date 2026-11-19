/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/fp32_kernels.h"
#include "families/blip/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
std::int32_t config_integer(const nlohmann::json& config, const char* key) {
    const auto& value = config.at(key);
    if (value.is_number_unsigned()) {
        const auto integer = value.get<std::uint64_t>();
        if (integer <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            return static_cast<std::int32_t>(integer);
    } else if (value.is_number_integer()) {
        const auto integer = value.get<std::int64_t>();
        if (integer >= std::numeric_limits<std::int32_t>::min() &&
            integer <= std::numeric_limits<std::int32_t>::max())
            return static_cast<std::int32_t>(integer);
    }
    throw std::invalid_argument(std::string("blip runtime ") + key + " must be an int32 integer");
}
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes)
        throw std::invalid_argument("blip does not support --kv-cache-size");
    if (context.reader.info().task != "images_text_to_text")
        throw std::invalid_argument("blip requires images_text_to_text");
    auto section = [&](const char* name) {
        const auto* found = context.reader.find_section(name);
        if (!found || !found->length)
            throw std::runtime_error(std::string("blip missing bundle section: ") + name);
        return context.reader.read_section(name);
    };
    const auto data = section("runtime.json");
    const auto cfg = nlohmann::json::parse(data.begin(), data.end());
    const auto vision_backend = cfg.value("vision_kernel_backend", std::string("trt"));
    if (vision_backend == "cuda_fp32")
        trtmc::blip::initialize_fp32_kernels();
    else if (vision_backend != "trt")
        throw std::invalid_argument("blip has an unsupported vision kernel backend");
    trtmc::blip::RuntimeConfig config;
    config.preprocess.input_image_h = config_integer(cfg, "image_size");
    config.preprocess.input_image_w = config_integer(cfg, "image_size");
    config.preprocess.image_mean = cfg.at("image_mean").get<std::vector<float>>();
    config.preprocess.image_std = cfg.at("image_std").get<std::vector<float>>();
    config.image_tokens = config_integer(cfg, "image_tokens");
    config.vision_width = config_integer(cfg, "vision_width");
    config.vocab_size = config_integer(cfg, "vocab_size");
    config.max_length = config_integer(cfg, "max_length");
    config.bos_token_id = config_integer(cfg, "bos_token_id");
    config.eos_token_id = config_integer(cfg, "eos_token_id");
    const auto tokenizer_data = section("tokenizer.json");
    auto tokenizer =
        trtmc::blip::CreateWordPieceTokenizer(tokenizer_data.data(), tokenizer_data.size());
    auto load = [&](const char* name) {
        auto plan = section(name);
        return context.backend.create_module(plan.data(), plan.size(),
                                             trtmc::ModuleCreateOptions{});
    };
    auto vision = load("vision.plan");
    auto decoder = load("decoder.plan");
    return new trtmc::blip::Pipeline(std::move(vision), std::move(decoder), std::move(tokenizer),
                                     std::move(config));
}
