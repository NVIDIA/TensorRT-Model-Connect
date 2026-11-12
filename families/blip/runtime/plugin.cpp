/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

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
    trtmc::blip::RuntimeConfig config;
    config.preprocess.input_image_h = cfg.at("image_size");
    config.preprocess.input_image_w = cfg.at("image_size");
    config.preprocess.image_mean = cfg.at("image_mean").get<std::vector<float>>();
    config.preprocess.image_std = cfg.at("image_std").get<std::vector<float>>();
    config.image_tokens = cfg.at("image_tokens");
    config.vision_width = cfg.at("vision_width");
    config.vocab_size = cfg.at("vocab_size");
    config.max_length = cfg.at("max_length");
    config.bos_token_id = cfg.at("bos_token_id");
    config.eos_token_id = cfg.at("eos_token_id");
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
