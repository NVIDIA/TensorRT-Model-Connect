/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/cosyvoice3/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::cosyvoice3 {
namespace {

std::vector<char> require_section(const BundleReader& reader, const std::string& name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("CosyVoice3 bundle is missing " + name);
    return reader.read_section(name);
}

std::string require_text(const BundleReader& reader, const char* name) {
    const auto bytes = require_section(reader, name);
    return {bytes.begin(), bytes.end()};
}

} // namespace

ITask* create(const FamilyContext& context) {
    if (context.reader.info().backend != "trt")
        throw std::runtime_error("CosyVoice3 requires the TensorRT backend");
    if (context.reader.info().task != IAudioGeneration::kTask)
        throw std::runtime_error("CosyVoice3 requires task=audio_generation");

    const auto config = nlohmann::json::parse(require_text(context.reader, "config.json"));
    const int schema = config.at("cosyvoice3_schema");
    if (schema != 2 || config.at("precision") != "fp32")
        throw std::runtime_error("Unsupported CosyVoice3 bundle");

    const auto& family = config.at("cosyvoice3");
    Settings settings;
    settings.instruction = family.at("instruction");
    settings.transcript = family.at("transcript");
    settings.max_context = family.at("max_context");
    settings.max_tokens = family.at("max_tokens");
    settings.greedy = family.at("greedy");
    settings.total_tokens = family.at("total_tokens");
    auto coefficients = nlohmann::json::parse(require_text(context.reader, "frontend.json"));

    const auto tokenizer_bytes = require_section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_bytes.data(), tokenizer_bytes.size(), false);
    if (tokenizer == nullptr)
        throw std::runtime_error("Cannot construct the CosyVoice3 tokenizer");

    auto factory = [reader = context.reader, backend = &context.backend](const std::string& name) {
        const auto bytes = require_section(reader, name + ".plan");
        auto module = backend->create_module(bytes.data(), bytes.size(), ModuleCreateOptions{});
        if (module == nullptr || !module->ok())
            throw std::runtime_error("Cannot load CosyVoice3 " + name);
        return module;
    };
    return new Pipeline(std::move(settings), std::move(tokenizer), std::move(factory),
                        std::move(coefficients));
}

} // namespace trtmc::cosyvoice3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("cosyvoice3 does not support --kv-cache-size");
    return trtmc::cosyvoice3::create(context);
}
