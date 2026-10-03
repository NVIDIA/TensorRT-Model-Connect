/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/cli.h"

#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
using Json = nlohmann::json;

std::int32_t integer(const Json& values, const char* name, std::int32_t fallback,
                     std::int32_t minimum = std::numeric_limits<std::int32_t>::min()) {
    if (!values.contains(name))
        return fallback;
    const auto value = values.at(name).get<std::int64_t>();
    if (value < minimum || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument(std::string(name) + " is outside the supported integer range");
    return static_cast<std::int32_t>(value);
}

float real(const Json& values, const char* name, float fallback) {
    const auto value = values.value(name, static_cast<double>(fallback));
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::invalid_argument(std::string(name) + " must be a finite float32 value");
    return static_cast<float>(value);
}

trtmc::TextGenerationConfig generation_config(const Json& values) {
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = integer(values, "max_new_tokens", 128, 1);
    config.temperature = real(values, "temperature", 1.0F);
    config.top_k = integer(values, "top_k", 1, 0);
    config.top_p = real(values, "top_p", 1.0F);
    config.min_p = real(values, "min_p", 0.0F);
    config.seed = integer(values, "seed", -1);
    config.repetition_penalty = real(values, "repetition_penalty", 1.0F);
    config.use_chat_template = values.value("use_chat_template", false);
    config.enable_thinking = values.value("enable_thinking", true);
    config.text_generation_mode = values.value("generation_mode", std::string("auto"));
    if (config.temperature < 0.0F || config.top_p < 0.0F || config.top_p > 1.0F ||
        config.min_p < 0.0F || config.min_p > 1.0F || config.repetition_penalty <= 0.0F)
        throw std::invalid_argument("invalid codegen sampling configuration");
    return config;
}

Json execute(const std::string& handler, const Json& values, const char* default_runtime_root) {
    if (handler != "generate")
        throw std::invalid_argument("unknown codegen CLI handler: " + handler);
    const trtmc::BundleReader reader(values.at("bundle").get<std::string>());
    if (reader.info().family != "codegen")
        throw std::invalid_argument("codegen CLI requires a codegen bundle");
    auto config = generation_config(values);
    auto task = trtmc::load_task(
        reader, values.value("runtime_root", std::string(default_runtime_root)), 0,
        values.value("runtime_cache", std::string{}), values.value("cuda_graphs", false));
    auto* generation = dynamic_cast<trtmc::ITextGeneration*>(task.get());
    if (generation == nullptr)
        throw std::invalid_argument("codegen bundle does not implement text generation");
    if (!values.contains("max_new_tokens"))
        config.max_new_tokens = generation->default_max_new_tokens();
    const auto result = generation->generate(values.at("prompt").get<std::string>(), config);
    if (!std::isfinite(result.setup_ms) || !std::isfinite(result.prefill_ms) ||
        !std::isfinite(result.decode_ms))
        throw std::runtime_error("codegen returned non-finite timing values");
    return {{"text", result.text},
            {"token_ids", result.token_ids},
            {"setup_ms", result.setup_ms},
            {"prefill_ms", result.prefill_ms},
            {"decode_ms", result.decode_ms}};
}
} // namespace

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        const auto result =
            execute(handler, Json::parse(values_json), default_runtime_root).dump() + '\n';
        output(context, result.data(), result.size());
        return 0;
    } catch (const std::exception& exception) {
        const std::string message = "Error: " + std::string(exception.what()) + '\n';
        error(context, message.data(), message.size());
        return 1;
    }
}
