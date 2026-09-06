/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/kv_cache.h"
#include "families/qwen3_omni/runtime/pipeline.h"
#include "families/qwen3_omni/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::qwen3_omni {
namespace {

template <typename T>
T require_value(const nlohmann::json& document, const char* name) {
    if (!document.contains(name))
        throw std::runtime_error(std::string("qwen3_omni runtime.json missing '") + name + "'");
    try {
        return document.at(name).get<T>();
    } catch (const nlohmann::json::exception&) {
        throw std::runtime_error(std::string("qwen3_omni runtime.json has invalid '") + name + "'");
    }
}

Qwen3OmniRuntimeConfig parse_config(const BundleReader& bundle) {
    nlohmann::json document;
    try {
        document = nlohmann::json::parse(require_text_section(bundle, "runtime.json"));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("qwen3_omni invalid runtime.json: " + std::string(error.what()));
    }
    if (!document.is_object())
        throw std::runtime_error("qwen3_omni runtime.json must be an object");

    Qwen3OmniRuntimeConfig config;
    config.precision = require_value<std::string>(document, "precision");
#define QWEN3_OMNI_INT(field) config.field = require_value<std::int32_t>(document, #field)
    QWEN3_OMNI_INT(thinker_num_layers);
    QWEN3_OMNI_INT(thinker_num_key_value_heads);
    QWEN3_OMNI_INT(thinker_head_dim);
    QWEN3_OMNI_INT(thinker_vocab_size);
    QWEN3_OMNI_INT(thinker_max_cache_length);
    QWEN3_OMNI_INT(thinker_eos_token_id);
#undef QWEN3_OMNI_INT

    if (config.precision != "bf16")
        throw std::runtime_error("qwen3_omni runtime requires the qualified bf16 Thinker plan");
    const std::int32_t positive[] = {
        config.thinker_num_layers, config.thinker_num_key_value_heads, config.thinker_head_dim,
        config.thinker_vocab_size, config.thinker_max_cache_length,
    };
    for (const std::int32_t value : positive) {
        if (value <= 0)
            throw std::runtime_error("qwen3_omni Thinker dimensions must be positive");
    }
    if (config.thinker_eos_token_id < 0 ||
        config.thinker_eos_token_id >= config.thinker_vocab_size) {
        throw std::runtime_error("qwen3_omni Thinker EOS token is out of range");
    }
    return config;
}

struct DualModules {
    std::unique_ptr<ITrtModule> prefill;
    std::unique_ptr<ITrtModule> decode;
};

DualModules load_thinker(IBackend& backend, const BundleReader& bundle) {
    const auto plan = require_section(bundle, "thinker.plan");
    auto modules = backend.create_dual_profile_modules(plan.data(), plan.size(), {});
    if (!modules.prefill || !modules.decode || !modules.prefill->ok() || !modules.decode->ok())
        throw std::runtime_error("qwen3_omni failed to load Thinker");
    modules.prefill->set_timing_label("Qwen3-Omni Thinker prefill");
    modules.decode->set_timing_label("Qwen3-Omni Thinker decode");
    return {std::move(modules.prefill), std::move(modules.decode)};
}

} // namespace

ITask* create(const FamilyContext& context) {
    Qwen3OmniRuntimeConfig config = parse_config(context.reader);
    DualModules thinker = load_thinker(context.backend, context.reader);
    auto state = std::make_unique<Qwen3OmniKvCache>(
        config.thinker_num_layers, config.thinker_max_cache_length,
        config.thinker_num_key_value_heads * config.thinker_head_dim, thinker.decode->stream(),
        DType::kBFloat16);
    if (!state->ok())
        throw std::runtime_error("qwen3_omni failed to allocate the Thinker KV cache");

    return new Qwen3OmniTextPipeline(std::move(thinker.prefill), std::move(thinker.decode),
                                     std::move(state), std::move(config),
                                     create_tokenizer(context.reader));
}

} // namespace trtmc::qwen3_omni

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen3_omni does not support --kv-cache-size");
    return trtmc::qwen3_omni::create(context);
}
