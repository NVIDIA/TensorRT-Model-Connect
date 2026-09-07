/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

bool parse_add_special_tokens_value(const nlohmann::json& value) {
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_number_integer()) {
        const auto integer = value.get<std::int64_t>();
        if (integer == 0)
            return false;
        if (integer == 1)
            return true;
    }
    throw std::runtime_error("tokenizer_add_special_tokens must be true, false, 0, or 1");
}

bool parse_add_special_tokens_config(const std::vector<char>& config) {
    const auto parsed = nlohmann::json::parse(config.begin(), config.end());
    if (!parsed.is_object())
        throw std::runtime_error("config.json root is not an object");
    const auto value = parsed.find("tokenizer_add_special_tokens");
    return value == parsed.end() ? true : parse_add_special_tokens_value(*value);
}

bool k2_horizon_add_special_tokens(const BundleFile& bundle) {
    if (bundle.info.tokenizer_add_special_tokens_present)
        return bundle.info.tokenizer_add_special_tokens;

    const auto* config = find_section(bundle, "config.json");
    if (config == nullptr || config->empty())
        return true;

    try {
        return parse_add_special_tokens_config(*config);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("Failed to parse K2-Horizon config.json: ") +
                                 error.what());
    }
}

} // namespace

std::unique_ptr<ITrtModule> load_k2_horizon_engine_plan(IBackend* backend, const BundleFile& bundle,
                                                        const ModuleCreateOptions& options) {
    if (backend == nullptr)
        throw std::runtime_error("K2-Horizon backend is not loaded");

    const auto* plan = find_section(bundle, "engine_plan");
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("K2-Horizon bundle is missing required engine_plan");

    const auto start = std::chrono::steady_clock::now();
    auto module = backend->create_module(plan->data(), plan->size(), options);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(6)
           << "[trtmc.load_timing] label=\"k2_horizon.engine_plan\" load_deserialize_ms=" << elapsed
           << " plan_bytes=" << plan->size() << '\n';
    std::cerr << timing.str();

    if (!module || !module->ok())
        throw std::runtime_error("Failed to create K2-Horizon ITrtModule for engine_plan");
    module->set_timing_label("k2_horizon.engine_plan");
    return module;
}

std::shared_ptr<ITokenizer> create_k2_horizon_bpe_tokenizer(const BundleFile& bundle) {
    const auto* tokenizer_json = find_section(bundle, "tokenizer.json");
    if (tokenizer_json == nullptr || tokenizer_json->empty())
        throw std::runtime_error("K2-Horizon bundle is missing required tokenizer.json");

    try {
        auto tokenizer = CreateBpeTokenizer(tokenizer_json->data(), tokenizer_json->size(),
                                            k2_horizon_add_special_tokens(bundle));
        if (!tokenizer)
            throw std::runtime_error("native BPE tokenizer factory returned null");
        std::cerr << "[trtmc] Using K2-Horizon native BPE tokenizer\n";
        return std::shared_ptr<ITokenizer>(std::move(tokenizer));
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("Failed to create K2-Horizon native BPE tokenizer: ") +
                                 error.what());
    }
}

} // namespace trtmc
