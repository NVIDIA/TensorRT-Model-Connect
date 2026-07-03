/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point start, SteadyClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void log_trt_load_timing(const char* label, double load_deserialize_ms, std::size_t plan_bytes) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.load_timing] label=\""
         << (label ? label : "engine") << "\" load_deserialize_ms=" << load_deserialize_ms
         << " plan_bytes=" << plan_bytes;
    std::cerr << line.str() << '\n';
}

} // namespace

LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options) {
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    if (!backend)
        throw std::runtime_error("No backend loaded");

    LoadedModule result;
    const auto t0 = SteadyClock::now();
    result.module = backend->create_module(plan->data(), plan->size(), options);
    const auto t1 = SteadyClock::now();
    log_trt_load_timing(label, elapsed_ms(t0, t1), plan->size());
    if (!result.module || !result.module->ok())
        throw std::runtime_error(std::string("Failed to create ITrtModule for ") + label);
    result.module->set_timing_label(label ? label : "engine");
    return result;
}

bool detect_add_special_tokens(const BundleFile& bundle) {
    auto* config_data = find_section(bundle, "config.json");
    if (config_data) {
        std::string cfg_text(config_data->begin(), config_data->end());
        auto pos = cfg_text.find("\"tokenizer_add_special_tokens\"");
        if (pos != std::string::npos) {
            auto val_pos = cfg_text.find(':', pos);
            auto value_pos = val_pos == std::string::npos
                                 ? std::string::npos
                                 : cfg_text.find_first_not_of(" \t\r\n", val_pos + 1);
            if (value_pos != std::string::npos) {
                if (cfg_text.compare(value_pos, 5, "false") == 0 || cfg_text[value_pos] == '0')
                    return false;
                if (cfg_text.compare(value_pos, 4, "true") == 0 || cfg_text[value_pos] == '1')
                    return true;
            }
        }
    }
    if (bundle.info.tokenizer_add_special_tokens_present)
        return bundle.info.tokenizer_add_special_tokens;
    return true;
}

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle) {
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;

    const bool add_special = detect_add_special_tokens(bundle);
    const char* data = tok_data->data();
    const std::size_t size = tok_data->size();

    try {
        if (auto tok = CreateBpeTokenizer(data, size, add_special))
            return std::shared_ptr<ITokenizer>(std::move(tok));
    } catch (...) {
    }
    try {
        if (auto tok = CreateWordPieceTokenizer(data, size, add_special))
            return std::shared_ptr<ITokenizer>(std::move(tok));
    } catch (...) {
    }
    try {
        if (auto tok = CreateUnigramTokenizer(data, size, add_special))
            return std::shared_ptr<ITokenizer>(std::move(tok));
    } catch (...) {
    }
    return nullptr;
}

} // namespace trtmc
