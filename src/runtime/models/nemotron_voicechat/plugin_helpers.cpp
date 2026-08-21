/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

using SteadyClock = std::chrono::steady_clock;

void log_trt_load_timing(const char* label, SteadyClock::time_point started,
                         std::size_t plan_bytes) {
    const auto elapsed =
        std::chrono::duration<double, std::milli>(SteadyClock::now() - started).count();
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.load_timing] label=\""
         << (label ? label : "engine") << "\" load_deserialize_ms=" << elapsed
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

    const auto started = SteadyClock::now();
    LoadedModule result;
    result.module = backend->create_module(plan->data(), plan->size(), options);
    log_trt_load_timing(label, started, plan->size());
    if (!result.module || !result.module->ok())
        throw std::runtime_error(std::string("Failed to create ITrtModule for ") + label);
    result.module->set_timing_label(label ? label : "engine");
    return result;
}

std::shared_ptr<ITokenizer> try_create_native_tokenizer(const BundleFile& bundle,
                                                        bool add_special_tokens) {
    const auto* data = find_section(bundle, "tokenizer.json");
    if (!data || data->empty())
        return nullptr;
    try {
        auto tokenizer = CreateBpeTokenizer(data->data(), data->size(), add_special_tokens);
        if (!tokenizer)
            return nullptr;
        return std::shared_ptr<ITokenizer>(std::move(tokenizer));
    } catch (const std::exception&) {
        return nullptr;
    }
}

std::vector<float> section_to_floats(const std::vector<char>* section) {
    if (!section || section->empty())
        return {};
    const auto count = section->size() / sizeof(float);
    std::vector<float> output(count);
    std::memcpy(output.data(), section->data(), count * sizeof(float));
    return output;
}

std::vector<int32_t> section_to_int32s(const std::vector<char>* section) {
    if (!section || section->empty())
        return {};
    const auto count = section->size() / sizeof(int32_t);
    std::vector<int32_t> output(count);
    std::memcpy(output.data(), section->data(), count * sizeof(int32_t));
    return output;
}

MelFilterbank load_mel_filterbank(const BundleFile& bundle) {
    MelFilterbank filterbank;
    const auto* data = find_section(bundle, "mel_filterbank");
    if (!data || data->size() < 2 * sizeof(int32_t))
        return filterbank;

    int32_t header[2] = {0, 0};
    std::memcpy(header, data->data(), sizeof(header));
    filterbank.n_freq_bins = header[0];
    filterbank.n_mel_bins = header[1];
    if (filterbank.n_freq_bins <= 0 || filterbank.n_mel_bins <= 0)
        return {};

    const auto count = static_cast<std::size_t>(filterbank.n_freq_bins) * filterbank.n_mel_bins;
    const auto payload_offset = 2 * sizeof(int32_t);
    if (data->size() < payload_offset + count * sizeof(float))
        return {};

    filterbank.data.resize(count);
    std::memcpy(filterbank.data.data(), data->data() + payload_offset, count * sizeof(float));
    return filterbank;
}

} // namespace trtmc
