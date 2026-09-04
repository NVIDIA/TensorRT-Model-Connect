/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mamba/runtime/plugin_helpers.h"

#include <stdexcept>

namespace trtmc::mamba {

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, std::string_view name) {
    const auto& data = require_section(bundle, name);
    return {data.begin(), data.end()};
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label) {
    auto engine = backend.create_module(plan.data(), plan.size(), {});
    if (engine == nullptr || !engine->ok())
        throw std::runtime_error(std::string("mamba failed to load ") + label);
    engine->set_timing_label(label);
    return engine;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    if (tokenizer == nullptr)
        throw std::runtime_error("mamba BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace trtmc::mamba
