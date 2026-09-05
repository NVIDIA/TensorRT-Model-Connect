/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/gpt2/runtime/plugin_helpers.h"

#include <stdexcept>

namespace trtmc::gpt2 {

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
        throw std::runtime_error(std::string("gpt2 failed to load ") + label);
    engine->set_timing_label(label);
    return engine;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    if (tokenizer == nullptr)
        throw std::runtime_error("gpt2 BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

std::int32_t rank_local_kv_dim(std::int32_t num_key_value_heads, std::int32_t head_dim,
                               std::int32_t tensor_parallel_size) {
    if (num_key_value_heads <= 0 || head_dim <= 0 || tensor_parallel_size <= 0 ||
        num_key_value_heads % tensor_parallel_size != 0) {
        throw std::runtime_error(
            "gpt2 num_key_value_heads must be divisible by tensor_parallel_size");
    }
    return (num_key_value_heads / tensor_parallel_size) * head_dim;
}

} // namespace trtmc::gpt2
