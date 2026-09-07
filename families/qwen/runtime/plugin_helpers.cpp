/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/plugin_helpers.h"

#include "families/qwen/runtime/tensor_names.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace trtmc::qwen {

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
        throw std::runtime_error(std::string("qwen failed to load ") + label);
    engine->set_timing_label(label);
    return engine;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    if (tokenizer == nullptr)
        throw std::runtime_error("qwen BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

namespace {

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("qwen KV cache byte accounting overflow");
    return left * right;
}

} // namespace

void validate_kv_row_contract(const ITrtModule& module, bool runtime_sized, std::int32_t num_layers,
                              std::int32_t kv_dim, std::int32_t bundle_max_rows,
                              const char* label) {
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        for (const char* prefix : {"cache_k", "cache_v"}) {
            const std::string name = qwen_layer_tensor_name(prefix, layer);
            if (!module.has_input(name) || module.input_is_dynamic(name) != runtime_sized) {
                throw std::runtime_error(std::string(label) +
                                         " engine does not match the dynamic KV contract");
            }
            if (!runtime_sized)
                continue;
            const auto minimum =
                module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMin);
            const auto maximum =
                module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMax);
            if (module.input_rank(name) != 2 || minimum != std::vector<std::int64_t>{1, kv_dim} ||
                maximum != std::vector<std::int64_t>{bundle_max_rows, kv_dim}) {
                throw std::runtime_error(std::string(label) +
                                         " engine has invalid dynamic KV row dimensions");
            }
        }
    }
    if (runtime_sized &&
        (!module.has_input("attention_mask") || !module.input_is_dynamic("attention_mask") ||
         module.input_rank("attention_mask") != 2)) {
        throw std::runtime_error(std::string(label) +
                                 " engine must expose a dynamic rank-2 attention mask");
    }
}

std::int32_t runtime_cache_rows(std::uint64_t requested_bytes, std::int32_t bundle_max_rows,
                                std::int32_t num_layers, std::int32_t num_key_value_heads,
                                std::int32_t head_dim, DType cache_dtype) {
    if (requested_bytes == 0)
        return bundle_max_rows;
    const std::uint64_t kv_dim = checked_multiply(static_cast<std::uint64_t>(num_key_value_heads),
                                                  static_cast<std::uint64_t>(head_dim));
    const std::uint64_t row_bytes = checked_multiply(
        checked_multiply(checked_multiply(static_cast<std::uint64_t>(num_layers), kv_dim),
                         static_cast<std::uint64_t>(dtype_size(cache_dtype))),
        2);
    const std::uint64_t requested_rows = requested_bytes / row_bytes;
    if (requested_rows == 0)
        throw std::invalid_argument("--kv-cache-size is smaller than one Qwen KV cache row");
    return static_cast<std::int32_t>(
        std::min(requested_rows, static_cast<std::uint64_t>(bundle_max_rows)));
}

} // namespace trtmc::qwen
