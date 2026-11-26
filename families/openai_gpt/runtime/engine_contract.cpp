/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/openai_gpt/runtime/engine_contract.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::openai_gpt {
namespace {

std::set<std::string> names(const std::vector<TensorInfo>& tensors) {
    std::set<std::string> result;
    for (const auto& tensor : tensors)
        result.insert(tensor.name);
    return result;
}

void require_rows(const ITrtModule& module, const char* name, std::int64_t max_rows) {
    const auto profile = module.profile_idx();
    if (module.tensor_dtype(name) != DType::kInt32 ||
        module.input_profile_shape(name, profile, ProfileShapeSelector::kMin) !=
            std::vector<int64_t>{1} ||
        module.input_profile_shape(name, profile, ProfileShapeSelector::kMax) !=
            std::vector<int64_t>{max_rows})
        throw std::runtime_error(std::string("OpenAI GPT engine profile mismatch for '") + name +
                                 "'");
}

void require_tensor(const ITrtModule& module, const std::string& name, DType dtype,
                    const std::vector<int64_t>& shape) {
    if (module.tensor_dtype(name) != dtype || module.tensor_shape(name) != shape)
        throw std::runtime_error("OpenAI GPT engine tensor '" + name +
                                 "' does not match runtime.json");
}

} // namespace

void validate_engine(const EngineShape& shape, const ITrtModule& module, std::int64_t max_tokens) {
    // Five control inputs and one logits output, plus a K and a V tensor per layer: the
    // declared layer count must match the engine before any name is built from it.
    const auto layers = static_cast<std::int64_t>(shape.num_layers);
    if (layers <= 0 || static_cast<std::int64_t>(module.input_info().size()) != 5 + 2 * layers ||
        static_cast<std::int64_t>(module.output_info().size()) != 1 + 2 * layers)
        throw std::runtime_error("OpenAI GPT engine does not have runtime.json's layer count");
    std::set<std::string> inputs{"token_id", "position_id", "cache_write_indices",
                                 "key_value_lengths", "key_positions"};
    std::set<std::string> outputs{"logits"};
    for (std::int32_t layer = 0; layer < shape.num_layers; ++layer) {
        const auto suffix = "_" + std::to_string(layer);
        inputs.insert({"cache_k" + suffix, "cache_v" + suffix});
        outputs.insert({"present_k" + suffix, "present_v" + suffix});
    }
    if (names(module.input_info()) != inputs || names(module.output_info()) != outputs)
        throw std::runtime_error("OpenAI GPT engine tensor inventory is not exact");
    if (module.optimization_profile_count() != 2)
        throw std::runtime_error("OpenAI GPT engine must have prefill and decode profiles");
    // Every request shape the runtime can send lies inside the profile: 1 to max_tokens new
    // rows, and 1 to max_cache_length rows read (the valid rows rounded up to 32).
    require_rows(module, "token_id", max_tokens);
    require_rows(module, "position_id", max_tokens);
    require_rows(module, "key_positions", shape.max_cache_length);
    require_tensor(module, "cache_write_indices", DType::kInt32, {1});
    require_tensor(module, "key_value_lengths", DType::kInt32, {1});
    require_tensor(module, "logits", DType::kFloat32, {1, shape.vocab_size});
    // The cache is allocated from runtime.json; it must describe the engine's own cache
    // tensors before any memory is sized from it.
    const std::vector<int64_t> cache{1, shape.num_heads, shape.max_cache_length, shape.head_dim};
    for (std::int32_t layer = 0; layer < shape.num_layers; ++layer) {
        const auto suffix = "_" + std::to_string(layer);
        for (const char* stem : {"cache_k", "cache_v", "present_k", "present_v"})
            require_tensor(module, stem + suffix, DType::kFloat32, cache);
    }
}

} // namespace trtmc::openai_gpt
