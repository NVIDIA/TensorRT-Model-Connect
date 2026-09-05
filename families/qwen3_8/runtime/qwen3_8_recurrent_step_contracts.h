/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {
namespace qwen3_8_recurrent {

// A null entry, or a negative expected count, is a contract violation rather
// than a pass: these validators run before the recurrent step reads the
// vectors, so they must not dereference what they are asked to check.
template <std::size_t N>
bool validate_state_layer_count(const std::array<const std::vector<std::vector<float>>*, N>& states,
                                int32_t expected_layers) {
    if (expected_layers < 0) {
        return false;
    }
    for (const auto* state : states) {
        if (state == nullptr || static_cast<int32_t>(state->size()) != expected_layers) {
            return false;
        }
    }
    return true;
}

struct StateTensorView {
    const std::vector<std::vector<float>>* values{nullptr};
    std::size_t expected_elems{0};
};

template <std::size_t N>
bool validate_state_tensor_sizes(const std::array<StateTensorView, N>& states, int32_t num_layers) {
    if (num_layers < 0) {
        return false;
    }
    const auto expected = static_cast<std::size_t>(num_layers);
    for (const auto& state : states) {
        if (state.values == nullptr || state.values->size() < expected) {
            return false;
        }
    }
    for (std::size_t idx = 0; idx < expected; ++idx) {
        for (const auto& state : states) {
            if ((*state.values)[idx].size() != state.expected_elems) {
                return false;
            }
        }
    }
    return true;
}

inline void initialize_layer_outputs(int32_t num_layers, std::size_t elems,
                                     std::vector<std::vector<float>>& outputs) {
    outputs.assign(static_cast<std::size_t>(num_layers), std::vector<float>(elems, 0.0F));
}

} // namespace qwen3_8_recurrent
} // namespace trtmc
