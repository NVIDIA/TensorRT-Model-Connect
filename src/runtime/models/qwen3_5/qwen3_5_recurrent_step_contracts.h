#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {
namespace qwen3_5_recurrent {

template <std::size_t N>
bool validate_state_layer_count(const std::array<const std::vector<std::vector<float>>*, N>& states,
                                int32_t expected_layers) {
    for (const auto* state : states) {
        if (static_cast<int32_t>(state->size()) != expected_layers) {
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
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        const auto idx = static_cast<std::size_t>(layer);
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

} // namespace qwen3_5_recurrent
} // namespace trtmc
