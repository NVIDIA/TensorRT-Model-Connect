#pragma once

#include "runtime/domains/recurrent/recurrent_step_contracts.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {
namespace recurrent {

inline void initialize_rwkv_outputs(int32_t num_layers, int32_t vocab_size, std::size_t state_elems,
                                    std::vector<float>& logits,
                                    std::vector<std::vector<float>>& present_attn_by_layer,
                                    std::vector<std::vector<float>>& present_ff_by_layer,
                                    std::vector<std::vector<float>>& present_num_by_layer,
                                    std::vector<std::vector<float>>& present_den_by_layer,
                                    std::vector<std::vector<float>>& present_max_by_layer) {
    logits.assign(static_cast<std::size_t>(vocab_size), 0.0F);
    initialize_layer_outputs(num_layers, state_elems, present_attn_by_layer);
    initialize_layer_outputs(num_layers, state_elems, present_ff_by_layer);
    initialize_layer_outputs(num_layers, state_elems, present_num_by_layer);
    initialize_layer_outputs(num_layers, state_elems, present_den_by_layer);
    initialize_layer_outputs(num_layers, state_elems, present_max_by_layer);
}

inline void initialize_mamba_outputs(int32_t num_layers, int32_t vocab_size, std::size_t conv_elems,
                                     std::size_t ssm_elems, std::vector<float>& logits,
                                     std::vector<std::vector<float>>& present_conv_by_layer,
                                     std::vector<std::vector<float>>& present_ssm_by_layer) {
    logits.assign(static_cast<std::size_t>(vocab_size), 0.0F);
    initialize_layer_outputs(num_layers, conv_elems, present_conv_by_layer);
    initialize_layer_outputs(num_layers, ssm_elems, present_ssm_by_layer);
}

} // namespace recurrent
} // namespace trtmc
