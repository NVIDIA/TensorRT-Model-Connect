/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Projected cross-attention K/V: a single-GPU Canary encoder engine outputs
// every decoder layer's cross-attention K and V (cross_k_<i>, cross_v_<i>,
// [samples, heads, positions, head_dim]) once per request, and the decoder
// reads them in every step instead of re-projecting the encoder output.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

// The rank of a decoder's cross_k_<i>/cross_v_<i> inputs a bundle of this
// tensor-parallel size carries: single-GPU decoders read the encoder's
// projected K/V heads (rank 4), tensor-parallel decoders the raw encoder
// states they shard and project themselves (rank 3).
inline int32_t canary_cross_input_rank(int32_t tensor_parallel_size) {
    return tensor_parallel_size == 1 ? 4 : 3;
}

// Whether decoder lane i reads encoder sample i for every lane: the decoder
// can then read the encoder's own K/V outputs with no copy (greedy decoding).
// Beam search maps several lanes to one sample and needs per-lane copies.
inline bool canary_lanes_are_samples(const std::vector<int32_t>& lane_to_sample,
                                     std::size_t sample_count) {
    if (lane_to_sample.empty() || lane_to_sample.size() > sample_count)
        return false;
    for (std::size_t lane = 0; lane < lane_to_sample.size(); ++lane) {
        if (lane_to_sample[lane] != static_cast<int32_t>(lane))
            return false;
    }
    return true;
}

// Bytes of one sample's projected K (or V) of one layer, from the tensor's
// [batch, heads, positions, head_dim] shape (the batch dimension is ignored).
inline std::size_t canary_cross_head_sample_bytes(const std::vector<int64_t>& shape,
                                                  std::size_t element_size) {
    if (shape.size() != 4 || element_size == 0)
        throw std::invalid_argument("Canary projected cross-attention K/V must be rank 4");
    std::size_t bytes = element_size;
    for (std::size_t axis = 1; axis < shape.size(); ++axis) {
        if (shape[axis] <= 0)
            throw std::invalid_argument("Canary projected cross-attention K/V has an empty axis");
        bytes *= static_cast<std::size_t>(shape[axis]);
    }
    return bytes;
}

} // namespace trtmc
