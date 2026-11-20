/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace trtmc {

// The rank of a decoder's cross_k_<i>/cross_v_<i> inputs a bundle of this
// tensor-parallel size carries. A single-GPU decoder reads each layer's
// cross-attention K and V projected and split into heads by the encoder engine
// once per request ([1, heads, positions, head_dim], rank 4). A tensor-parallel
// decoder reads the raw encoder states ([positions, hidden], rank 2), which
// every rank shards and projects itself.
inline int32_t whisper_cross_input_rank(int32_t tensor_parallel_size) {
    return tensor_parallel_size == 1 ? 4 : 2;
}

// Why this encoder and decoder cannot share projected cross-attention K/V, or
// an empty string: the encoder must take the padded-rows input and output
// every decoder layer's cross_k_<i> and cross_v_<i> with the shape and dtype
// the decoder reads.
template <typename Encoder, typename Decoder>
inline std::string
whisper_projected_cross_kv_mismatch(const Encoder& encoder, const Decoder& decoder,
                                    int32_t decoder_layers, const std::string& padded_rows_input) {
    if (!encoder.has_input(padded_rows_input))
        return "encoder has no " + padded_rows_input + " input";
    for (int32_t layer = 0; layer < decoder_layers; ++layer) {
        for (const char* stem : {"cross_k_", "cross_v_"}) {
            const std::string name = stem + std::to_string(layer);
            if (!encoder.has_output(name) || !decoder.has_input(name) ||
                encoder.tensor_shape(name) != decoder.tensor_shape(name) ||
                encoder.tensor_dtype(name) != decoder.tensor_dtype(name))
                return "encoder and decoder disagree on projected " + name;
        }
    }
    return {};
}

// Bind every decoder layer's cross_k_<i> and cross_v_<i> input to the encoder's
// output buffer of the same name: the decoder reads the request's K/V in place.
template <typename Encoder, typename Decoder>
inline void bind_whisper_projected_cross_kv(const Encoder& encoder, Decoder& decoder,
                                            int32_t decoder_layers) {
    for (int32_t layer = 0; layer < decoder_layers; ++layer) {
        for (const char* stem : {"cross_k_", "cross_v_"}) {
            const std::string name = stem + std::to_string(layer);
            decoder.bind_external(name, encoder.device_ptr(name));
        }
    }
}

struct WhisperCrossKvPlan {
    std::size_t buffer_bytes{0};
    std::size_t valid_bytes{0};
    std::size_t pad_bytes{0};
    bool zero_pad_encoder_output{false};
};

inline WhisperCrossKvPlan make_whisper_cross_kv_plan(int32_t encoder_seq_length,
                                                     int32_t hidden_size,
                                                     int32_t actual_encoder_seq_length) {
    WhisperCrossKvPlan plan;
    if (encoder_seq_length <= 0 || hidden_size <= 0) {
        return plan;
    }

    const std::size_t encoder_tokens = static_cast<std::size_t>(encoder_seq_length);
    const std::size_t hidden = static_cast<std::size_t>(hidden_size);
    plan.buffer_bytes = encoder_tokens * hidden * sizeof(float);

    if (actual_encoder_seq_length <= 0 || actual_encoder_seq_length >= encoder_seq_length) {
        plan.valid_bytes = plan.buffer_bytes;
        return plan;
    }

    plan.valid_bytes = static_cast<std::size_t>(actual_encoder_seq_length) * hidden * sizeof(float);
    plan.pad_bytes = plan.buffer_bytes - plan.valid_bytes;
    plan.zero_pad_encoder_output = plan.pad_bytes > 0;
    return plan;
}

} // namespace trtmc
