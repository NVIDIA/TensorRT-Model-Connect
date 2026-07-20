/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

bool personaplex_select_token(const float* logits, int32_t vocab_size, float temperature,
                              int32_t top_k, int32_t max_token_id, uint64_t* rng_state,
                              int32_t* token_id, cudaStream_t stream);

void personaplex_prepare_depth_embedding(
    float* output, const void* temporal_hidden, DType temporal_hidden_dtype,
    const float* depth_projection, int64_t depth_projection_elements,
    const float* depth_text_embedding, int64_t depth_text_embedding_elements,
    const float* depth_audio_embeddings, int64_t depth_audio_embedding_elements,
    const int32_t* selected_tokens, int32_t codebook, int32_t forced_text_token,
    bool text_token_is_forced, int32_t forced_previous_token, bool previous_token_is_forced,
    int32_t depth_hidden, int32_t temporal_hidden_dim, int32_t depth_text_vocab,
    int32_t audio_vocab_size, int32_t num_depformer_embeddings, cudaStream_t stream);

} // namespace trtmc
