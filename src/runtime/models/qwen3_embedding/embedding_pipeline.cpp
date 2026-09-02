/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_embedding/embedding_pipeline.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

void validate_pooling_tensors(const std::vector<float>& hidden_states,
                              const std::vector<int32_t>& attention_mask, int32_t batch_size,
                              int32_t sequence_length, int32_t hidden_size) {
    if (batch_size <= 0 || sequence_length <= 0 || hidden_size <= 0)
        throw std::invalid_argument("Qwen embedding pooling dimensions must be positive");
    const auto rows = static_cast<std::size_t>(batch_size) * sequence_length;
    if (attention_mask.size() != rows)
        throw std::invalid_argument("Qwen embedding attention-mask shape does not match");
    if (hidden_states.size() != rows * static_cast<std::size_t>(hidden_size))
        throw std::invalid_argument("Qwen embedding hidden-state shape does not match");
}

int32_t find_last_valid_token(const std::vector<int32_t>& attention_mask, std::size_t mask_offset,
                              int32_t sequence_length) {
    int32_t last_valid = -1;
    for (int32_t token = 0; token < sequence_length; ++token) {
        if (attention_mask[mask_offset + token] != 0)
            last_valid = token;
    }
    if (last_valid < 0)
        throw std::invalid_argument("Qwen embedding attention-mask row has no valid token");
    return last_valid;
}

void copy_normalized_row(const std::vector<float>& hidden_states, std::size_t source_offset,
                         std::vector<float>& pooled, std::size_t destination_offset,
                         int32_t hidden_size) {
    double squared_norm = 0.0;
    for (int32_t hidden = 0; hidden < hidden_size; ++hidden) {
        const float value = hidden_states[source_offset + hidden];
        pooled[destination_offset + hidden] = value;
        squared_norm += static_cast<double>(value) * value;
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm <= 1.0e-12)
        throw std::runtime_error("Qwen embedding last-token vector has invalid L2 norm");
    for (int32_t hidden = 0; hidden < hidden_size; ++hidden)
        pooled[destination_offset + hidden] =
            static_cast<float>(pooled[destination_offset + hidden] / norm);
}

void append_eos_and_validate(std::vector<int32_t>& input_ids, int32_t eos_token_id) {
    if (input_ids.empty() || input_ids.back() != eos_token_id)
        input_ids.push_back(eos_token_id);
    if (input_ids.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("QwenEmbeddingPipeline: token sequence is too large");
}

void validate_hidden_output(const Tensor& hidden, std::size_t token_count) {
    if (hidden.dtype != DType::kFloat32)
        throw std::runtime_error("QwenEmbeddingPipeline: hidden states must be FP32");
    if (hidden.shape.size() != 2 || hidden.shape[0] != static_cast<int64_t>(token_count))
        throw std::runtime_error("QwenEmbeddingPipeline: hidden-state shape mismatch");
    if (hidden.shape[1] <= 0 || hidden.shape[1] > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("QwenEmbeddingPipeline: hidden size is invalid");
}

} // namespace

std::vector<float> qwen_last_token_pool_and_normalize(const std::vector<float>& hidden_states,
                                                      const std::vector<int32_t>& attention_mask,
                                                      int32_t batch_size, int32_t sequence_length,
                                                      int32_t hidden_size) {
    validate_pooling_tensors(hidden_states, attention_mask, batch_size, sequence_length,
                             hidden_size);
    std::vector<float> pooled(static_cast<std::size_t>(batch_size) * hidden_size);
    for (int32_t batch = 0; batch < batch_size; ++batch) {
        const auto mask_offset = static_cast<std::size_t>(batch) * sequence_length;
        const auto last_valid = find_last_valid_token(attention_mask, mask_offset, sequence_length);
        const auto source_offset =
            (mask_offset + static_cast<std::size_t>(last_valid)) * hidden_size;
        const auto destination_offset = static_cast<std::size_t>(batch) * hidden_size;
        copy_normalized_row(hidden_states, source_offset, pooled, destination_offset, hidden_size);
    }
    return pooled;
}

QwenEmbeddingPipeline::QwenEmbeddingPipeline(std::unique_ptr<TrtModule> encoder,
                                             std::shared_ptr<ITokenizer> tokenizer,
                                             int32_t eos_token_id, std::string model_id)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), eos_token_id_(eos_token_id),
      model_id_(std::move(model_id)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("QwenEmbeddingPipeline: invalid TensorRT module");
    if (!tokenizer_)
        throw std::runtime_error("QwenEmbeddingPipeline: tokenizer is required");
    if (eos_token_id_ < 0)
        throw std::runtime_error("QwenEmbeddingPipeline: EOS token ID is required");
}

EmbeddingResult QwenEmbeddingPipeline::embed(const std::string& text) {
    auto input_ids = tokenizer_->encode(text);
    append_eos_and_validate(input_ids, eos_token_id_);

    std::vector<int32_t> position_ids(input_ids.size());
    std::iota(position_ids.begin(), position_ids.end(), 0);
    Tensor input;
    input.data = input_ids.data();
    input.shape = {static_cast<int64_t>(input_ids.size())};
    input.dtype = DType::kInt32;
    Tensor positions;
    positions.data = position_ids.data();
    positions.shape = input.shape;
    positions.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = input;
    inputs["position_id"] = positions;
    auto outputs = encoder_->forward(inputs);
    const auto found = outputs.find("hidden_states");
    if (found == outputs.end() || !found->second.data)
        throw std::runtime_error("QwenEmbeddingPipeline: engine produced no hidden states");
    const Tensor& hidden = found->second;
    validate_hidden_output(hidden, input_ids.size());

    const auto hidden_size = static_cast<int32_t>(hidden.shape[1]);
    const auto element_count = input_ids.size() * static_cast<std::size_t>(hidden_size);
    std::vector<float> hidden_states(element_count);
    std::memcpy(hidden_states.data(), hidden.data, element_count * sizeof(float));
    std::vector<int32_t> attention_mask(input_ids.size(), 1);

    EmbeddingResult result;
    result.data = qwen_last_token_pool_and_normalize(
        hidden_states, attention_mask, 1, static_cast<int32_t>(input_ids.size()), hidden_size);
    result.dim = hidden_size;
    return result;
}

} // namespace trtmc
