/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/embeddinggemma2/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace trtmc {
EmbeddingGemma2Pipeline::EmbeddingGemma2Pipeline(std::unique_ptr<ITrtModule> encoder,
                                                 std::shared_ptr<ITokenizer> tokenizer,
                                                 int32_t dimension, int32_t vocab_size,
                                                 int32_t capacity,
                                                 std::vector<int32_t> unsupported_ids,
                                                 std::string model_id)
    : encoder_(std::move(encoder)), tokenizer_(std::move(tokenizer)), dimension_(dimension),
      vocab_size_(vocab_size), capacity_(capacity), unsupported_ids_(std::move(unsupported_ids)),
      model_id_(std::move(model_id)) {
    if (!encoder_ || !encoder_->ok() || !tokenizer_ || dimension_ < 1 || vocab_size_ < 1 ||
        capacity_ < 2 || capacity_ > 8192 || model_id_.empty())
        throw std::invalid_argument("EmbeddingGemma2 invalid bundle runtime contract");
    for (const auto& name : {"token_id", "position_id"}) {
        const auto maximum = encoder_->input_profile_shape(name, encoder_->profile_idx(),
                                                           ProfileShapeSelector::kMax);
        if (maximum != std::vector<int64_t>{capacity_})
            throw std::invalid_argument(
                "EmbeddingGemma2 engine profile does not match runtime capacity");
    }
}

EmbeddingResult EmbeddingGemma2Pipeline::embed(const std::string& text) {
    auto ids = tokenizer_->encode(text);
    if (ids.empty() || ids.size() > static_cast<std::size_t>(capacity_))
        throw std::invalid_argument("EmbeddingGemma2 input exceeds engine sequence capacity");
    for (const auto id : ids) {
        if (id < 0 || id >= vocab_size_ ||
            std::find(unsupported_ids_.begin(), unsupported_ids_.end(), id) !=
                unsupported_ids_.end())
            throw std::invalid_argument(
                "EmbeddingGemma2 requires text without modality placeholders");
    }
    std::vector<int32_t> positions(ids.size());
    std::iota(positions.begin(), positions.end(), 0);
    Tensor tokens, position_tensor;
    tokens.data = ids.data();
    tokens.shape = {static_cast<int64_t>(ids.size())};
    tokens.dtype = DType::kInt32;
    position_tensor.data = positions.data();
    position_tensor.shape = tokens.shape;
    position_tensor.dtype = DType::kInt32;
    auto outputs = encoder_->forward({{"token_id", tokens}, {"position_id", position_tensor}});
    const auto found = outputs.find("hidden_states");
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.shape != std::vector<int64_t>{static_cast<int64_t>(ids.size()), dimension_})
        throw std::runtime_error("EmbeddingGemma2 invalid hidden-state output");
    const auto* hidden = static_cast<const float*>(found->second.data);
    std::vector<double> sums(static_cast<std::size_t>(dimension_), 0.0);
    for (std::size_t row = 0; row < ids.size(); ++row)
        for (int32_t column = 0; column < dimension_; ++column) {
            const auto value = hidden[row * dimension_ + column];
            if (!std::isfinite(value))
                throw std::runtime_error("EmbeddingGemma2 non-finite hidden-state output");
            sums[column] += value;
        }
    double norm = 0.0;
    for (auto& sum : sums) {
        sum /= ids.size();
        norm += sum * sum;
    }
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm <= 1e-12)
        throw std::runtime_error("EmbeddingGemma2 invalid pooled L2 norm");
    EmbeddingResult result;
    result.dim = dimension_;
    result.data.reserve(sums.size());
    for (const auto sum : sums)
        result.data.push_back(static_cast<float>(sum / norm));
    return result;
}

internal::SemanticEmbeddingResult
EmbeddingGemma2Pipeline::run(const internal::TextToEmbeddingRequest& request,
                             internal::ConfigView config) {
    internal::validate_config({}, config);
    std::string text(request.text);
    if (request.role == internal::EmbeddingRole::Query)
        text = "task: search result | query: " + text;
    else if (request.role == internal::EmbeddingRole::Document)
        text = "title: none | text: " + text;
    else if (request.role != internal::EmbeddingRole::Default)
        throw std::invalid_argument("EmbeddingGemma2 invalid embedding role");
    auto result = embed(text);
    return {std::move(result.data), model_id_, "mean", "l2"};
}
} // namespace trtmc
