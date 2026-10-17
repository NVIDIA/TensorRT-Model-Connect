/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/nomic_bert/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace trtmc::nomic_bert {

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
                   int32_t length, int32_t vocab_size, std::string embedding_space)
    : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), length_(length),
      vocab_size_(vocab_size), embedding_space_(std::move(embedding_space)) {
    if (!engine_ || !engine_->ok() || !tokenizer_ || length < 2 || length > 2048 ||
        vocab_size < 103)
        throw std::invalid_argument("invalid Nomic runtime configuration");
    if (engine_->tensor_shape("input_ids") != std::vector<int64_t>{length} ||
        engine_->tensor_dtype("input_ids") != DType::kInt32 ||
        engine_->tensor_shape("attention_mask") != std::vector<int64_t>{length} ||
        engine_->tensor_dtype("attention_mask") != DType::kFloat32 ||
        engine_->tensor_shape("embedding") != std::vector<int64_t>{1, 768} ||
        engine_->tensor_dtype("embedding") != DType::kFloat32)
        throw std::invalid_argument("Nomic engine does not match its bundle contract");
}

internal::SemanticEmbeddingResult Pipeline::run(const internal::TextToEmbeddingRequest& request,
                                                internal::ConfigView config) {
    internal::validate_config({}, config);
    if (request.text.size() > static_cast<size_t>(length_) * 1024)
        throw std::invalid_argument("Nomic input exceeds the text byte limit");
    std::string text;
    switch (request.role) {
    case internal::EmbeddingRole::Default:
        break;
    case internal::EmbeddingRole::Query:
        text = "search_query: ";
        break;
    case internal::EmbeddingRole::Document:
        text = "search_document: ";
        break;
    default:
        throw std::invalid_argument("invalid Nomic embedding role");
    }
    if (!request.text.empty())
        text.append(request.text.data(), request.text.size());
    auto ids = tokenizer_->encode(text);
    if (ids.size() < 2 || ids.size() > static_cast<size_t>(length_))
        throw std::invalid_argument("Nomic input exceeds the engine token limit");
    if (std::any_of(ids.begin(), ids.end(),
                    [this](int32_t id) { return id < 0 || id >= vocab_size_; }))
        throw std::invalid_argument("Nomic tokenizer produced an invalid token ID");
    std::vector<float> mask(static_cast<size_t>(length_), 0.0f);
    std::fill_n(mask.begin(), ids.size(), 1.0f);
    ids.resize(static_cast<size_t>(length_), 0);
    Tensor id_tensor{ids.data(), {length_}, DType::kInt32};
    Tensor mask_tensor{mask.data(), {length_}, DType::kFloat32};
    auto outputs = engine_->forward({{"input_ids", id_tensor}, {"attention_mask", mask_tensor}});
    const auto found = outputs.find("embedding");
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.shape != std::vector<int64_t>{1, 768} || !found->second.data)
        throw std::runtime_error("invalid Nomic embedding output");
    internal::SemanticEmbeddingResult result;
    result.values.resize(768);
    std::memcpy(result.values.data(), found->second.data, 768 * sizeof(float));
    result.embedding_space = embedding_space_;
    result.pooling = "mean";
    result.normalization = "l2";
    return result;
}

} // namespace trtmc::nomic_bert
