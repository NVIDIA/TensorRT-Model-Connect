/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/eagle_vlm/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc {

namespace {

// Infer the hidden dimension from the last axis of the first output tensor.
int32_t infer_output_hidden_dim(const TrtModule& module) {
    for (const auto& info : module.output_info())
        if (!info.shape.empty())
            return static_cast<int32_t>(info.shape.back());
    return 0;
}

// Mean-pool [seq_len, hidden] over the first actual_len positions,
// then L2-normalize. Returns the pooled vector of size hidden.
std::vector<float> mean_pool_and_normalize(const float* data, int32_t actual_len, int32_t hidden) {
    std::vector<float> pooled(static_cast<std::size_t>(hidden), 0.0f);
    const float inv_len = 1.0f / static_cast<float>(actual_len);
    for (int32_t s = 0; s < actual_len; ++s)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] += data[static_cast<std::size_t>(s) * hidden + h];
    for (int32_t h = 0; h < hidden; ++h)
        pooled[h] *= inv_len;

    float norm = 0.0f;
    for (int32_t h = 0; h < hidden; ++h)
        norm += pooled[h] * pooled[h];
    norm = std::sqrt(norm);
    if (norm > 1e-12f)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] /= norm;

    return pooled;
}

// Check whether the engine's attention_mask input expects int32.
bool engine_mask_is_int32(const TrtModule& module) {
    for (const auto& info : module.input_info())
        if (info.name == "attention_mask")
            return info.dtype == DType::kInt32;
    return false;
}

std::size_t checked_element_count(const std::vector<int64_t>& shape) {
    if (shape.empty())
        throw std::runtime_error("EncoderPipeline: output tensor is missing shape metadata");

    std::size_t count = 1;
    for (const auto dim : shape) {
        if (dim <= 0)
            throw std::runtime_error("EncoderPipeline: output tensor has a non-positive dimension");
        const auto size = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / size)
            throw std::runtime_error("EncoderPipeline: output tensor element count overflow");
        count *= size;
    }
    return count;
}

// The pinned Nemotron reranker processor's fixed " \n \n " separator is
// encoded by the HF fast tokenizer as the single vocabulary token "ĠĊĠĊ".
// TRTMC's BPE tokenizer currently preserves the two pre-tokenized "ĠĊ"
// pieces instead. Canonicalize that merge at this family-owned input boundary
// so the engine sees the checkpoint's exact token sequence.
void canonicalize_reranking_separator_tokens(const ITokenizer& tokenizer,
                                             std::vector<int32_t>& ids) {
    const auto separator_piece = tokenizer.id_for_token("ĠĊ");
    const auto merged_separator = tokenizer.id_for_token("ĠĊĠĊ");
    if (separator_piece < 0 || merged_separator < 0 || separator_piece == merged_separator)
        return;

    std::vector<int32_t> canonical;
    canonical.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i + 1 < ids.size() && ids[i] == separator_piece && ids[i + 1] == separator_piece) {
            canonical.push_back(merged_separator);
            ++i;
        } else {
            canonical.push_back(ids[i]);
        }
    }
    ids = std::move(canonical);
}

void validate_reranking_score_output(const EmbeddingResult& result,
                                     const std::vector<int64_t>& shape, std::size_t token_count) {
    if (result.data.empty())
        throw std::runtime_error("EncoderPipeline: reranking engine produced no score output");

    const auto element_count = checked_element_count(shape);
    if (element_count != result.data.size())
        throw std::runtime_error("EncoderPipeline: reranking score shape does not match its data");
    if (element_count == 1)
        return;
    if (shape.size() < 2 || shape.back() != 1)
        throw std::runtime_error(
            "EncoderPipeline: reranking score output must have one value per position");
    if (token_count > element_count)
        throw std::runtime_error(
            "EncoderPipeline: reranking score output is shorter than the token sequence");
}

float average_prefix(const std::vector<float>& values, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        sum += values[i];
    return static_cast<float>(sum / static_cast<double>(count));
}

float select_reranking_score(const EmbeddingResult& result, const std::vector<int64_t>& shape,
                             std::size_t token_count, const std::string& pooling) {
    validate_reranking_score_output(result, shape, token_count);
    if (result.data.size() == 1)
        return result.data.front();
    if (pooling == "avg")
        return average_prefix(result.data, token_count);
    if (pooling == "last")
        return result.data[token_count - 1];
    throw std::runtime_error("EncoderPipeline: unsupported reranking pooling mode: " + pooling);
}

} // namespace

// ─── EncoderPipeline ───

EncoderPipeline::EncoderPipeline(std::unique_ptr<TrtModule> encoder, std::string mode,
                                 std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                                 std::string reranking_pooling)
    : encoder_(std::move(encoder)), mode_(std::move(mode)), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), reranking_pooling_(std::move(reranking_pooling)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("EncoderPipeline: invalid encoder module");
}

EmbeddingResult EncoderPipeline::embed(const std::string& text) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    auto ids = tokenizer_->encode(text);
    auto raw = encode_ids(ids);

    // For embedding models: the TRT engine returns [max_seq, hidden] hidden
    // states. Mean-pool over actual input positions and L2-normalize.
    if (mode_ != "embedding" || raw.data.empty())
        return raw;

    const auto actual_len = static_cast<int32_t>(ids.size());
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden <= 0 || actual_len <= 0 || raw.dim < actual_len * hidden)
        return raw;

    auto pooled = mean_pool_and_normalize(raw.data.data(), actual_len, hidden);
    raw.data = std::move(pooled);
    raw.dim = hidden;
    return raw;
}

EmbeddingResult EncoderPipeline::encode(const std::string& text) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    auto ids = tokenizer_->encode(text);
    auto raw = encode_ids(ids);

    // Extract CLS token (first hidden_dim values) from the full hidden state
    // matrix [max_seq, hidden]. This matches HF model.encode() behavior for
    // encoder-only models (BERT, RoBERTa, etc.).
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden > 0 && raw.dim > hidden) {
        raw.data.resize(static_cast<std::size_t>(hidden));
        raw.dim = hidden;
    }
    return raw;
}

float EncoderPipeline::rerank(const std::string& query, const std::string& document) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    if (supports_batched_reranking())
        return rerank_batch(query, {document}).front();
    // Match LlamaNemotronVLRerankProcessor.prompt_template_question_passage()
    // from the pinned checkpoint. The whitespace is part of the trained input
    // contract and changes the token sequence.
    std::string combined = "question:" + query + " \n \n passage:" + document;
    auto ids = tokenizer_->encode(combined);
    canonicalize_reranking_separator_tokens(*tokenizer_, ids);
    if (ids.empty())
        throw std::runtime_error("EncoderPipeline: reranking input produced no tokens");

    auto output = encode_ids_with_shape(ids);
    // Eagle's score head emits one scalar per sequence position as
    // [..., sequence, 1]. Apply the pooling contract recorded by the pinned
    // checkpoint to the valid input positions.
    return select_reranking_score(output.result, output.shape, ids.size(), reranking_pooling_);
}

std::vector<float> EncoderPipeline::rerank_batch(const std::string& query,
                                                 const std::vector<std::string>& documents) {
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    if (documents.empty())
        return {};
    if (!supports_batched_reranking()) {
        std::vector<float> scores;
        scores.reserve(documents.size());
        for (const auto& document : documents)
            scores.push_back(rerank(query, document));
        return scores;
    }

    const auto profile_max = reranking_profile_max_shape();
    const auto max_batch = static_cast<std::size_t>(profile_max[0]);
    const auto max_sequence = static_cast<std::size_t>(profile_max[1]);
    std::vector<float> scores;
    scores.reserve(documents.size());

    for (std::size_t first = 0; first < documents.size(); first += max_batch) {
        const auto batch_size = std::min(max_batch, documents.size() - first);
        std::vector<std::vector<int32_t>> token_rows;
        token_rows.reserve(batch_size);
        std::size_t sequence_length = 0;
        for (std::size_t index = 0; index < batch_size; ++index) {
            std::string combined =
                "question:" + query + " \n \n passage:" + documents[first + index];
            auto ids = tokenizer_->encode(combined);
            canonicalize_reranking_separator_tokens(*tokenizer_, ids);
            if (ids.empty())
                throw std::runtime_error("EncoderPipeline: reranking input produced no tokens");
            if (ids.size() > max_sequence)
                throw std::runtime_error(
                    "EncoderPipeline: reranking input exceeds the engine sequence profile");
            sequence_length = std::max(sequence_length, ids.size());
            token_rows.push_back(std::move(ids));
        }

        // Masked token ID zero is never attended to and therefore does not
        // affect valid positions. Right padding matches the reference batch.
        std::vector<int32_t> padded(batch_size * sequence_length, 0);
        std::vector<int32_t> attention_mask(batch_size * sequence_length, 0);
        for (std::size_t row = 0; row < batch_size; ++row) {
            const auto offset = row * sequence_length;
            std::copy(token_rows[row].begin(), token_rows[row].end(), padded.begin() + offset);
            std::fill_n(attention_mask.begin() + offset, token_rows[row].size(), 1);
        }

        auto output = encode_batch_with_shape(padded, attention_mask, batch_size, sequence_length);
        const auto element_count = checked_element_count(output.shape);
        if (output.result.data.size() != element_count || output.shape.size() < 2 ||
            output.shape.front() != static_cast<int64_t>(batch_size) ||
            element_count % batch_size != 0)
            throw std::runtime_error(
                "EncoderPipeline: batched reranking score shape does not match its inputs");
        const auto elements_per_row = element_count / batch_size;
        if (elements_per_row > 1 && output.shape.back() != 1)
            throw std::runtime_error(
                "EncoderPipeline: batched reranking output must have one score per position");

        for (std::size_t row = 0; row < batch_size; ++row) {
            const auto valid_tokens = token_rows[row].size();
            if (elements_per_row == 1) {
                scores.push_back(output.result.data[row]);
            } else {
                if (valid_tokens > elements_per_row)
                    throw std::runtime_error(
                        "EncoderPipeline: batched reranking output is shorter than an input");
                const auto offset = row * elements_per_row;
                if (reranking_pooling_ == "last") {
                    scores.push_back(output.result.data[offset + valid_tokens - 1]);
                } else if (reranking_pooling_ == "avg") {
                    double sum = 0.0;
                    for (std::size_t index = 0; index < valid_tokens; ++index)
                        sum += output.result.data[offset + index];
                    scores.push_back(static_cast<float>(sum / static_cast<double>(valid_tokens)));
                } else {
                    throw std::runtime_error(
                        "EncoderPipeline: unsupported reranking pooling mode: " +
                        reranking_pooling_);
                }
            }
        }
    }
    return scores;
}

EmbeddingResult EncoderPipeline::encode_ids(const std::vector<int32_t>& input_ids) {
    return encode_ids_with_shape(input_ids).result;
}

EncoderPipeline::EncodedOutput
EncoderPipeline::encode_ids_with_shape(const std::vector<int32_t>& input_ids) {
    std::vector<int32_t> attention_mask(input_ids.size(), 1);
    return forward_ids(input_ids, attention_mask, {static_cast<int64_t>(input_ids.size())});
}

EncoderPipeline::EncodedOutput
EncoderPipeline::encode_batch_with_shape(const std::vector<int32_t>& input_ids,
                                         const std::vector<int32_t>& attention_mask,
                                         std::size_t batch_size, std::size_t sequence_length) {
    if (batch_size == 0 || sequence_length == 0 ||
        input_ids.size() != batch_size * sequence_length ||
        attention_mask.size() != input_ids.size())
        throw std::runtime_error("EncoderPipeline: invalid batched reranking input shape");
    return forward_ids(input_ids, attention_mask,
                       {static_cast<int64_t>(batch_size), static_cast<int64_t>(sequence_length)});
}

EncoderPipeline::EncodedOutput
EncoderPipeline::forward_ids(const std::vector<int32_t>& input_ids,
                             const std::vector<int32_t>& attention_mask,
                             const std::vector<int64_t>& shape) {
    std::vector<float> mask_f32;
    if (!engine_mask_is_int32(*encoder_)) {
        mask_f32.reserve(attention_mask.size());
        for (const auto value : attention_mask)
            mask_f32.push_back(static_cast<float>(value));
    }

    auto ids_copy = input_ids;
    Tensor ids_t;
    ids_t.data = ids_copy.data();
    ids_t.shape = shape;
    ids_t.dtype = DType::kInt32;

    // Match the engine's expected dtype for the attention mask.
    Tensor mask_t;
    if (engine_mask_is_int32(*encoder_)) {
        mask_t.data = const_cast<int32_t*>(attention_mask.data());
        mask_t.shape = shape;
        mask_t.dtype = DType::kInt32;
    } else {
        mask_t.data = mask_f32.data();
        mask_t.shape = shape;
        mask_t.dtype = DType::kFloat32;
    }

    TensorMap inputs;
    inputs["input_ids"] = ids_t;
    inputs["attention_mask"] = mask_t;

    auto outputs = encoder_->forward(inputs);

    EncodedOutput output;
    for (auto& [name, tensor] : outputs) {
        if (name.find("logits") != std::string::npos || name.find("embed") != std::string::npos ||
            name.find("output") != std::string::npos || name.find("hidden") != std::string::npos ||
            name.find("score") != std::string::npos) {
            const auto element_count = checked_element_count(tensor.shape);
            if (!tensor.data)
                throw std::runtime_error("EncoderPipeline: output tensor has no data");
            if (element_count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
                throw std::runtime_error("EncoderPipeline: output tensor is too large");
            output.result.data.resize(element_count);
            std::memcpy(output.result.data.data(), tensor.data, element_count * sizeof(float));
            output.result.dim = static_cast<int32_t>(element_count);
            output.shape = tensor.shape;
            break;
        }
    }

    return output;
}

bool EncoderPipeline::supports_batched_reranking() const {
    for (const auto& input : encoder_->input_info())
        if (input.name == "input_ids")
            return input.shape.size() == 2;
    return false;
}

std::vector<int64_t> EncoderPipeline::reranking_profile_max_shape() const {
    const auto shape = encoder_->input_profile_shape("input_ids", encoder_->profile_idx(),
                                                     ProfileShapeSelector::kMax);
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0)
        throw std::runtime_error("EncoderPipeline: invalid batched reranking profile");
    return shape;
}

} // namespace trtmc
