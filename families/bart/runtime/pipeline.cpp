/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bart/runtime/pipeline.h"

#include "families/bart/runtime/decode_runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc {

BartPipeline::BartPipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                           std::unique_ptr<BartInferenceState> state, int32_t hidden_size,
                           int32_t num_decoder_layers, int32_t max_source_length,
                           int32_t decoder_start_token_id, int32_t eos_token_id,
                           int32_t bos_token_id, int32_t pad_token_id, int32_t vocab_size,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                           std::string task, std::string model_id)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder)), state_(std::move(state)),
      hidden_size_(hidden_size), num_decoder_layers_(num_decoder_layers),
      max_source_length_(max_source_length), decoder_start_token_id_(decoder_start_token_id),
      eos_token_id_(eos_token_id), bos_token_id_(bos_token_id), pad_token_id_(pad_token_id),
      vocab_size_(vocab_size), stream_(stream), tokenizer_(std::move(tokenizer)),
      task_(std::move(task)), model_id_(std::move(model_id)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("BartPipeline: invalid encoder");
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("BartPipeline: invalid decoder");
    if (!state_ || !state_->ok())
        throw std::runtime_error("BartPipeline: invalid state");
    if (!tokenizer_)
        throw std::runtime_error("BartPipeline: no tokenizer configured");
    if (vocab_size_ <= 0)
        throw std::runtime_error("BartPipeline: vocabulary size must be positive");

    cross_kv_bytes_ = static_cast<std::size_t>(max_source_length_) *
                      static_cast<std::size_t>(hidden_size_) * sizeof(float);
    cross_k_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_));
    cross_v_ptrs_.resize(static_cast<std::size_t>(num_decoder_layers_));
    for (int32_t i = 0; i < num_decoder_layers_; ++i) {
        const std::size_t layer = static_cast<std::size_t>(i);
        if (cross_k_ptrs_[layer].allocate(cross_kv_bytes_) != cudaSuccess)
            throw std::runtime_error("BartPipeline: unable to allocate cross-attention key buffer");
        if (cross_v_ptrs_[layer].allocate(cross_kv_bytes_) != cudaSuccess)
            throw std::runtime_error(
                "BartPipeline: unable to allocate cross-attention value buffer");
    }
}

std::vector<internal::TaskInstance> BartPipeline::task_bindings() {
    return {internal::bind<internal::IConditionalTextGeneration>(*this, bart_config_fields)};
}

internal::TextResult BartPipeline::run(const internal::ConditionalTextGenerationRequest& request,
                                       internal::ConfigView config) {
    internal::validate_config(bart_config_fields, config);
    int32_t max_tokens = 128;
    if (const auto val =
            internal::config_get<std::int64_t>(config, bart_config_fields, "max_new_tokens")) {
        if (*val <= 0 || *val > std::numeric_limits<int32_t>::max())
            throw internal::ConfigError(
                "BartPipeline: max_new_tokens must be positive and fit in int32");
        max_tokens = static_cast<int32_t>(*val);
    }
    if (const auto val = internal::config_get<double>(config, bart_config_fields, "temperature")) {
        if (*val < 0.0)
            throw internal::ConfigError("BartPipeline: temperature must be non-negative");
    }
    if (const auto val = internal::config_get<std::int64_t>(config, bart_config_fields, "top_k")) {
        if (*val < 0)
            throw internal::ConfigError("BartPipeline: top_k must be non-negative");
    }

    auto [padded, copy_len] = prepare_encoder_input_semantic(request.source);

    run_encoder(padded, copy_len);
    setup_cross_attention();

    auto output_ids = run_decoder(max_tokens);

    internal::TextResult out;
    out.token_ids = std::move(output_ids);
    if (tokenizer_ && !out.token_ids.empty())
        out.text = tokenizer_->decode(out.token_ids);
    return out;
}

TextResult BartPipeline::generate(const std::string& prompt, const TextGenerationConfig& cfg) {
    auto [padded, copy_len] = prepare_encoder_input(prompt);
    if (copy_len == 0)
        return TextResult{"[empty input]", {}};

    run_encoder(padded, copy_len);
    setup_cross_attention();

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
    auto output_ids = run_decoder(max_tokens);

    TextResult out;
    out.token_ids = std::move(output_ids);
    if (tokenizer_ && !out.token_ids.empty())
        out.text = tokenizer_->decode(out.token_ids);
    return out;
}

std::pair<std::vector<int32_t>, int32_t>
BartPipeline::prepare_encoder_input_semantic(const internal::TextSource& source) {
    std::vector<int32_t> ids;
    if (const auto* view = std::get_if<std::string_view>(&source)) {
        if (view->empty())
            throw std::invalid_argument("BartPipeline: empty input");
        if (!tokenizer_)
            throw std::runtime_error("BartPipeline: no tokenizer configured");
        ids = tokenizer_->encode(std::string(*view));
        if (ids.empty())
            throw std::invalid_argument("BartPipeline: empty input");
    } else {
        const auto span = std::get<Span<const std::int32_t>>(source);
        if (span.empty())
            throw std::invalid_argument("BartPipeline: empty input");
        for (const auto id : span) {
            if (id < 0 || id >= vocab_size_)
                throw std::invalid_argument("BartPipeline: token id outside model vocabulary");
        }
        ids.assign(span.begin(), span.end());
    }

    // Add BOS/EOS if not present
    if (bos_token_id_ >= 0 && (ids.empty() || ids.front() != bos_token_id_))
        ids.insert(ids.begin(), bos_token_id_);
    if (eos_token_id_ >= 0 && (ids.empty() || ids.back() != eos_token_id_))
        ids.push_back(eos_token_id_);

    if (static_cast<int32_t>(ids.size()) > max_source_length_)
        throw std::invalid_argument("BartPipeline: input exceeds maximum sequence length");

    int32_t copy_len = static_cast<int32_t>(ids.size());
    std::vector<int32_t> padded(static_cast<std::size_t>(max_source_length_), pad_token_id_);
    std::copy_n(ids.begin(), ids.size(), padded.begin());
    return {std::move(padded), copy_len};
}

std::pair<std::vector<int32_t>, int32_t>
BartPipeline::prepare_encoder_input(const std::string& prompt) {
    std::vector<int32_t> ids;
    if (tokenizer_)
        ids = tokenizer_->encode(prompt);
    if (ids.empty())
        return {{}, 0};

    // Add BOS/EOS if the native tokenizer didn't
    if (bos_token_id_ >= 0 && (ids.empty() || ids.front() != bos_token_id_))
        ids.insert(ids.begin(), bos_token_id_);
    if (eos_token_id_ >= 0 && (ids.empty() || ids.back() != eos_token_id_))
        ids.push_back(eos_token_id_);

    int32_t copy_len = std::min(static_cast<int32_t>(ids.size()), max_source_length_);
    std::vector<int32_t> padded(static_cast<std::size_t>(max_source_length_), pad_token_id_);
    std::copy_n(ids.begin(), copy_len, padded.begin());
    return {std::move(padded), copy_len};
}

void BartPipeline::run_encoder(const std::vector<int32_t>& padded_ids, int32_t actual_len) {
    actual_enc_len_ = actual_len;

    // Build attention mask: 0.0 for valid positions, -1e9 for padding
    encoder_attention_mask_.assign(static_cast<std::size_t>(max_source_length_), -1e9f);
    for (int32_t i = 0; i < actual_len; ++i)
        encoder_attention_mask_[static_cast<std::size_t>(i)] = 0.0f;

    TensorMap inputs;
    Tensor ids_tensor;
    ids_tensor.data = const_cast<int32_t*>(padded_ids.data());
    ids_tensor.shape = {max_source_length_};
    ids_tensor.dtype = DType::kInt32;
    inputs["input_ids"] = ids_tensor;

    // Provide attention mask if the encoder expects it
    Tensor mask_tensor;
    if (encoder_->has_input("attention_mask")) {
        mask_tensor.data = encoder_attention_mask_.data();
        mask_tensor.shape = {max_source_length_};
        mask_tensor.dtype = DType::kFloat32;
        inputs["attention_mask"] = mask_tensor;
    }

    encoder_->forward_async(inputs);
    encoder_->sync();
}

void BartPipeline::setup_cross_attention() {
    void* enc_out = encoder_->device_ptr("encoder_output");
    if (!enc_out)
        throw std::runtime_error("BartPipeline: no encoder_output");
    for (int32_t i = 0; i < num_decoder_layers_; ++i) {
        auto idx = static_cast<std::size_t>(i);
        cudaMemcpy(cross_k_ptrs_[idx].get(), enc_out, cross_kv_bytes_, cudaMemcpyDeviceToDevice);
        cudaMemcpy(cross_v_ptrs_[idx].get(), enc_out, cross_kv_bytes_, cudaMemcpyDeviceToDevice);
    }
    for (int32_t i = 0; i < num_decoder_layers_; ++i) {
        std::string s = "_" + std::to_string(i);
        decoder_->bind_external("cross_k" + s, cross_k_ptrs_[static_cast<std::size_t>(i)].get());
        decoder_->bind_external("cross_v" + s, cross_v_ptrs_[static_cast<std::size_t>(i)].get());
    }
}

std::vector<int32_t> BartPipeline::run_decoder(int32_t max_new_tokens) {
    state_->reset();
    state_->bind_to(*decoder_);
    std::vector<int32_t> output_ids;
    std::vector<float> logits;
    int32_t current_token = decoder_start_token_id_;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        run_decoder_step(current_token, logits);
        int32_t next = bart_select_argmax_token(logits);
        if (next == eos_token_id_)
            break;
        output_ids.push_back(next);
        current_token = next;
    }
    return output_ids;
}

void BartPipeline::run_decoder_step(int32_t token_id, std::vector<float>& logits) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    Tensor cross_mask_tensor;
    if (decoder_->has_input("cross_attention_mask")) {
        cross_mask_tensor.data = encoder_attention_mask_.data();
        cross_mask_tensor.shape = {max_source_length_};
        cross_mask_tensor.dtype = DType::kFloat32;
        inputs["cross_attention_mask"] = cross_mask_tensor;
    }
    state_->prepare_step(inputs);
    TensorMap outputs = decoder_->forward(inputs);
    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("BartPipeline: no logits output");
    auto num = it->second.numel();
    logits.resize(static_cast<std::size_t>(num));
    std::memcpy(logits.data(), it->second.data, num * sizeof(float));
    state_->advance();
}

} // namespace trtmc
