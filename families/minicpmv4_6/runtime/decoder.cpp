/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/decoder.h"

#include "families/minicpmv4_6/runtime/engine_io.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace trtmc::minicpmv4_6 {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

// Argument order is unspecified: never dereference a module that failed to load.
cudaStream_t stream_of(const ITrtModule* module) {
    return module ? module->stream() : nullptr;
}

// Validate both engines before the decoder state allocates its buffers.
const DecoderGeometry& checked_engines(const DecoderGeometry& geometry, const ITrtModule* decoder,
                                       const ITrtModule* prefill, std::int32_t vocab_size,
                                       std::int32_t hidden_size) {
    if (decoder == nullptr || !decoder->ok() || prefill == nullptr || !prefill->ok())
        throw std::runtime_error("MiniCPM-V 4.6 decoder engines failed to load");
    if (prefill->stream() != decoder->stream())
        throw std::runtime_error("MiniCPM-V 4.6 prefill and decoder engines must share a stream");
    validate_decoder_engine(*decoder, geometry, false, vocab_size, hidden_size);
    validate_decoder_engine(*prefill, geometry, true, vocab_size, hidden_size);
    prefill_rows(*prefill, geometry, hidden_size);
    return geometry;
}

} // namespace

Decoder::Decoder(std::unique_ptr<ITrtModule> decoder, std::unique_ptr<ITrtModule> prefill,
                 const DecoderGeometry& geometry, std::int32_t vocab_size, std::int32_t hidden_size)
    : decoder_(std::move(decoder)), prefill_(std::move(prefill)),
      state_(checked_engines(geometry, decoder_.get(), prefill_.get(), vocab_size, hidden_size),
             stream_of(decoder_.get())),
      vocab_size_(vocab_size), hidden_size_(hidden_size),
      prefill_rows_(prefill_rows(*prefill_, geometry, hidden_size)), dtype_(geometry.cache_dtype) {}

void Decoder::copy_logits(const ITrtModule& module, std::vector<float>& logits) {
    const void* device = module.device_ptr("logits");
    if (device == nullptr)
        throw std::runtime_error("MiniCPM-V 4.6 engine has no logits output");
    logits.resize(static_cast<std::size_t>(vocab_size_));
    const auto status =
        cudaMemcpy(logits.data(), device, logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("MiniCPM-V 4.6 logits copy failed: ") +
                                 cudaGetErrorString(status));
}

void Decoder::prefill(const PromptRows& prompt, std::vector<float>& logits) {
    const auto row_bytes = static_cast<std::size_t>(hidden_size_) * dtype_size(dtype_);
    const auto total = static_cast<std::int32_t>(prompt.token_ids.size());
    state_.bind_prefill(*prefill_);
    try {
        for (std::int32_t start = 0; start < total; start += prefill_rows_) {
            const std::int32_t rows = std::min(prefill_rows_, total - start);
            chunk_features_.assign(static_cast<std::size_t>(rows) * row_bytes, 0);
            chunk_selector_.assign(static_cast<std::size_t>(rows), 0);
            for (std::int32_t i = 0; i < rows; ++i) {
                const auto feature = prompt.feature_rows[static_cast<std::size_t>(start + i)];
                if (feature < 0)
                    continue;
                chunk_selector_[static_cast<std::size_t>(i)] = 1;
                std::memcpy(chunk_features_.data() + static_cast<std::size_t>(i) * row_bytes,
                            prompt.features.data() + static_cast<std::size_t>(feature) * row_bytes,
                            row_bytes);
            }
            TensorMap inputs;
            inputs["token_id"] = Tensor{
                const_cast<std::int32_t*>(prompt.token_ids.data() + start), {rows}, DType::kInt32};
            inputs["visual_embedding"] =
                Tensor{chunk_features_.data(), {rows, hidden_size_}, dtype_};
            inputs["use_visual_embedding"] =
                Tensor{chunk_selector_.data(), {rows, 1}, DType::kInt32};
            state_.prepare(inputs, rows);
            prefill_->forward_async(inputs);
            state_.commit_prefill(*prefill_, rows);
        }
        prefill_->sync();
    } catch (...) {
        // Queued launches may still read request buffers: drain before unwinding.
        try {
            prefill_->sync();
        } catch (...) {
        }
        throw;
    }
    copy_logits(*prefill_, logits);
}

void Decoder::step(std::int32_t token, std::vector<float>& logits) {
    TensorMap inputs;
    inputs["token_id"] = Tensor{&token, {1}, DType::kInt32};
    state_.prepare(inputs, 1);
    decoder_->forward_async(inputs);
    decoder_->sync();
    copy_logits(*decoder_, logits);
    state_.commit_decoder();
}

Generation Decoder::generate(const PromptRows& prompt, std::int32_t max_new_tokens,
                             const std::vector<std::int32_t>& eos_ids, std::int32_t token_limit,
                             Sampler& sampler) {
    if (token_limit <= 0 || token_limit > vocab_size_)
        throw std::invalid_argument("MiniCPM-V 4.6 token limit must be in 1..vocab_size");
    const auto count = prompt.token_ids.size();
    const auto row_bytes = static_cast<std::size_t>(hidden_size_) * dtype_size(dtype_);
    if (prompt.features.size() % row_bytes)
        throw std::invalid_argument("MiniCPM-V 4.6 prompt rows are inconsistent");
    check_prompt_rows(prompt.token_ids, prompt.feature_rows,
                      static_cast<std::int64_t>(prompt.features.size() / row_bytes), token_limit);
    if (max_new_tokens <= 0)
        throw std::invalid_argument("MiniCPM-V 4.6 max_new_tokens must be positive");
    if (static_cast<std::int64_t>(count) + max_new_tokens > state_.capacity())
        throw std::invalid_argument("MiniCPM-V 4.6 prompt of " + std::to_string(count) +
                                    " tokens plus " + std::to_string(max_new_tokens) +
                                    " new tokens exceeds the bundle's " +
                                    std::to_string(state_.capacity()) + "-token capacity");

    Generation result;
    std::vector<float> logits;
    state_.reset();
    auto start = std::chrono::steady_clock::now();
    prefill(prompt, logits);
    result.prefill_ms = elapsed_ms(start);
    start = std::chrono::steady_clock::now();
    state_.bind_decoder(*decoder_);
    try {
        for (std::int32_t index = 0; index < max_new_tokens; ++index) {
            const std::int32_t token = sampler.sample(logits, token_limit);
            result.token_ids.push_back(token);
            if (std::find(eos_ids.begin(), eos_ids.end(), token) != eos_ids.end() ||
                index + 1 == max_new_tokens)
                break;
            step(token, logits);
        }
    } catch (...) {
        try {
            decoder_->sync();
        } catch (...) {
        }
        throw;
    }
    result.decode_ms = elapsed_ms(start);
    return result;
}

} // namespace trtmc::minicpmv4_6
