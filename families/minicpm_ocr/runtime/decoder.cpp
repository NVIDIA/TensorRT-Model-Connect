/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpm_ocr/runtime/decoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

namespace trtmc::minicpm_ocr {
namespace {

// Additive score of a hidden cache row; far below any attention logit.
constexpr float kMaskedScore = -1.0e4F;

std::string layer_name(const char* stem, int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("MiniCPM-OCR decoder: ") + what + ": " +
                                 cudaGetErrorString(status));
}

void expect_tensor(const ITrtModule& engine, const std::string& name,
                   const std::vector<int64_t>& shape, DType dtype) {
    if ((!engine.has_input(name) && !engine.has_output(name)) ||
        engine.tensor_shape(name) != shape || engine.tensor_dtype(name) != dtype)
        throw std::runtime_error("MiniCPM-OCR decoder engine disagrees with runtime.json at " +
                                 name);
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

Decoder::Decoder(std::unique_ptr<ITrtModule> engine, DecoderGeometry geometry,
                 std::vector<int32_t> eos_token_ids)
    : engine_(std::move(engine)), geometry_(geometry), eos_token_ids_(std::move(eos_token_ids)) {
    const auto& g = geometry_;
    if (!engine_ || !engine_->ok() || g.hidden_size <= 0 || g.vocab_size <= 0 ||
        g.max_cache_length <= 0 || g.linear_layers <= 0 || g.attention_layers <= 0 ||
        g.conv_dim <= 0 || g.conv_kernel <= 0 || g.linear_heads <= 0 || g.linear_key_dim <= 0 ||
        g.linear_value_dim <= 0 || g.kv_dim <= 0 || eos_token_ids_.empty() ||
        (g.dtype != DType::kFloat16 && g.dtype != DType::kFloat32))
        throw std::runtime_error("Invalid MiniCPM-OCR decoder geometry");
    stream_ = engine_->stream();
    const int64_t cap = g.max_cache_length;
    expect_tensor(*engine_, "token_id", {1}, DType::kInt32);
    expect_tensor(*engine_, "position_id", {1}, DType::kInt32);
    expect_tensor(*engine_, "attention_mask", {1, cap + 1}, DType::kFloat32);
    expect_tensor(*engine_, "visual_embedding", {1, g.hidden_size}, g.dtype);
    expect_tensor(*engine_, "use_visual_embedding", {1, 1}, DType::kInt32);
    expect_tensor(*engine_, "logits", {1, g.vocab_size}, DType::kFloat32);
    const std::vector<int64_t> conv{g.conv_dim, g.conv_kernel};
    const std::vector<int64_t> ssm{g.linear_heads, g.linear_key_dim, g.linear_value_dim};
    for (int32_t i = 0; i < g.linear_layers; ++i) {
        for (const auto& [stem, shape, states] : {std::tuple{"conv_state", conv, &conv_state_},
                                                  std::tuple{"present_conv", conv, &conv_present_},
                                                  std::tuple{"ssm_state", ssm, &ssm_state_},
                                                  std::tuple{"present_ssm", ssm, &ssm_present_}}) {
            expect_tensor(*engine_, layer_name(stem, i), shape, DType::kFloat32);
            states->push_back(DeviceTensor::zeros(shape, DType::kFloat32, stream_));
            engine_->bind_external(layer_name(stem, i), states->back().data());
        }
    }
    for (int32_t i = 0; i < g.attention_layers; ++i) {
        for (const auto& [stem, rows, cache] :
             {std::tuple{"cache_k", cap, &cache_k_}, std::tuple{"cache_v", cap, &cache_v_},
              std::tuple{"present_k", int64_t{1}, &present_k_},
              std::tuple{"present_v", int64_t{1}, &present_v_}}) {
            const std::vector<int64_t> shape{rows, g.kv_dim};
            expect_tensor(*engine_, layer_name(stem, i), shape, g.dtype);
            cache->push_back(DeviceTensor::zeros(shape, g.dtype, stream_));
            engine_->bind_external(layer_name(stem, i), cache->back().data());
        }
    }
    for (const auto* states : {&conv_state_, &conv_present_, &ssm_state_, &ssm_present_, &cache_k_,
                               &cache_v_, &present_k_, &present_v_})
        for (const auto& tensor : *states)
            if (!tensor.ok())
                throw std::runtime_error("MiniCPM-OCR decoder state allocation failed");
    mask_.assign(static_cast<std::size_t>(cap) + 1, kMaskedScore);
    logits_.resize(static_cast<std::size_t>(g.vocab_size));
    zero_row_.assign(static_cast<std::size_t>(g.hidden_size) * dtype_size(g.dtype), 0);
}

bool Decoder::is_eos(int32_t token) const {
    return std::find(eos_token_ids_.begin(), eos_token_ids_.end(), token) != eos_token_ids_.end();
}

void Decoder::reset() {
    position_ = 0;
    std::fill(mask_.begin(), mask_.end(), kMaskedScore);
    mask_.back() = 0.0F; // the current token's own key and value
    // The recurrence starts from zero; stale KV rows stay hidden by the mask.
    for (auto* states : {&conv_state_, &ssm_state_})
        for (auto& tensor : *states)
            check_cuda(cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream_), "state reset");
    check_cuda(cudaStreamSynchronize(stream_), "state reset");
}

const std::vector<float>& Decoder::step(int32_t token, const uint8_t* visual_row) {
    const auto& g = geometry_;
    if (position_ >= g.max_cache_length)
        throw std::invalid_argument("MiniCPM-OCR sequence exceeds the bundle's cache capacity");
    if (token < 0 || token >= g.vocab_size)
        throw std::invalid_argument("MiniCPM-OCR token id is outside the vocabulary");
    int32_t position = position_;
    int32_t use_visual = visual_row != nullptr ? 1 : 0;
    TensorMap inputs;
    inputs["token_id"] = Tensor{&token, {1}, DType::kInt32};
    inputs["position_id"] = Tensor{&position, {1}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{mask_.data(), {1, static_cast<int64_t>(mask_.size())}, DType::kFloat32};
    inputs["visual_embedding"] =
        Tensor{const_cast<uint8_t*>(visual_row != nullptr ? visual_row : zero_row_.data()),
               {1, g.hidden_size},
               g.dtype};
    inputs["use_visual_embedding"] = Tensor{&use_visual, {1, 1}, DType::kInt32};
    engine_->forward_async(inputs);
    if (logits_device_ == nullptr) {
        engine_->sync();
        logits_device_ = engine_->device_ptr("logits");
        if (logits_device_ == nullptr)
            throw std::runtime_error("MiniCPM-OCR decoder engine has no logits output");
    }
    check_cuda(cudaMemcpyAsync(logits_.data(), logits_device_, logits_.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_),
               "logits copy");
    // Commit this token: present recurrent state becomes the next input state
    // and the new key/value row is appended at this position.
    for (int32_t i = 0; i < g.linear_layers; ++i)
        if (!conv_state_[i].copy_from(conv_present_[i]) ||
            !ssm_state_[i].copy_from(ssm_present_[i]))
            throw std::runtime_error("MiniCPM-OCR decoder: recurrent state update failed");
    const std::size_t row_bytes = static_cast<std::size_t>(g.kv_dim) * dtype_size(g.dtype);
    for (int32_t i = 0; i < g.attention_layers; ++i) {
        auto* k = static_cast<uint8_t*>(cache_k_[i].data()) + row_bytes * position_;
        auto* v = static_cast<uint8_t*>(cache_v_[i].data()) + row_bytes * position_;
        check_cuda(
            cudaMemcpyAsync(k, present_k_[i].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_),
            "cache append");
        check_cuda(
            cudaMemcpyAsync(v, present_v_[i].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_),
            "cache append");
    }
    check_cuda(cudaStreamSynchronize(stream_), "decoder step");
    mask_[static_cast<std::size_t>(position_)] = 0.0F;
    ++position_;
    return logits_;
}

int32_t Decoder::sample(const std::vector<float>& logits, const SamplingOptions& sampling,
                        uint64_t& rng) const {
    const auto argmax = [&] {
        return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                    logits.begin());
    };
    if (sampling.top_k == 1 || sampling.temperature <= 0.0 || sampling.top_p <= 0.0)
        return argmax();
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto keep_k =
        sampling.top_k > 0
            ? std::min<std::size_t>(static_cast<std::size_t>(sampling.top_k), order.size())
            : order.size();
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep_k),
                      order.end(), [&](int32_t a, int32_t b) {
                          return logits[static_cast<std::size_t>(a)] >
                                 logits[static_cast<std::size_t>(b)];
                      });
    order.resize(keep_k);
    std::vector<double> probabilities(keep_k);
    const double top = logits[static_cast<std::size_t>(order.front())];
    double total = 0.0;
    for (std::size_t i = 0; i < keep_k; ++i) {
        probabilities[i] =
            std::exp((logits[static_cast<std::size_t>(order[i])] - top) / sampling.temperature);
        total += probabilities[i];
    }
    std::size_t keep = keep_k;
    if (sampling.top_p < 1.0) {
        double cumulative = 0.0;
        for (keep = 0; keep < keep_k;) {
            cumulative += probabilities[keep++] / total;
            if (cumulative >= sampling.top_p)
                break;
        }
    }
    std::mt19937_64 engine(rng);
    rng = engine();
    std::discrete_distribution<std::size_t> draw(
        probabilities.begin(), probabilities.begin() + static_cast<std::ptrdiff_t>(keep));
    return order[draw(engine)];
}

DecoderOutput Decoder::generate(const DecoderPrompt& prompt, const SamplingOptions& sampling,
                                int32_t max_new_tokens) {
    const auto& g = geometry_;
    const std::size_t row_bytes = static_cast<std::size_t>(g.hidden_size) * dtype_size(g.dtype);
    const std::size_t rows = prompt.features.size() / row_bytes;
    if (prompt.token_ids.empty() || max_new_tokens <= 0 ||
        prompt.feature_rows.size() != prompt.token_ids.size() ||
        prompt.features.size() % row_bytes != 0)
        throw std::invalid_argument("Invalid MiniCPM-OCR decoder prompt");
    if (prompt.token_ids.size() + static_cast<std::size_t>(max_new_tokens) - 1 >
        static_cast<std::size_t>(g.max_cache_length))
        throw std::invalid_argument(
            "MiniCPM-OCR prompt (" + std::to_string(prompt.token_ids.size()) +
            " tokens) plus max_new_tokens (" + std::to_string(max_new_tokens) +
            ") exceeds the bundle's sequence capacity (" + std::to_string(g.max_cache_length) +
            "); rebuild with a larger max_sequence_length");
    for (const auto row : prompt.feature_rows)
        if (row < -1 || (row >= 0 && static_cast<std::size_t>(row) >= rows))
            throw std::invalid_argument("MiniCPM-OCR prompt names a missing visual row");
    uint64_t rng = sampling.seed >= 0 ? static_cast<uint64_t>(sampling.seed)
                                      : static_cast<uint64_t>(std::random_device{}());
    DecoderOutput output;
    try {
        reset();
        const auto prefill_start = std::chrono::steady_clock::now();
        const std::vector<float>* logits = nullptr;
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i) {
            const auto row = prompt.feature_rows[i];
            logits =
                &step(prompt.token_ids[i],
                      row >= 0 ? prompt.features.data() + row_bytes * static_cast<std::size_t>(row)
                               : nullptr);
        }
        output.prefill_ms = elapsed_ms(prefill_start);
        const auto decode_start = std::chrono::steady_clock::now();
        for (int32_t n = 0; n < max_new_tokens; ++n) {
            const int32_t token = sample(*logits, sampling, rng);
            output.token_ids.push_back(token);
            if (is_eos(token)) {
                output.stopped = true;
                break;
            }
            if (n + 1 < max_new_tokens)
                logits = &step(token, nullptr);
        }
        output.decode_ms = elapsed_ms(decode_start);
    } catch (...) {
        try {
            engine_->sync();
        } catch (...) {
        }
        throw;
    }
    return output;
}

} // namespace trtmc::minicpm_ocr
