/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/sampler.h"

#include "families/qwen3_omni/runtime/residual_multinomial_kernel.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::qwen3_omni {
namespace {

constexpr std::size_t kTopK = 50;
constexpr float kTopPComplement = 0.2F;
constexpr std::size_t kPredictorVocabulary = 2048;

struct Candidate {
    float logit;
    std::int32_t token;
};

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3-Omni residual sampler failed to ") + action +
                                 ": " + cudaGetErrorString(status));
}

std::vector<float> softmax(const float* logits, const std::vector<bool>& keep) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t token = 0; token < keep.size(); ++token) {
        if (keep[token])
            maximum = std::max(maximum, logits[token]);
    }
    std::vector<float> probabilities(keep.size(), 0.0F);
    float total = 0.0F;
    for (std::size_t token = 0; token < keep.size(); ++token) {
        if (!keep[token])
            continue;
        probabilities[token] = std::exp(logits[token] - maximum);
        total += probabilities[token];
    }
    if (!std::isfinite(total) || total <= 0.0F)
        throw std::runtime_error("Qwen3-Omni residual sampler softmax failed");
    for (float& probability : probabilities)
        probability /= total;
    return probabilities;
}

} // namespace

ResidualCodeSampler::ResidualCodeSampler(std::uint64_t seed) : seed_(seed) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_probabilities_),
                          kPredictorVocabulary * sizeof(float)),
               "allocate probabilities");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_token_), sizeof(std::int32_t)),
                   "allocate output token");
    } catch (...) {
        cudaFree(device_probabilities_);
        device_probabilities_ = nullptr;
        throw;
    }
}

ResidualCodeSampler::~ResidualCodeSampler() {
    cudaFree(device_probabilities_);
    cudaFree(device_token_);
}

std::int32_t ResidualCodeSampler::sample(const float* logits, std::size_t count) {
    if (logits == nullptr)
        throw std::invalid_argument("Qwen3-Omni residual sampler requires logits");
    if (count != kPredictorVocabulary)
        throw std::invalid_argument("Qwen3-Omni residual sampler requires 2048 logits");

    std::vector<float> ordered_logits;
    ordered_logits.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float logit = logits[index];
        if (!std::isfinite(logit))
            throw std::invalid_argument("Qwen3-Omni residual sampler requires finite logits");
        ordered_logits.push_back(logit);
    }

    std::vector<float> top_values = ordered_logits;
    std::nth_element(top_values.begin(), top_values.begin() + (kTopK - 1), top_values.end(),
                     std::greater<float>());
    const float top_k_threshold = top_values[kTopK - 1];
    std::vector<bool> keep(count, false);
    std::vector<Candidate> ascending;
    ascending.reserve(count);
    for (std::size_t token = 0; token < count; ++token) {
        if (logits[token] >= top_k_threshold) {
            keep[token] = true;
            ascending.push_back({logits[token], static_cast<std::int32_t>(token)});
        }
    }
    std::sort(ascending.begin(), ascending.end(),
              [](const Candidate& left, const Candidate& right) {
                  if (left.logit != right.logit)
                      return left.logit < right.logit;
                  return left.token < right.token;
              });
    const std::vector<float> top_k_probabilities = softmax(logits, keep);
    float low_cumulative = 0.0F;
    for (std::size_t index = 0; index + 1 < ascending.size(); ++index) {
        const auto token = static_cast<std::size_t>(ascending[index].token);
        low_cumulative += top_k_probabilities[token];
        if (low_cumulative <= kTopPComplement)
            keep[token] = false;
    }

    const std::vector<float> probabilities = softmax(logits, keep);
    check_cuda(cudaMemcpy(device_probabilities_, probabilities.data(),
                          probabilities.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy probabilities to CUDA");
    launch_residual_exponential_race(device_probabilities_, probabilities.size(), seed_, draws_,
                                     device_token_, nullptr);
    check_cuda(cudaGetLastError(), "launch the CUDA exponential race");
    std::int32_t best_token = -1;
    check_cuda(cudaMemcpy(&best_token, device_token_, sizeof(best_token), cudaMemcpyDeviceToHost),
               "copy the sampled token from CUDA");
    if (best_token < 0 || static_cast<std::size_t>(best_token) >= count ||
        probabilities[static_cast<std::size_t>(best_token)] <= 0.0F) {
        throw std::runtime_error("Qwen3-Omni residual sampler returned an invalid token");
    }
    ++draws_;
    return best_token;
}

} // namespace trtmc::qwen3_omni
