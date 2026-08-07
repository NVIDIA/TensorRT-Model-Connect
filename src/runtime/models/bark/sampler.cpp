/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/bark/sampler.h"

#if TRTMC_HAS_CUDA_KERNELS
#include "runtime/models/bark/sparse_multinomial_kernel.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace trtmc {

namespace {

struct FilteredDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probabilities;
    int32_t rows{0};
    int32_t vocab_size{0};
    int32_t keep{0};
};

FilteredDistribution build_filtered_distributions(const float* logits, int32_t rows,
                                                  int32_t row_stride, int32_t vocab_size,
                                                  float temperature, int32_t top_k) {
    const int32_t keep = std::max(1, std::min(top_k, vocab_size));
    FilteredDistribution distribution;
    distribution.rows = rows;
    distribution.vocab_size = vocab_size;
    distribution.keep = keep;
    distribution.indices.resize(static_cast<std::size_t>(rows) * keep);
    distribution.probabilities.resize(static_cast<std::size_t>(rows) * keep);
    const float safe_temperature = std::max(temperature, 1e-6F);
    std::vector<int32_t> row_indices(static_cast<std::size_t>(vocab_size));
    for (int32_t row = 0; row < rows; ++row) {
        const float* row_logits = logits + static_cast<std::size_t>(row) * row_stride;
        std::iota(row_indices.begin(), row_indices.end(), 0);
        if (keep < vocab_size) {
            std::partial_sort(row_indices.begin(), row_indices.begin() + keep, row_indices.end(),
                              [row_logits](int32_t lhs, int32_t rhs) {
                                  return row_logits[lhs] > row_logits[rhs];
                              });
        }

        const int32_t max_token = *std::max_element(
            row_indices.begin(), row_indices.begin() + keep,
            [row_logits](int32_t lhs, int32_t rhs) { return row_logits[lhs] < row_logits[rhs]; });
        const float normalizer_logit = row_logits[max_token];
        const std::size_t output_offset = static_cast<std::size_t>(row) * keep;
        float normalizer = 0.0F;
        for (int32_t index = 0; index < keep; ++index) {
            const int32_t token_id = row_indices[static_cast<std::size_t>(index)];
            const float probability =
                std::exp((row_logits[token_id] - normalizer_logit) / safe_temperature);
            distribution.indices[output_offset + index] = token_id;
            distribution.probabilities[output_offset + index] = probability;
            normalizer += probability;
        }
        for (int32_t index = 0; index < keep; ++index) {
            distribution.probabilities[output_offset + index] /= normalizer;
        }
    }
    return distribution;
}

} // namespace

class BarkSampler::Impl {
  public:
    explicit Impl(void* stream)
#if TRTMC_HAS_CUDA_KERNELS
        : stream_(static_cast<cudaStream_t>(stream))
#endif
    {
#if TRTMC_HAS_CUDA_KERNELS
        ensure_device_buffers(1, 1);
#else
        (void)stream;
#endif
    }

    ~Impl() {
#if TRTMC_HAS_CUDA_KERNELS
        cudaFree(device_indices_);
        cudaFree(device_probabilities_);
        cudaFree(device_token_ids_);
#endif
    }

    void reset(int64_t seed) {
        if (seed < 0) {
            return;
        }
#if TRTMC_HAS_CUDA_KERNELS
        seed_ = static_cast<uint64_t>(seed);
        current_offset_ = 0;
#else
        rng_.seed(static_cast<std::mt19937::result_type>(seed));
#endif
    }

    int32_t sample(const float* logits, int32_t vocab_size, float temperature, int32_t top_k) {
        const std::vector<int32_t> tokens =
            sample_rows(logits, 1, vocab_size, vocab_size, temperature, top_k);
        return tokens.empty() ? 0 : tokens.front();
    }

    std::vector<int32_t> sample_rows(const float* logits, int32_t rows, int32_t row_stride,
                                     int32_t vocab_size, float temperature, int32_t top_k) {
        if (logits == nullptr || rows <= 0 || row_stride < vocab_size || vocab_size <= 0) {
            return {};
        }
        const FilteredDistribution distribution =
            build_filtered_distributions(logits, rows, row_stride, vocab_size, temperature, top_k);
        std::vector<int32_t> selected_tokens(static_cast<std::size_t>(rows));

#if TRTMC_HAS_CUDA_KERNELS
        const int32_t numel = rows * vocab_size;
        ensure_execution_policy(numel);
        ensure_device_buffers(static_cast<int32_t>(distribution.indices.size()), rows);
        const std::size_t index_bytes = distribution.indices.size() * sizeof(int32_t);
        const std::size_t probability_bytes = distribution.probabilities.size() * sizeof(float);
        cudaMemcpyAsync(device_indices_, distribution.indices.data(), index_bytes,
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(device_probabilities_, distribution.probabilities.data(), probability_bytes,
                        cudaMemcpyHostToDevice, stream_);
        bark_gpu_sparse_torch_multinomial_exact(
            device_indices_, device_probabilities_, distribution.rows, distribution.vocab_size,
            distribution.keep, seed_, current_offset_, total_threads_, device_token_ids_, stream_);
        cudaMemcpyAsync(selected_tokens.data(), device_token_ids_,
                        selected_tokens.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        current_offset_ += counter_offset_;
        return selected_tokens;
#else
        std::uniform_real_distribution<float> random(0.0F, 1.0F);
        for (int32_t row = 0; row < rows; ++row) {
            const float selected = random(rng_);
            const std::size_t row_offset = static_cast<std::size_t>(row) * distribution.keep;
            float cumulative = 0.0F;
            int32_t token = distribution.indices[row_offset + distribution.keep - 1];
            for (int32_t index = 0; index < distribution.keep; ++index) {
                cumulative += distribution.probabilities[row_offset + index];
                if (selected < cumulative) {
                    token = distribution.indices[row_offset + index];
                    break;
                }
            }
            selected_tokens[static_cast<std::size_t>(row)] = token;
        }
        return selected_tokens;
#endif
    }

    const char* sampler_type() const {
#if TRTMC_HAS_CUDA_KERNELS
        return "torch_cuda_multinomial";
#else
        return "host_multinomial";
#endif
    }

  private:
#if TRTMC_HAS_CUDA_KERNELS
    void ensure_device_buffers(int32_t entries, int32_t rows) {
        if (entries > entry_capacity_) {
            cudaFree(device_indices_);
            cudaFree(device_probabilities_);
            cudaMalloc(&device_indices_, static_cast<std::size_t>(entries) * sizeof(int32_t));
            cudaMalloc(&device_probabilities_, static_cast<std::size_t>(entries) * sizeof(float));
            entry_capacity_ = entries;
        }
        if (rows > row_capacity_) {
            cudaFree(device_token_ids_);
            cudaMalloc(&device_token_ids_, static_cast<std::size_t>(rows) * sizeof(int32_t));
            row_capacity_ = rows;
        }
    }

    void ensure_execution_policy(int32_t numel) {
        if (numel == cached_numel_) {
            return;
        }
        const BarkTorchMultinomialExecutionPolicy policy =
            bark_compute_torch_multinomial_execution_policy(numel);
        cached_numel_ = numel;
        total_threads_ = policy.total_threads;
        counter_offset_ = policy.counter_offset;
    }

    uint64_t seed_{0};
    uint64_t current_offset_{0};
    cudaStream_t stream_{nullptr};
    int32_t* device_indices_{nullptr};
    float* device_probabilities_{nullptr};
    int32_t* device_token_ids_{nullptr};
    int32_t entry_capacity_{0};
    int32_t row_capacity_{0};
    int32_t cached_numel_{-1};
    int32_t total_threads_{0};
    uint64_t counter_offset_{0};
#else
    std::mt19937 rng_{std::random_device{}()};
#endif
};

BarkSampler::BarkSampler(void* stream) : impl_(std::make_unique<Impl>(stream)) {}

BarkSampler::~BarkSampler() = default;

void BarkSampler::reset(int64_t seed) {
    impl_->reset(seed);
}

int32_t BarkSampler::sample(const float* logits, int32_t vocab_size, float temperature,
                            int32_t top_k) {
    return impl_->sample(logits, vocab_size, temperature, top_k);
}

std::vector<int32_t> BarkSampler::sample_rows(const float* logits, int32_t rows, int32_t row_stride,
                                              int32_t vocab_size, float temperature,
                                              int32_t top_k) {
    return impl_->sample_rows(logits, rows, row_stride, vocab_size, temperature, top_k);
}

const char* BarkSampler::sampler_type() const {
    return impl_->sampler_type();
}

} // namespace trtmc
