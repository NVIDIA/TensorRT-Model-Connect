/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// GemmaISampler implementations: GreedySampler, TopKSampler, and factory.
//
// GreedySampler wraps the same std::max_element logic as the original
// GemmaTextGenerationPipeline::argmax() and select_argmax_token(), producing
// bit-identical token sequences.
//
// TopKSampler handles temperature, top-k, top-p, and min-p sampling on host
// logits, with internal xorshift64 RNG state.

#include "runtime/models/gemma/sampler.h"

#include "trtmc/pipeline.h"
#if TRTMC_HAS_CUDA_KERNELS
#include "runtime/models/gemma/argmax_kernel.h"
#include "runtime/models/gemma/sparse_multinomial_kernel.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace trtmc {

bool gemma_is_eos_token(const GemmaSamplingParams& params, int32_t token_id) {
    if (!params.eos_token_ids.empty()) {
        return std::find(params.eos_token_ids.begin(), params.eos_token_ids.end(), token_id) !=
               params.eos_token_ids.end();
    }
    return params.eos_token_id >= 0 && token_id == params.eos_token_id;
}

// Shared argmax helper — returns {token_id, logprob} for the highest logit.
static GemmaSampleResult argmax_over_logits(const float* logits, int32_t vocab_size,
                                            const GemmaSamplingParams& params) {
    GemmaSampleResult result;
    const float* best = logits;
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > *best)
            best = logits + i;
    }
    result.token_id = static_cast<int32_t>(best - logits);
    result.logprob = *best;
    result.is_eos = gemma_is_eos_token(params, result.token_id);
    return result;
}

struct FilteredDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probs;
    int32_t keep{0};
};

static constexpr float kSamplingEpsilon = 1e-6F;

static float sanitized_temperature(float temperature) {
    if (!std::isfinite(temperature))
        return 1.0F;
    return std::max(temperature, 0.0F);
}

static float sanitized_top_p(float top_p) {
    if (!std::isfinite(top_p))
        return 1.0F;
    return std::min(std::max(top_p, 0.0F), 1.0F);
}

static float sanitized_min_p(float min_p) {
    if (!std::isfinite(min_p))
        return 0.0F;
    return std::min(std::max(min_p, 0.0F), 1.0F);
}

static bool top_p_enabled(float top_p) {
    return top_p > 0.0F && top_p < 1.0F - kSamplingEpsilon;
}

static bool greedy_equivalent(const GemmaSamplingParams& params) {
    const float temperature = sanitized_temperature(params.temperature);
    const float top_p = sanitized_top_p(params.top_p);
    return temperature < kSamplingEpsilon || top_p <= 0.0F;
}

static void topk_indices_and_softmax(FilteredDistribution& dist, const float* logits, int32_t n,
                                     int32_t k, float temperature) {
    dist.indices.resize(static_cast<std::size_t>(n));
    std::iota(dist.indices.begin(), dist.indices.end(), 0);
    std::partial_sort(dist.indices.begin(), dist.indices.begin() + k, dist.indices.end(),
                      [&](int32_t a, int32_t b) {
                          return logits[static_cast<std::size_t>(a)] >
                                 logits[static_cast<std::size_t>(b)];
                      });
    const float max_logit = logits[static_cast<std::size_t>(dist.indices[0])];
    dist.probs.resize(static_cast<std::size_t>(k));
    float sum = 0.0F;
    for (int32_t i = 0; i < k; ++i) {
        const float scaled =
            (logits[static_cast<std::size_t>(dist.indices[static_cast<std::size_t>(i)])] -
             max_logit) /
            temperature;
        dist.probs[static_cast<std::size_t>(i)] = std::isfinite(scaled) ? std::exp(scaled) : 0.0F;
        sum += dist.probs[static_cast<std::size_t>(i)];
    }
    if (sum > 0.0F) {
        for (int32_t i = 0; i < k; ++i)
            dist.probs[static_cast<std::size_t>(i)] /= sum;
    } else {
        const float uniform = 1.0F / static_cast<float>(k);
        for (int32_t i = 0; i < k; ++i)
            dist.probs[static_cast<std::size_t>(i)] = uniform;
    }
}

static int32_t apply_min_p(const FilteredDistribution& dist, int32_t k, float min_p) {
    if (min_p <= 0.0F)
        return k;
    const float max_prob = dist.probs.empty() ? 1.0F : dist.probs[0];
    if (max_prob <= 0.0F)
        return k;
    const float min_prob = min_p * max_prob;
    int32_t keep = 0;
    while (keep < k && dist.probs[static_cast<std::size_t>(keep)] >= min_prob)
        ++keep;
    return std::max(keep, 1);
}

static int32_t apply_top_p(const FilteredDistribution& dist, int32_t keep, float top_p) {
    if (!top_p_enabled(top_p))
        return keep;
    float cumulative = 0.0F;
    int32_t top_p_keep = 0;
    while (top_p_keep < keep) {
        cumulative += dist.probs[static_cast<std::size_t>(top_p_keep)];
        ++top_p_keep;
        if (cumulative >= top_p)
            break;
    }
    return std::max(top_p_keep, 1);
}

static void renormalize_kept_prefix(FilteredDistribution& dist, int32_t keep) {
    float kept_sum = 0.0F;
    for (int32_t i = 0; i < keep; ++i)
        kept_sum += dist.probs[static_cast<std::size_t>(i)];
    if (kept_sum > 0.0F) {
        for (int32_t i = 0; i < keep; ++i)
            dist.probs[static_cast<std::size_t>(i)] /= kept_sum;
    } else {
        const float uniform = 1.0F / static_cast<float>(keep);
        for (int32_t i = 0; i < keep; ++i)
            dist.probs[static_cast<std::size_t>(i)] = uniform;
    }
}

static FilteredDistribution build_filtered_distribution(const float* logits, int32_t vocab_size,
                                                        const GemmaSamplingParams& params) {
    const int32_t n = vocab_size;
    const float temperature = sanitized_temperature(params.temperature);
    const float top_p = sanitized_top_p(params.top_p);
    const float min_p = sanitized_min_p(params.min_p);
    const bool full_vocab_for_top_p = top_p_enabled(top_p) && params.top_k <= 1;
    const int32_t k =
        (params.top_k <= 0 || full_vocab_for_top_p) ? n : std::min(std::max(params.top_k, 1), n);
    FilteredDistribution dist;
    topk_indices_and_softmax(dist, logits, n, k, temperature);
    int32_t keep = apply_min_p(dist, k, min_p);
    keep = apply_top_p(dist, keep, top_p);
    if (keep < k)
        renormalize_kept_prefix(dist, keep);
    dist.keep = keep;
    return dist;
}

// ─────────────────────────────────────────────────────────────
// GreedySampler: deterministic argmax (identical to select_argmax_token)
// ─────────────────────────────────────────────────────────────

class GreedySampler final : public GemmaISampler {
  public:
    GemmaSampleResult sample(const float* logits, int32_t vocab_size,
                             const GemmaSamplingParams& params) override {
        if (vocab_size <= 0 || logits == nullptr) {
            GemmaSampleResult result;
            result.token_id = 0;
            result.is_eos = gemma_is_eos_token(params, 0);
            return result;
        }

        return argmax_over_logits(logits, vocab_size, params);
    }

    GemmaLogitsLocation logits_location() const override { return GemmaLogitsLocation::HOST; }
    const char* sampler_type() const override { return "greedy"; }
};

// ─────────────────────────────────────────────────────────────
// TopKSampler: temperature-scaled top-k with xorshift64 RNG
// (identical logic to sample_token_topk)
// ─────────────────────────────────────────────────────────────

class TopKSampler final : public GemmaISampler {
  public:
    explicit TopKSampler(uint64_t initial_seed)
        : rng_state_(initial_seed == 0 ? 1 : initial_seed),
          initial_seed_(initial_seed == 0 ? 1 : initial_seed) {}

    GemmaSampleResult sample(const float* logits, int32_t vocab_size,
                             const GemmaSamplingParams& params) override {
        GemmaSampleResult result;

        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = gemma_is_eos_token(params, 0);
            return result;
        }

        if (greedy_equivalent(params))
            return argmax_over_logits(logits, vocab_size, params);

        const FilteredDistribution dist = build_filtered_distribution(logits, vocab_size, params);

        // xorshift64 random number generation
        rng_state_ ^= rng_state_ << 13;
        rng_state_ ^= rng_state_ >> 7;
        rng_state_ ^= rng_state_ << 17;
        float u = static_cast<float>(rng_state_ & 0xFFFFFFFF) / 4294967296.0F;

        // Sample from cumulative distribution
        float cumulative = 0.0F;
        for (int32_t i = 0; i < dist.keep; ++i) {
            cumulative += dist.probs[static_cast<std::size_t>(i)];
            if (u < cumulative) {
                result.token_id = dist.indices[static_cast<std::size_t>(i)];
                result.logprob = std::log(std::max(dist.probs[static_cast<std::size_t>(i)],
                                                   std::numeric_limits<float>::min()));
                result.is_eos = gemma_is_eos_token(params, result.token_id);
                return result;
            }
        }

        result.token_id = dist.indices[static_cast<std::size_t>(dist.keep - 1)];
        result.logprob = std::log(std::max(dist.probs[static_cast<std::size_t>(dist.keep - 1)],
                                           std::numeric_limits<float>::min()));
        result.is_eos = gemma_is_eos_token(params, result.token_id);
        return result;
    }

    GemmaLogitsLocation logits_location() const override { return GemmaLogitsLocation::HOST; }
    const char* sampler_type() const override { return "top_k"; }

    void reset() override { rng_state_ = initial_seed_; }

  private:
    uint64_t rng_state_;
    uint64_t initial_seed_;
};

#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
class TorchCudaMultinomialSampler final : public GemmaISampler {
  public:
    explicit TorchCudaMultinomialSampler(uint64_t initial_seed)
        : initial_seed_(initial_seed == 0 ? 1 : initial_seed) {
        cudaMalloc(&d_token_id_, sizeof(int32_t));
    }

    ~TorchCudaMultinomialSampler() override {
        cudaFree(d_indices_);
        cudaFree(d_probs_);
        cudaFree(d_token_id_);
    }

    TorchCudaMultinomialSampler(const TorchCudaMultinomialSampler&) = delete;
    TorchCudaMultinomialSampler& operator=(const TorchCudaMultinomialSampler&) = delete;

    GemmaSampleResult sample(const float* logits, int32_t vocab_size,
                             const GemmaSamplingParams& params) override {
        GemmaSampleResult result;

        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = gemma_is_eos_token(params, 0);
            return result;
        }

        if (greedy_equivalent(params))
            return argmax_over_logits(logits, vocab_size, params);

        const FilteredDistribution dist = build_filtered_distribution(logits, vocab_size, params);
        ensure_execution_policy(vocab_size);
        ensure_device_buffers(dist.keep);

        cudaMemcpyAsync(d_indices_, dist.indices.data(),
                        static_cast<std::size_t>(dist.keep) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_probs_, dist.probs.data(),
                        static_cast<std::size_t>(dist.keep) * sizeof(float), cudaMemcpyHostToDevice,
                        stream_);
        gemma_gpu_sparse_torch_multinomial_exact(d_indices_, d_probs_, dist.keep, initial_seed_,
                                                 current_offset_, total_threads_, d_token_id_,
                                                 stream_);
        cudaMemcpyAsync(&h_token_id_, d_token_id_, sizeof(int32_t), cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);
        current_offset_ += counter_offset_;

        result.token_id = h_token_id_;
        float picked_prob = 0.0F;
        for (int32_t i = 0; i < dist.keep; ++i) {
            if (dist.indices[static_cast<std::size_t>(i)] == result.token_id) {
                picked_prob = dist.probs[static_cast<std::size_t>(i)];
                break;
            }
        }
        result.logprob = std::log(std::max(picked_prob, std::numeric_limits<float>::min()));
        result.is_eos = gemma_is_eos_token(params, result.token_id);
        return result;
    }

    GemmaLogitsLocation logits_location() const override { return GemmaLogitsLocation::HOST; }
    const char* sampler_type() const override { return "torch_multinomial"; }

    void reset() override { current_offset_ = 0; }

  private:
    void ensure_device_buffers(int32_t keep) {
        if (keep > capacity_) {
            cudaFree(d_indices_);
            cudaFree(d_probs_);
            cudaMalloc(&d_indices_, static_cast<std::size_t>(keep) * sizeof(int32_t));
            cudaMalloc(&d_probs_, static_cast<std::size_t>(keep) * sizeof(float));
            capacity_ = keep;
        }
    }

    void ensure_execution_policy(int32_t vocab_size) {
        if (vocab_size != cached_vocab_size_) {
            const GemmaTorchMultinomialExecutionPolicy policy =
                gemma_compute_torch_multinomial_execution_policy(vocab_size);
            cached_vocab_size_ = vocab_size;
            total_threads_ = policy.total_threads;
            counter_offset_ = policy.counter_offset;
        }
    }

    uint64_t initial_seed_;
    uint64_t current_offset_{0};
    cudaStream_t stream_{nullptr};
    int32_t* d_indices_{nullptr};
    float* d_probs_{nullptr};
    int32_t* d_token_id_{nullptr};
    int32_t h_token_id_{0};
    int32_t capacity_{0};
    int32_t cached_vocab_size_{-1};
    int32_t total_threads_{0};
    uint64_t counter_offset_{0};
};
#endif

// ─────────────────────────────────────────────────────────────
// GpuGreedySampler: on-device argmax (no D2H logit transfer)
// ─────────────────────────────────────────────────────────────

#if TRTMC_HAS_CUDA_KERNELS
class GpuGreedySampler final : public GemmaISampler {
  public:
    explicit GpuGreedySampler(cudaStream_t stream) : stream_(stream) {
        cudaMalloc(&d_token_id_, sizeof(int32_t));
        cudaMalloc(&d_logit_val_, sizeof(float));
    }

    ~GpuGreedySampler() override {
        cudaFree(d_token_id_);
        cudaFree(d_logit_val_);
    }

    GpuGreedySampler(const GpuGreedySampler&) = delete;
    GpuGreedySampler& operator=(const GpuGreedySampler&) = delete;

    GemmaSampleResult sample(const float* logits, int32_t vocab_size,
                             const GemmaSamplingParams& params) override {
        GemmaSampleResult result;
        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = gemma_is_eos_token(params, 0);
            return result;
        }

        // logits is a device pointer — run GPU argmax kernel
        gemma_gpu_argmax(logits, vocab_size, d_token_id_, d_logit_val_, stream_);

        // D2H: copy only token_id + logit (8 bytes total vs vocab_size*4 bytes)
        cudaMemcpyAsync(&h_token_id_, d_token_id_, sizeof(int32_t), cudaMemcpyDeviceToHost,
                        stream_);
        cudaMemcpyAsync(&h_logit_val_, d_logit_val_, sizeof(float), cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);

        result.token_id = h_token_id_;
        result.logprob = h_logit_val_;
        result.is_eos = gemma_is_eos_token(params, result.token_id);
        return result;
    }

    GemmaLogitsLocation logits_location() const override { return GemmaLogitsLocation::DEVICE; }
    const char* sampler_type() const override { return "gpu_greedy"; }

  private:
    cudaStream_t stream_{nullptr};
    int32_t* d_token_id_{nullptr};
    float* d_logit_val_{nullptr};
    int32_t h_token_id_{0};
    float h_logit_val_{0.0f};
};
#endif // TRTMC_HAS_CUDA_KERNELS

// ─────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────

GemmaSamplingParams
gemma_sampling_params_from_config(const GenerateConfig& cfg,
                                  const std::vector<int32_t>& default_eos_token_ids) {
    GemmaSamplingParams p;
    p.temperature = cfg.temperature;
    p.top_k = cfg.top_k;
    p.top_p = cfg.top_p;
    p.min_p = cfg.min_p;
    p.seed = cfg.seed;
    p.eos_token_ids =
        (cfg.eos_token_id >= 0) ? std::vector<int32_t>{cfg.eos_token_id} : default_eos_token_ids;
    p.eos_token_id = p.eos_token_ids.empty() ? -1 : p.eos_token_ids.front();
    return p;
}

GemmaSamplingParams gemma_sampling_params_from_config(const GenerateConfig& cfg,
                                                      int32_t default_eos) {
    const std::vector<int32_t> defaults =
        default_eos >= 0 ? std::vector<int32_t>{default_eos} : std::vector<int32_t>{};
    return gemma_sampling_params_from_config(cfg, defaults);
}

std::unique_ptr<GemmaISampler>
create_gemma_sampler(const GemmaSamplingParams& params,
                     [[maybe_unused]] const GemmaSamplerFactoryOptions& options) {
    // Greedy when sampling is fully disabled and no explicit random seed is set.
    const float top_p = sanitized_top_p(params.top_p);
    const float min_p = sanitized_min_p(params.min_p);
    if (params.top_k <= 1 && top_p >= 1.0F - kSamplingEpsilon && min_p <= 0.0F && params.seed < 0) {
        return std::make_unique<GreedySampler>();
    }

    uint64_t seed = (params.seed >= 0) ? static_cast<uint64_t>(params.seed)
                                       : 42ULL; // deterministic default for reproducibility
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (options.prefer_torch_cuda_multinomial)
        return std::make_unique<TorchCudaMultinomialSampler>(seed);
#endif

    // TopK sampler with xorshift64 RNG
    return std::make_unique<TopKSampler>(seed);
}

std::unique_ptr<GemmaISampler> create_gemma_sampler(const GemmaSamplingParams& params) {
    return create_gemma_sampler(params, GemmaSamplerFactoryOptions{});
}

std::unique_ptr<GemmaISampler> create_gemma_gpu_greedy_sampler(void* stream) {
#if TRTMC_HAS_CUDA_KERNELS
    return std::make_unique<GpuGreedySampler>(static_cast<cudaStream_t>(stream));
#else
    (void)stream;
    return nullptr;
#endif
}

} // namespace trtmc
