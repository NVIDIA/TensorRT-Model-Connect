/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Gpt2ISampler: token selection abstraction for autoregressive generation.
//
// Decouples token selection strategy (greedy, top-k, nucleus, beam search,
// grammar-constrained, on-device argmax) from the generation loop.
// Pipelines call sampler->sample() without knowing the selection strategy.
//
// Gpt2LogitsLocation tells the pipeline whether to D2H transfer logits before
// calling sample(). HOST samplers (current default) require logits on CPU.
// DEVICE samplers (future: TASK-08) read logits directly from GPU memory.

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

/// Sampling parameters -- controls token selection behavior.
struct Gpt2SamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    float repetition_penalty{1.0f};
    int32_t seed{-1}; // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
};

/// Factory options for choosing concrete sampler implementations.
struct Gpt2SamplerFactoryOptions {
    bool prefer_torch_cuda_multinomial{true};
};

/// Where the sampler expects logits to live.
enum class Gpt2LogitsLocation {
    HOST,   // Sampler reads from CPU memory (current default)
    DEVICE, // Sampler reads from GPU memory (for on-device sampling)
};

/// Token selection result.
struct Gpt2SampleResult {
    int32_t token_id{0};
    float logprob{0.0f}; // log-probability of selected token (informational)
    bool is_eos{false};  // true if token_id matches eos_token_id
};

/// gpt2-owned sampler interface.
class Gpt2ISampler {
  public:
    virtual ~Gpt2ISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] on host or device (see logits_location()).
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual Gpt2SampleResult sample(const float* logits, int32_t vocab_size,
                                    const Gpt2SamplingParams& params) = 0;

    /// Where does this sampler expect logits?
    /// HOST: pipeline must D2H logits before calling sample().
    /// DEVICE: pipeline passes device pointer directly (no D2H).
    virtual Gpt2LogitsLocation logits_location() const = 0;

    /// Human-readable name for diagnostics.
    virtual const char* sampler_type() const = 0;

    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build Gpt2SamplingParams from GenerateConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct GenerateConfig; // defined in trtmc/pipeline.h

Gpt2SamplingParams gpt2_sampling_params_from_config(const GenerateConfig& cfg,
                                                    int32_t default_eos = -1);

/// Factory: create sampler from Gpt2SamplingParams.
/// - top_k <= 1 && top_p/min_p disabled && seed == -1 => GreedySampler
/// - otherwise => TorchCudaMultinomialSampler when compiled in and preferred,
///   falling back to TopKSampler
std::unique_ptr<Gpt2ISampler> create_gpt2_sampler(const Gpt2SamplingParams& params);
std::unique_ptr<Gpt2ISampler> create_gpt2_sampler(const Gpt2SamplingParams& params,
                                                  const Gpt2SamplerFactoryOptions& options);

/// Factory: create a GPU-side greedy sampler (on-device argmax).
/// Requires CUDA kernels (TRTMC_HAS_CUDA_KERNELS). Returns nullptr if unavailable.
/// The sampler reads logits directly from GPU memory and copies back only the
/// token ID (4 bytes) instead of the full logit vector (~600KB for 151K vocab).
/// Pass the CUDA stream used by the pipeline for synchronized kernel execution.
std::unique_ptr<Gpt2ISampler> create_gpt2_gpu_greedy_sampler(void* stream);

} // namespace trtmc
