/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Family-owned host samplers for autoregressive generation.

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace trtmc {

/// Sampling parameters -- controls token selection behavior.
struct NemotronBoltSamplingParams {
    float temperature{1.0f};
    int32_t top_k{1};  // 1 = greedy unless top_p is active; <=0 = no top-k limit
    float top_p{1.0f}; // 1.0 = disabled; 0.0 = greedy; (0,1) = nucleus
    float min_p{0.0f}; // 0.0 = disabled; filters tokens below min_p * max_prob
    int32_t seed{-1};  // -1 = deterministic (argmax)
    int32_t eos_token_id{-1};
};

/// Token selection result.
struct NemotronBoltSampleResult {
    int32_t token_id{0};
    bool is_eos{false}; // true if token_id matches eos_token_id
};

/// nemotron_bolt-owned sampler interface.
class NemotronBoltISampler {
  public:
    virtual ~NemotronBoltISampler() = default;

    /// Select the next token from logits.
    /// logits: float[vocab_size] on host.
    /// vocab_size: number of logit values.
    /// params: sampling parameters for this step.
    virtual NemotronBoltSampleResult sample(const float* logits, int32_t vocab_size,
                                            const NemotronBoltSamplingParams& params) = 0;
    /// Reset sampler state (e.g., RNG state between sequences).
    virtual void reset() {}
};

/// Build NemotronBoltSamplingParams from TextGenerationConfig fields.
/// Forward-declared here; defined in sampler.cpp alongside the factory.
struct TextGenerationConfig; // defined in trtmc/task.h

NemotronBoltSamplingParams
nemotron_bolt_sampling_params_from_config(const TextGenerationConfig& cfg,
                                          int32_t default_eos = -1);

/// Create the exact host sampler selected by the request.
std::unique_ptr<NemotronBoltISampler>
create_nemotron_bolt_sampler(const NemotronBoltSamplingParams& params);

// Exact conditional eight-candidate proposal probabilities and the selected column.
struct BoltDraftChoice {
    std::array<int32_t, 8> ids{};
    std::array<double, 8> probability{};
    int column{};
};
class BoltVerificationSampler {
  public:
    explicit BoltVerificationSampler(NemotronBoltSamplingParams params);
    BoltDraftChoice propose(const int32_t* ids, const float* scores);
    int32_t target(const std::vector<float>& logits);
    // Accept with min(1,p/q); on rejection sample normalized max(p-q,0).
    std::pair<int32_t, bool> verify(const std::vector<float>& logits,
                                    const BoltDraftChoice& choice);

  private:
    double uniform();
    std::vector<double> distribution(const std::vector<float>& logits) const;
    int32_t draw(const std::vector<double>& weights);
    NemotronBoltSamplingParams params_;
    uint64_t rng_;
};

} // namespace trtmc
