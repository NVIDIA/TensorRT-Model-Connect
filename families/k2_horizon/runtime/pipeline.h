/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// K2-Horizon-owned plain-completion pipeline. The initial family contract is
// intentionally narrow: one native fixed-KV decoder and host greedy sampling.

#include "families/k2_horizon/runtime/kv_cache.h"
#include "families/k2_horizon/runtime/sampler.h"
#include "families/k2_horizon/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct K2HorizonTextGenConfig {
    int32_t vocab_size{0};
    std::vector<int32_t> eos_token_ids;
    std::string token_id_name{"token_id"};
    std::string logits_output_name{"logits"};
    bool enable_cuda_graph{false};
    bool log_runtime_stats{false};
};

// Validate the family-owned request boundary before any engine execution.
void k2_horizon_validate_generate_config(const TextGenerationConfig& config);
void k2_horizon_validate_generation_inputs(const std::vector<int32_t>& token_ids,
                                           int32_t max_new_tokens, int32_t vocab_size);

class K2HorizonTextGenerationPipeline final : public ITextGeneration {
  public:
    K2HorizonTextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                                    std::unique_ptr<K2HorizonKvCache> cache,
                                    K2HorizonTextGenConfig config,
                                    std::shared_ptr<ITokenizer> tokenizer = nullptr);

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const TextGenerationConfig& cfg);

  private:
    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
    };

    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids, int32_t max_new_tokens,
                                     const K2HorizonSamplingParams& params,
                                     const TextGenerationConfig& cfg);
    void reset_generation_context();
    void run_step(int32_t token_id, std::vector<float>& logits);
    int32_t run_decode_loop(K2HorizonISampler& sampler, const K2HorizonSamplingParams& params,
                            std::vector<int32_t>& output, std::vector<float>& logits,
                            int32_t max_new_tokens);
    void log_decode_summary(int32_t steps, double milliseconds) const;

    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<K2HorizonKvCache> cache_;
    K2HorizonTextGenConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    double last_setup_ms_{0.0};
};

} // namespace trtmc
