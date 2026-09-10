/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/k2_horizon_uno/runtime/algorithm.h"
#include "families/k2_horizon_uno/runtime/kv_cache.h"
#include "families/k2_horizon_uno/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class K2HorizonUnoTextGenerationPipeline final : public ITextGeneration {
  public:
    K2HorizonUnoTextGenerationPipeline(std::unique_ptr<ITrtModule> decoder,
                                       std::unique_ptr<K2HorizonUnoKvCache> cache,
                                       std::shared_ptr<ITokenizer> tokenizer = nullptr);

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg = {}) override;
    int32_t default_max_new_tokens() const override { return 128; }

  private:
    struct DecodeStats {
        K2HorizonUnoGenerationMode mode{K2HorizonUnoGenerationMode::kLinearPsiSpec};
        int32_t block_length{1};
        int32_t forwards{0};
        int32_t committed_tokens{0};
        int32_t lookaheads{0};
    };

    struct TimedGenResult {
        std::vector<int32_t> token_ids;
        double prefill_ms{0.0};
        double decode_ms{0.0};
        DecodeStats stats;
    };

    TimedGenResult generate_from_ids(const std::vector<int32_t>& input_ids,
                                     const TextGenerationConfig& cfg);
    TimedGenResult generate_ar(const std::vector<int32_t>& input_ids,
                               const TextGenerationConfig& cfg,
                               const K2HorizonUnoResolvedGenerateConfig& resolved);
    TimedGenResult generate_linear_psi_spec(const std::vector<int32_t>& input_ids,
                                            const TextGenerationConfig& cfg,
                                            const K2HorizonUnoResolvedGenerateConfig& resolved);

    void reset_generation_context();
    void run_prefill(const std::vector<int32_t>& input_ids, std::vector<float>& logits);
    void run_block(const std::vector<int32_t>& token_ids, const std::vector<float>& lora_mask,
                   std::vector<float>& logits);
    std::vector<int32_t> effective_eos_ids(const TextGenerationConfig& cfg) const;
    void log_decode_receipt(const DecodeStats& stats) const;

    std::unique_ptr<ITrtModule> decoder_;
    std::unique_ptr<K2HorizonUnoKvCache> cache_;
    std::shared_ptr<ITokenizer> tokenizer_;
    double last_setup_ms_{0.0};
};

} // namespace trtmc
