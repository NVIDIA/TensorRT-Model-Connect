/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/qwen3_omni/runtime/kv_cache.h"
#include "families/qwen3_omni/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Qwen3OmniRuntimeConfig {
    std::string precision;
    std::int32_t thinker_num_layers{0};
    std::int32_t thinker_num_key_value_heads{0};
    std::int32_t thinker_head_dim{0};
    std::int32_t thinker_vocab_size{0};
    std::int32_t thinker_max_cache_length{0};
    std::int32_t thinker_eos_token_id{-1};
};

class Qwen3OmniTextPipeline final : public ITextGeneration {
  public:
    Qwen3OmniTextPipeline(std::unique_ptr<ITrtModule> thinker_prefill,
                          std::unique_ptr<ITrtModule> thinker_decode,
                          std::unique_ptr<Qwen3OmniKvCache> thinker_state,
                          Qwen3OmniRuntimeConfig config, std::shared_ptr<ITokenizer> tokenizer);

    std::int32_t default_max_new_tokens() const override { return 128; }
    TextResult generate(const std::string& prompt,
                        const TextGenerationConfig& config = {}) override;

  private:
    std::vector<float> run_token_prefill(const std::vector<std::int32_t>& token_ids);
    std::vector<float> run_token_step(std::int32_t token_id);
    std::vector<std::int32_t> run_thinker(const std::string& prompt, std::int32_t max_new_tokens);

    std::unique_ptr<ITrtModule> thinker_prefill_;
    std::unique_ptr<ITrtModule> thinker_decode_;
    std::unique_ptr<Qwen3OmniKvCache> thinker_state_;
    Qwen3OmniRuntimeConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
};

} // namespace trtmc
