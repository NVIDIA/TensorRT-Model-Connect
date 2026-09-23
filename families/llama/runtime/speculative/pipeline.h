/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/llama/runtime/speculative/engine.h"
#include "families/llama/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"

namespace trtmc::llama::speculative {

class Pipeline final : public ITextGeneration {
  public:
    explicit Pipeline(const FamilyContext& context);
    TextResult generate(const std::string& prompt,
                        const TextGenerationConfig& config = {}) override;
    TextResult generate_ids(const std::vector<std::int32_t>& prompt, int count,
                            bool speculative = true, bool ignore_eos = false, int draft_width = 2);
    int32_t default_max_new_tokens() const override { return 128; }
    const std::vector<int>& accepted_lengths() const { return accepted_lengths_; }

  private:
    nlohmann::json manifest_;
    std::unique_ptr<Engine> target_, draft_;
    DeviceTensor prompt_features_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::vector<std::int32_t> mapping_, eos_;
    std::vector<int> accepted_lengths_;
    std::string template_format_;
    int depth_{0};
};

} // namespace trtmc::llama::speculative
