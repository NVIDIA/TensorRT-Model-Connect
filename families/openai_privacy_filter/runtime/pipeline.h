/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace trtmc::openai_privacy_filter {

struct RuntimeConfig {
    std::int64_t vocab_size{0};
    std::int64_t max_sequence_length{0};
    std::int64_t num_labels{0};
};

// One request is one sequence: the request's tokens, unframed, run through the
// engine once, and its raw label logits returned as a [tokens, labels] matrix.
class TokenClassifier final : public internal::IModel, public internal::ITextToHeadScores {
  public:
    using Encoder = std::function<std::vector<std::int32_t>(std::string_view)>;

    TokenClassifier(std::unique_ptr<ITrtModule> engine, Encoder encode, RuntimeConfig config);

    const char* task() const noexcept override { return internal::ITextToHeadScores::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::HeadScoresResult run(const internal::TextToHeadScoresRequest& request,
                                   internal::ConfigView config) override;

  private:
    std::vector<std::int32_t> resolve_ids(const internal::TextSource& source) const;

    std::unique_ptr<ITrtModule> engine_;
    Encoder encode_;
    RuntimeConfig config_;
};

} // namespace trtmc::openai_privacy_filter
