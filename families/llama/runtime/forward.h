/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_module.h"

namespace trtmc::llama {

class ForwardPipeline final : public internal::IModel,
                              public internal::ITextPredictionPositionsToTokenScores {
  public:
    ForwardPipeline(std::unique_ptr<ITrtModule> module, int64_t vocab_size,
                    int64_t max_sequence_length, std::string vocabulary_id);
    const char* task() const noexcept override { return kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::VocabularyScoresResult
    run(const internal::TextPredictionPositionsToTokenScoresRequest& request,
        internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> module_;
    int64_t vocab_size_, max_sequence_length_;
    std::string vocabulary_id_;
};

ITask* create_forward(const FamilyContext& context);

} // namespace trtmc::llama
