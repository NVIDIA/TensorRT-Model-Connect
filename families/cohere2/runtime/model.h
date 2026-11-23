/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/cohere2/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <mutex>

namespace trtmc::cohere2 {

struct ModelConfig {
    std::int32_t length;
    std::int32_t vocabulary;
    std::int32_t bos;
    std::int32_t eos;
    std::int32_t pad;
};

class Model final : public internal::IModel, public internal::ITextContinuation {
  public:
    Model(std::unique_ptr<ITrtModule> engine, std::unique_ptr<Tokenizer> tokenizer,
          ModelConfig config);
    const char* task() const noexcept override { return internal::ITextContinuation::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    TextResult run(const internal::TextContinuationRequest&, internal::ConfigView) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<Tokenizer> tokenizer_;
    ModelConfig config_;
    std::mutex generation_;
};

} // namespace trtmc::cohere2
