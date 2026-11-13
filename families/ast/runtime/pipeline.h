/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/ast/runtime/fbank.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct AstRuntimeConfig {
    ast::FbankConfig fbank;
    std::vector<std::string> labels;
};

// One engine and one pass. The filter-bank frontend is the family's own work;
// everything after it is a ViT over the patches it produces.
class AstPipeline final : public internal::IModel, public internal::IAudioToClassScores {
  public:
    AstPipeline(std::unique_ptr<ITrtModule> engine, AstRuntimeConfig config);

    const char* task() const noexcept override {
        return internal::IAudioToClassScores::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::LabelScoresResult run(const internal::AudioToClassScoresRequest& request,
                                    internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    AstRuntimeConfig config_;
};

} // namespace trtmc
