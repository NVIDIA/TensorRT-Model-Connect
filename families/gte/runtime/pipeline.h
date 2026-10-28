/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/gte/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct GteRuntimeConfig {
    std::int32_t max_sequence_length{512};
    std::int32_t hidden_size{768};
    std::int32_t pad_token_id{0};
};

// One encoder, one task. The engine returns the final rows and the pooling
// happens here: 1_Pooling/config.json sets pooling_mode_cls_token, so the
// embedding is row 0 and nothing else is read.
class GtePipeline final : public internal::IModel, public internal::ITextToEmbedding {
  public:
    GtePipeline(std::unique_ptr<ITrtModule> encoder, std::shared_ptr<ITokenizer> tokenizer,
                GteRuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override;

  private:
    std::vector<float> embed(std::string_view text);

    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<ITokenizer> tokenizer_;
    GteRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
