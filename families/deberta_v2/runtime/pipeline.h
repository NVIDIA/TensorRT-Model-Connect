/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/deberta_v2/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct DebertaV2RuntimeConfig {
    std::int32_t max_sequence_length{512};
    std::int32_t hidden_size{768};
    std::int32_t pad_token_id{0};
    std::int32_t cls_token_id{1};
    std::int32_t sep_token_id{2};
};

// One encoder, one task. The engine runs a fixed-length window; the rows that
// come back here are the real ones only, which the Task contract allows and
// which keeps the padded rows - whose values are masked out of every real
// row's attention but are not themselves meaningful - out of the result.
class DebertaV2Pipeline final : public internal::IModel, public internal::ITextToTokenFeatures {
  public:
    DebertaV2Pipeline(std::unique_ptr<ITrtModule> encoder, std::shared_ptr<ITokenizer> tokenizer,
                      DebertaV2RuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::TokenFeaturesResult run(const internal::TextToTokenFeaturesRequest& request,
                                      internal::ConfigView config) override;

  private:
    std::vector<std::int32_t> token_ids(const internal::TextSource& source) const;

    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<ITokenizer> tokenizer_;
    DebertaV2RuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
