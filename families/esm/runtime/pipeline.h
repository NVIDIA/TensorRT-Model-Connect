/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/esm/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct EsmRuntimeConfig {
    std::int32_t max_sequence_length{1024};
    std::int32_t vocab_size{33};
    std::int32_t pad_token_id{1};
    std::int32_t cls_token_id{0};
    std::int32_t eos_token_id{2};
    std::int32_t mask_token_id{32};
    std::int32_t unk_token_id{3};
    std::string vocabulary_id{"esm"};
};

// One encoder, one task. The engine scores the whole fixed-length window;
// which rows are returned is decided here - the <mask> positions when the
// caller marked any, and every real position otherwise, so that a sequence
// with no mask still yields a usable per-residue distribution.
class EsmPipeline final : public internal::IModel, public internal::IMaskedTextToTokenScores {
  public:
    EsmPipeline(std::unique_ptr<ITrtModule> encoder, std::shared_ptr<IEsmTokenizer> tokenizer,
                EsmRuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::VocabularyScoresResult run(const internal::MaskedTextToTokenScoresRequest& request,
                                         internal::ConfigView config) override;

  private:
    std::vector<std::int32_t> token_ids(const internal::TextSource& source) const;

    std::unique_ptr<ITrtModule> encoder_;
    std::shared_ptr<IEsmTokenizer> tokenizer_;
    EsmRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
