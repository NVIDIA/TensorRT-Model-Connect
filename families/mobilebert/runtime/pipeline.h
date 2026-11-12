/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/mobilebert/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct MobileBertRuntimeConfig {
    std::int32_t max_sequence_length{0};
    std::int32_t vocab_size{0};
    std::int32_t type_vocab_size{0};
    std::int32_t pad_token_id{0};
    std::int32_t unk_token_id{0};
    std::int32_t cls_token_id{0};
    std::int32_t sep_token_id{0};
    std::int32_t mask_token_id{0};
    std::string vocabulary_id;
    // Meaning of the next-sentence head's two scores, in index order.
    std::vector<std::string> relation_labels;
    std::string relation_vocabulary_id;
};

// One fixed-length engine with both pretraining heads of MobileBertForPreTraining.
// Every loaded bundle serves both Tasks; the bundle's primary task only names the
// one it was built for.
//
// masked_text_to_token_scores frames the text as [CLS] text [SEP] (caller token ids
// are taken as given) and returns the masked-LM logits of the [MASK] positions, or
// of every position when the input has no [MASK].
//
// text_pair_to_pretraining_relation_scores frames [CLS] first [SEP] second [SEP]
// with segment 0 up to the first [SEP] and 1 after it, and returns the two
// next-sentence logits.
//
// An input longer than the engine window is refused rather than truncated: cutting
// it would drop masked positions or the end of the second text without telling
// the caller.
class MobileBertPipeline final : public internal::IModel,
                                 public internal::IMaskedTextToTokenScores,
                                 public internal::ITextPairToPretrainingRelationScores {
  public:
    MobileBertPipeline(std::unique_ptr<ITrtModule> engine,
                       std::shared_ptr<IMobileBertTokenizer> tokenizer,
                       MobileBertRuntimeConfig config, std::string primary_task);

    const char* task() const noexcept override { return primary_task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::VocabularyScoresResult run(const internal::MaskedTextToTokenScoresRequest& request,
                                         internal::ConfigView config) override;
    internal::LabelScoresResult
    run(const internal::TextPairToPretrainingRelationScoresRequest& request,
        internal::ConfigView config) override;

  private:
    struct Scores {
        std::vector<float> token_logits; // [requested row, vocab]
        std::array<float, 2> relation{};
    };

    std::vector<std::int32_t> single_ids(const internal::TextSource& source) const;
    // Runs the window; copies the masked-LM logits of `rows` only.
    Scores forward(const std::vector<std::int32_t>& ids, const std::vector<std::int32_t>& segments,
                   const std::vector<std::size_t>& rows);

    std::unique_ptr<ITrtModule> engine_;
    std::shared_ptr<IMobileBertTokenizer> tokenizer_;
    MobileBertRuntimeConfig config_;
    std::string primary_task_;
};

} // namespace trtmc
