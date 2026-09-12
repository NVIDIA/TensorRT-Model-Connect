/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "api_internal.h"

namespace trtmc::api {

static_assert(static_cast<std::uint32_t>(internal::ScoreKind::Logit) == TRTMC_SCORE_LOGIT);
static_assert(static_cast<std::uint32_t>(internal::ScoreKind::Probability) ==
              TRTMC_SCORE_PROBABILITY);
static_assert(static_cast<std::uint32_t>(internal::ScoreKind::Unbounded) == TRTMC_SCORE_UNBOUNDED);

LabelScoresStorage::LabelScoresStorage(internal::LabelScoresResult value)
    : result(std::move(value)) {
    if (!result.labels.empty() && result.labels.size() != result.scores.size())
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "family label and score counts differ"};
    switch (result.kind) {
    case internal::ScoreKind::Logit:
    case internal::ScoreKind::Probability:
    case internal::ScoreKind::Unbounded:
        break;
    default:
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned an unknown score kind"};
    }
    labels.reserve(result.labels.size());
    for (const auto& label : result.labels)
        labels.push_back(borrowed_string(label));
}

void fill_label_scores_view(const LabelScoresStorage& storage,
                            trtmc_label_scores_view_v1* output) noexcept {
    *output = {storage.result.scores.data(),
               storage.result.scores.size(),
               {storage.labels.data(), storage.labels.size()},
               static_cast<std::uint32_t>(storage.result.kind),
               borrowed_string(storage.result.vocabulary_id)};
}

trtmc_status TRTMC_CALL label_scores_result_view(const trtmc_result* result,
                                                 trtmc_label_scores_view_v1* output,
                                                 trtmc_error** error) noexcept {
    if (output)
        *output = {};
    return guarded(error, [&] {
        require(output != nullptr, "label scores output is required");
        fill_label_scores_view(require_result<LabelScoresStorage>(result), output);
    });
}

} // namespace trtmc::api
