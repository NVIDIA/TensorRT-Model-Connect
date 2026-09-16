/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/scores.h"

namespace trtmc {

class LabelScoresResult {
  public:
    // Used by typed header-only Task wrappers immediately after the C call.
    LabelScoresResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    LabelScoresResult(const LabelScoresResult&) = delete;
    LabelScoresResult& operator=(const LabelScoresResult&) = delete;
    LabelScoresResult(LabelScoresResult&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    LabelScoresResult& operator=(LabelScoresResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }

    Span<const float> scores() const noexcept {
        return {view_.scores, static_cast<std::size_t>(view_.count)};
    }
    std::uint32_t kind() const noexcept { return view_.kind; }
    std::string_view vocabulary_id() const { return detail::string_view(view_.vocabulary_id); }
    std::vector<std::string_view> labels() const {
        std::vector<std::string_view> labels;
        for (std::uint64_t i = 0; i < view_.labels.size; ++i)
            labels.push_back(detail::string_view(view_.labels.data[i]));
        return labels;
    }
    // C views and the convenience views above borrow this result owner.
    trtmc_label_scores_view_v1& wire_view() noexcept { return view_; }

  private:
    detail::ResultOwner owner_;
    trtmc_label_scores_view_v1 view_{};
};

} // namespace trtmc
