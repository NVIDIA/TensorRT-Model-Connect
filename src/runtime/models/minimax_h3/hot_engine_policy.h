/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

namespace trtmc::minimax_h3 {

inline constexpr std::string_view kDenoiserTransitionPrefix = "denoiser_transition_";
inline constexpr std::string_view kPlanSuffix = "_plan";

inline bool is_fast_h3_transition_plan(std::string_view section) {
    if (section.size() != kDenoiserTransitionPrefix.size() + 2U + kPlanSuffix.size() ||
        section.substr(0, kDenoiserTransitionPrefix.size()) != kDenoiserTransitionPrefix ||
        section.substr(section.size() - kPlanSuffix.size()) != kPlanSuffix) {
        return false;
    }
    const auto index = section.substr(kDenoiserTransitionPrefix.size(), 2);
    if (index[0] < '0' || index[0] > '9' || index[1] < '0' || index[1] > '9')
        return false;
    return (index[0] - '0') * 10 + (index[1] - '0') < 49;
}

inline bool should_retain_hot_engine(std::string_view name, bool retain_engines) {
    if (!retain_engines)
        return false;
    return name == "denoiser_head_plan" || name == "denoiser_tail_plan" ||
           name == "denoiser_finish_plan" || name == "denoiser_entry_plan" ||
           is_fast_h3_transition_plan(name) || name == "vae_tile_decoder_plan" ||
           name == "audio_vae_decoder_plan";
}

} // namespace trtmc::minimax_h3
