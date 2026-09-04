/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>

namespace trtmc::minimax_h3 {

inline bool should_retain_hot_engine(std::string_view name, bool retain_engines) {
    if (!retain_engines)
        return false;
    return name == "denoiser_head_plan" || name == "denoiser_tail_plan" ||
           name == "denoiser_finish_plan" || name == "vae_tile_decoder_plan" ||
           name == "audio_vae_decoder_plan";
}

inline bool uses_serial_execution_context(std::string_view name) {
    // The original-weight FirstBlockCache head, tail, and finish execute
    // strictly in sequence on one stream. Dynamic profiles otherwise give all
    // three contexts independent max-shape activation allocations, which
    // forces unified-memory paging on the 64 GiB Windows profile even for a
    // five-second request. Reuse the native TRT-RTX user-managed arena so the
    // three engines reserve only their largest requirement, not their sum.
    return name == "denoiser_head_plan" || name == "denoiser_tail_plan" ||
           name == "denoiser_finish_plan";
}

inline std::int64_t staged_plan_weight_streaming_budget(std::string_view name,
                                                        std::int64_t bundle_budget_bytes,
                                                        bool retain_engines,
                                                        std::int64_t retained_tail_budget_bytes) {
    if (should_retain_hot_engine(name, retain_engines)) {
        if (name == "denoiser_head_plan" || name == "denoiser_finish_plan" ||
            name == "vae_tile_decoder_plan" || name == "audio_vae_decoder_plan") {
            return std::numeric_limits<std::int64_t>::max();
        }
        if (name == "denoiser_tail_plan")
            return std::min(bundle_budget_bytes, retained_tail_budget_bytes);
    }
    return (name == "denoiser_head_plan" || name == "denoiser_finish_plan") ? 0
                                                                            : bundle_budget_bytes;
}

} // namespace trtmc::minimax_h3
