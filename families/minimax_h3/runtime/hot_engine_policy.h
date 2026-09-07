/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
    // strictly in sequence on one stream. Ref2VA likewise invokes one dynamic
    // denoiser repeatedly. Dynamic profiles otherwise reserve max-shape
    // activation memory even for a much smaller request. Reuse the native
    // TRT-RTX user-managed arena so every context is sized from its live shape;
    // the three split contexts additionally share one high-water allocation.
    return name == "denoiser_head_plan" || name == "denoiser_tail_plan" ||
           name == "denoiser_finish_plan" || name == "ref2va_denoiser_plan";
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
            // The explicit retained-tail setting is allowed to exceed the
            // bundle's portable streaming default on high-memory systems.
            // TensorRT-RTX clamps the request to the engine's streamable
            // weight size, so this remains safe for smaller plans.
            return retained_tail_budget_bytes;
    }
    return (name == "denoiser_head_plan" || name == "denoiser_finish_plan") ? 0
                                                                            : bundle_budget_bytes;
}

} // namespace trtmc::minimax_h3
