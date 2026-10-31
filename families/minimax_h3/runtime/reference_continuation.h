/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/reference_denoiser.h"

namespace trtmc::minimax_h3 {

struct ReferenceContinuationWindow {
    int32_t start_frame;
    int32_t end_frame;
    int32_t overlap_frames;
    int32_t video_latent_frames;
    int32_t audio_latents;
    int32_t guide_video_latent_frames;
    int32_t guide_audio_latents;
    double temporal_offset;
};

// Source-defined 17n+5 video grid, 24-fps video and 40-Hz audio latent clock.
// Round only global audio slice boundaries; keep fractional rotary positions.
// The explicit window cap bounds allocation, not a model/profile declaration.
std::vector<ReferenceContinuationWindow> plan_reference_continuation(int32_t total_frames,
                                                                     int32_t window_frames,
                                                                     int32_t overlap_frames,
                                                                     int32_t maximum_windows);

// Only family-owned normalized latents are accepted here. These helpers neither
// encode media nor establish accepted parent pixels or public state ownership.
ReferenceLatentGuide reference_latent_guide(const ReferenceDenoiserResult& history,
                                            const ReferenceContinuationWindow& window);
ReferenceDenoiserResult append_reference_window(const ReferenceDenoiserResult& history,
                                                const ReferenceDenoiserResult& generated,
                                                const ReferenceContinuationWindow& window);

} // namespace trtmc::minimax_h3
