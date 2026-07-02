/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace trtmc {

struct SpeechOutputPlan {
    int32_t effective_frames{0};
    int32_t extra_tail{0};
    int32_t output_frames{0};
    int32_t total_iters{0};
};

struct SpeechOutputPlanInput {
    int32_t sample_rate{0};
    float frame_rate{0.0F};
    int32_t num_frames{0};
    int32_t num_input_samples{0};
    int32_t input_sample_rate{0};
    int32_t tail_frames{0};
    int32_t max_output_frames{0};
    int32_t max_delay{0};
};

inline SpeechOutputPlan ComputeSpeechOutputPlan(const SpeechOutputPlanInput& input) {
    const int32_t nominal_frame_size =
        (input.frame_rate > 0.0F) ? static_cast<int32_t>(std::lround(
                                        static_cast<float>(input.sample_rate) / input.frame_rate))
                                  : 0;
    int32_t nominal_input_frames = input.num_frames;
    if (nominal_frame_size > 0) {
        int64_t effective_input_samples = input.num_input_samples;
        if (input.input_sample_rate > 0 && input.input_sample_rate != input.sample_rate) {
            effective_input_samples =
                (effective_input_samples * input.sample_rate) / input.input_sample_rate;
        }
        nominal_input_frames = static_cast<int32_t>(effective_input_samples / nominal_frame_size);
    }

    SpeechOutputPlan plan;
    plan.effective_frames = std::min(input.num_frames, nominal_input_frames);
    plan.effective_frames = std::max(0, plan.effective_frames - 2);
    plan.extra_tail = std::max(0, input.tail_frames);

    int64_t target_frames =
        static_cast<int64_t>(plan.effective_frames) + static_cast<int64_t>(plan.extra_tail);
    target_frames = std::max<int64_t>(0, target_frames);
    plan.output_frames =
        std::min(static_cast<int32_t>(std::min<int64_t>(
                     target_frames, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))),
                 input.max_output_frames);
    plan.total_iters = plan.output_frames + input.max_delay + 1;
    return plan;
}

} // namespace trtmc
