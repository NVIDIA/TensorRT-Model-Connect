/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-15
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech output planning for resampling, clamping, and partial frames
// Preconditions:  Output plans with varied rates, frame limits, and tail lengths
// Postconditions:  Derived frame counts remain bounded and preserve partial codec frames
// =============================================================================

#include "runtime/models/personaplex/speech_generation_policy.h"

#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void test_output_plan_resample_path() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 12.5F;
    input.num_frames = 50;
    input.num_input_samples = 96000;
    input.input_sample_rate = 48000;
    input.tail_frames = 5;
    input.max_output_frames = 100;
    input.max_delay = 4;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 23, "output plan: resample effective frames");
    check(plan.extra_tail == 5, "output plan: resample extra tail");
    check(plan.output_frames == 28, "output plan: resample output frames");
    check(plan.total_iters == 33, "output plan: resample total iters");
}

void test_output_plan_frame_rate_disabled_and_clamped() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 0.0F;
    input.num_frames = 10;
    input.num_input_samples = 12345;
    input.input_sample_rate = 16000;
    input.tail_frames = -3;
    input.max_output_frames = 4;
    input.max_delay = 2;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 8, "output plan: disabled frame-rate effective frames");
    check(plan.extra_tail == 0, "output plan: negative tail clamps to zero");
    check(plan.output_frames == 4, "output plan: max output clamp");
    check(plan.total_iters == 7, "output plan: disabled frame-rate total iters");
}

void test_output_plan_small_inputs_do_not_go_negative() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 12.5F;
    input.num_frames = 1;
    input.num_input_samples = 0;
    input.input_sample_rate = 24000;
    input.tail_frames = 3;
    input.max_output_frames = 10;
    input.max_delay = 1;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 0, "output plan: effective frames floor");
    check(plan.output_frames == 3, "output plan: small input output frames");
    check(plan.total_iters == 5, "output plan: small input total iters");
}

void test_output_plan_large_target_clamps_to_max_output() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 0.0F;
    input.num_frames = 100;
    input.num_input_samples = 1;
    input.input_sample_rate = 0;
    input.tail_frames = 50;
    input.max_output_frames = 60;
    input.max_delay = 4;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 98, "output plan: large target effective frames");
    check(plan.extra_tail == 50, "output plan: large target tail");
    check(plan.output_frames == 60, "output plan: large target output clamp");
    check(plan.total_iters == 65, "output plan: large target total iters");
}

void test_output_plan_keeps_the_final_partial_codec_frame() {
    trtmc::SpeechOutputPlanInput input;
    input.sample_rate = 24000;
    input.frame_rate = 12.5F;
    input.num_frames = 345;
    input.num_input_samples = 661632;
    input.input_sample_rate = 24000;
    input.max_output_frames = 400;
    input.max_delay = 1;

    const auto plan = trtmc::ComputeSpeechOutputPlan(input);

    check(plan.effective_frames == 343, "output plan: partial frame uses ceil semantics");
    check(plan.output_frames == 343, "output plan: partial frame matches official output length");
}

} // namespace

int main() {
    test_output_plan_resample_path();
    test_output_plan_frame_rate_disabled_and_clamped();
    test_output_plan_small_inputs_do_not_go_negative();
    test_output_plan_large_target_clamps_to_max_output();
    test_output_plan_keeps_the_final_partial_codec_frame();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cout << "All PersonaPlex speech output plan tests passed\n";
    return 0;
}
