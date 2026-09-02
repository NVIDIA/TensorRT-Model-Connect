/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/parakeet_tdt/tdt_config.h"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_duration_value_controls_frame_advance() {
    const auto decision = trtmc::make_tdt_greedy_decision(7, 2, {0, 1, 3}, 9);
    check(decision.emit_token, "nonblank should emit");
    check(decision.frame_advance == 3, "duration table value should advance frames");
}

void test_zero_duration_blank_forces_progress_without_emit() {
    const auto decision = trtmc::make_tdt_greedy_decision(9, 0, {0, 1, 3}, 9);
    check(!decision.emit_token, "blank should not emit");
    check(decision.frame_advance == 1, "zero-duration blank must force progress");
}

void test_zero_duration_nonblank_stays_on_frame() {
    const auto decision = trtmc::make_tdt_greedy_decision(7, 0, {0, 1, 3}, 9);
    check(decision.emit_token, "zero-duration nonblank should emit");
    check(decision.frame_advance == 0, "zero-duration nonblank must stay on the frame");
}

void test_invalid_duration_index_is_rejected() {
    bool threw = false;
    try {
        (void)trtmc::make_tdt_greedy_decision(7, 3, {0, 1, 2}, 9);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "duration indices outside the table must throw");
}

void test_negative_duration_is_rejected() {
    bool threw = false;
    try {
        (void)trtmc::make_tdt_greedy_decision(7, 1, {0, -1, 2}, 9);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "negative duration values must throw");
}

void test_streaming_schedule_derives_timing_and_cache_geometry() {
    const auto schedule = trtmc::make_tdt_streaming_schedule(20, 3, 8000, 80, 4);
    check(schedule.encoder_frame_ms == 40, "encoder frame timing derives from hop and stride");
    check(schedule.chunk_samples == 1280, "chunk samples derive from schedule timing");
    check(schedule.next_pre_encode_cache_mel_frames == 5,
          "pre-encode cache derives from subsampling");
    check(schedule.drop_extra_pre_encoded == 2, "pre-encode drop derives from cache overlap");
}

void test_mel_geometry_accepts_matching_dimensions() {
    trtmc::TdtConfig config;
    trtmc::validate_tdt_mel_geometry(config, 257, 128);
}

void test_mel_geometry_rejects_invalid_or_mismatched_dimensions() {
    trtmc::TdtConfig config;
    config.mel_n_fft = 0;
    bool invalid_threw = false;
    try {
        trtmc::validate_tdt_mel_geometry(config, 257, 128);
    } catch (const std::invalid_argument&) {
        invalid_threw = true;
    }
    check(invalid_threw, "non-positive mel dimensions must throw");

    config.mel_n_fft = 512;
    bool mismatch_threw = false;
    try {
        trtmc::validate_tdt_mel_geometry(config, 256, 128);
    } catch (const std::invalid_argument&) {
        mismatch_threw = true;
    }
    check(mismatch_threw, "filterbank dimensions must match the runtime config");
}

} // namespace

int main() {
    test_duration_value_controls_frame_advance();
    test_zero_duration_blank_forces_progress_without_emit();
    test_zero_duration_nonblank_stays_on_frame();
    test_invalid_duration_index_is_rejected();
    test_negative_duration_is_rejected();
    test_streaming_schedule_derives_timing_and_cache_geometry();
    test_mel_geometry_accepts_matching_dimensions();
    test_mel_geometry_rejects_invalid_or_mismatched_dimensions();
    if (failures) {
        std::cerr << failures << " TDT decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}
