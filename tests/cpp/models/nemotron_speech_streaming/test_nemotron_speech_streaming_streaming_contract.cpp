/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_speech_streaming/rnnt_config.h"

#include <exception>
#include <iostream>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

void check_schedule(int right, int chunk_ms, int first_mel, int next_mel, int valid_out) {
    const auto s = trtmc::make_nemotron_streaming_schedule(70, right);
    check(s.att_context_left == 70, "left context is fixed at 70");
    check(s.att_context_right == right, "right context propagated");
    check(s.chunk_ms == chunk_ms, "chunk duration matches NeMo model card");
    check(s.chunk_samples == chunk_ms * 16, "chunk samples at 16 kHz");
    check(s.first_chunk_mel_frames == first_mel, "first chunk mel frames match NeMo");
    check(s.next_chunk_mel_frames == next_mel, "next chunk mel frames match NeMo");
    check(s.first_shift_mel_frames == first_mel, "first shift mel frames match NeMo");
    check(s.next_shift_mel_frames == next_mel, "next shift mel frames match NeMo");
    check(s.first_pre_encode_cache_mel_frames == 0, "first pre-encode cache matches NeMo");
    check(s.next_pre_encode_cache_mel_frames == 9, "next pre-encode cache matches NeMo");
    check(s.valid_encoder_frames == valid_out, "valid encoder frames match NeMo");
    check(s.drop_extra_pre_encoded == 2, "drop_extra_pre_encoded matches NeMo");
}

void test_supported_contexts() {
    check_schedule(0, 80, 1, 8, 1);
    check_schedule(1, 160, 9, 16, 2);
    check_schedule(6, 560, 49, 56, 7);
    check_schedule(13, 1120, 105, 112, 14);
}

void test_invalid_context_rejected() {
    bool threw = false;
    try {
        (void)trtmc::make_nemotron_streaming_schedule(70, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "unsupported right context is rejected");

    threw = false;
    try {
        (void)trtmc::make_nemotron_streaming_schedule(64, 13);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "unsupported left context is rejected");
}

} // namespace

int main() {
    test_supported_contexts();
    test_invalid_context_rejected();
    if (g_failures) {
        std::cerr << g_failures << " RNNT streaming contract test(s) failed\n";
        return 1;
    }
    return 0;
}
