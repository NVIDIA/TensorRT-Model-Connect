/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rnnt_config.h"

#include <exception>
#include <iostream>
#include <vector>

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

// nemotron-3.5-asr-streaming-0.6b ships att_context_size pairs with left=56 and
// right in {13, 6, 3, 0}. right=1 is NOT in the multilingual checkpoint, so we
// deliberately exclude it from the supported-context coverage below.
void check_3_5_schedule(int right, int chunk_ms, int first_mel, int next_mel, int valid_out) {
    const auto s = trtmc::make_nemotron_streaming_schedule(56, right);
    check(s.att_context_left == 56, "left context is 56 for nemotron-3.5");
    check(s.att_context_right == right, "right context propagated");
    check(s.chunk_ms == chunk_ms, "chunk duration matches nemotron-3.5 card");
    check(s.chunk_samples == chunk_ms * 16, "chunk samples at 16 kHz");
    check(s.first_chunk_mel_frames == first_mel, "first chunk mel frames match nemotron-3.5");
    check(s.next_chunk_mel_frames == next_mel, "next chunk mel frames match nemotron-3.5");
    check(s.first_shift_mel_frames == first_mel, "first shift mel frames match nemotron-3.5");
    check(s.next_shift_mel_frames == next_mel, "next shift mel frames match nemotron-3.5");
    check(s.first_pre_encode_cache_mel_frames == 0, "first pre-encode cache matches nemotron-3.5");
    check(s.next_pre_encode_cache_mel_frames == 9, "next pre-encode cache matches nemotron-3.5");
    check(s.valid_encoder_frames == valid_out, "valid encoder frames match nemotron-3.5");
    check(s.drop_extra_pre_encoded == 2, "drop_extra_pre_encoded matches nemotron-3.5");
}

void test_3_5_supported_contexts() {
    // right=1 is intentionally absent — not in the nemotron-3.5 checkpoint.
    check_3_5_schedule(0, 80, 1, 8, 1);
    check_3_5_schedule(3, 320, 25, 32, 4);
    check_3_5_schedule(6, 560, 49, 56, 7);
    check_3_5_schedule(13, 1120, 105, 112, 14);
}

void test_3_5_invalid_context_rejected() {
    const std::vector<int32_t> supported_right_3_5 = {0, 3, 6, 13};
    check(!trtmc::is_supported_nemotron_att_context(56, 1, 56, supported_right_3_5),
          "right=1 is not in the nemotron-3.5 supported list");
    check(!trtmc::is_supported_nemotron_att_context(56, 2, 56, supported_right_3_5),
          "right=2 is not in the nemotron-3.5 supported list");
    check(!trtmc::is_supported_nemotron_att_context(70, 13, 56, supported_right_3_5),
          "left=70 is rejected when nemotron-3.5 cache_left=56");
    check(trtmc::is_supported_nemotron_att_context(56, 0, 56, supported_right_3_5),
          "right=0 is accepted for nemotron-3.5");
    check(trtmc::is_supported_nemotron_att_context(56, 3, 56, supported_right_3_5),
          "right=3 is accepted for nemotron-3.5");
    check(trtmc::is_supported_nemotron_att_context(56, 6, 56, supported_right_3_5),
          "right=6 is accepted for nemotron-3.5");
    check(trtmc::is_supported_nemotron_att_context(56, 13, 56, supported_right_3_5),
          "right=13 is accepted for nemotron-3.5");
}

void test_invalid_context_rejected() {
    const std::vector<int32_t> supported_right_70 = {0, 1, 6, 13};
    check(!trtmc::is_supported_nemotron_att_context(70, 2, 70, supported_right_70),
          "unsupported right context is rejected");
    check(!trtmc::is_supported_nemotron_att_context(64, 13, 70, supported_right_70),
          "unsupported left context is rejected");
}

} // namespace

int main() {
    test_supported_contexts();
    test_3_5_supported_contexts();
    test_invalid_context_rejected();
    test_3_5_invalid_context_rejected();
    if (g_failures) {
        std::cerr << g_failures << " RNNT streaming contract test(s) failed\n";
        return 1;
    }
    return 0;
}
