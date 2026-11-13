/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/canary_stage_timing.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

bool near(double actual, double expected) {
    return std::abs(actual - expected) < 1e-12;
}

trtmc::TextResult segment(const char* text, std::vector<std::int32_t> token_ids) {
    return trtmc::TextResult{text, std::move(token_ids)};
}

void test_elapsed_time_is_monotonic() {
    const auto started = trtmc::CanaryStageClock::now();
    const double first = trtmc::canary_stage_elapsed_ms(started);
    const double second = trtmc::canary_stage_elapsed_ms(started);
    check(first >= 0.0 && second >= first, "canary stage clock never runs backwards");
}

void test_single_segment_chunk_keeps_the_whole_wall() {
    std::vector<trtmc::TextResult> results{segment("hello", {7, 8})};
    trtmc::share_canary_chunk_stage_times({12.0, 30.0, 40.0}, {0}, 0, 1, {0}, results);
    check(near(results[0].setup_ms, 12.0) && near(results[0].prefill_ms, 30.0) &&
              near(results[0].decode_ms, 40.0),
          "a one-segment chunk reports its stage walls unchanged");
    check(results[0].text == "hello" && results[0].token_ids == std::vector<std::int32_t>({7, 8}),
          "sharing stage times leaves the transcript and token ids untouched");
}

void test_chunk_walls_are_shared_and_conserved() {
    // Segments 4, 1 and 3 form one chunk of a larger work list; segment 1's
    // mel extraction failed, so only 4 and 3 reached the engines.
    std::vector<trtmc::TextResult> results(5);
    results[1] = segment("[mel extraction failed]", {});
    const std::vector<std::size_t> indices{0, 4, 1, 3, 2};
    trtmc::share_canary_chunk_stage_times({9.0, 20.0, 50.0}, indices, 1, 4, {4, 3}, results);

    check(near(results[4].setup_ms, 3.0) && near(results[1].setup_ms, 3.0) &&
              near(results[3].setup_ms, 3.0),
          "every prepared segment takes an equal share of the frontend wall");
    check(near(results[4].prefill_ms, 10.0) && near(results[3].prefill_ms, 10.0) &&
              near(results[4].decode_ms, 25.0) && near(results[3].decode_ms, 25.0),
          "segments that reached the engines share the encoder and decoder walls");
    check(results[1].prefill_ms == 0.0 && results[1].decode_ms == 0.0 &&
              results[1].text == "[mel extraction failed]",
          "a segment that never reached the engines reports no engine time");
    check(results[0].setup_ms == 0.0 && results[2].setup_ms == 0.0 &&
              results[0].prefill_ms == 0.0 && results[2].decode_ms == 0.0,
          "segments outside the chunk are not charged");

    trtmc::TextResult combined;
    for (const auto& result : results)
        trtmc::add_canary_stage_times(combined, result);
    check(near(combined.setup_ms, 9.0) && near(combined.prefill_ms, 20.0) &&
              near(combined.decode_ms, 50.0),
          "summing a request's segments returns each chunk wall exactly once");
}

void test_chunk_without_engine_segments_reports_only_the_frontend() {
    std::vector<trtmc::TextResult> results(2);
    trtmc::share_canary_chunk_stage_times({8.0, 0.0, 0.0}, {0, 1}, 0, 2, {}, results);
    check(near(results[0].setup_ms, 4.0) && near(results[1].setup_ms, 4.0) &&
              results[0].prefill_ms == 0.0 && results[1].decode_ms == 0.0,
          "a chunk whose segments all failed still reports its frontend wall");
}

void test_merge_accumulates_all_three_stages() {
    trtmc::TextResult combined;
    trtmc::TextResult first;
    first.setup_ms = 1.5;
    first.prefill_ms = 2.5;
    first.decode_ms = 3.5;
    trtmc::TextResult second;
    second.setup_ms = 0.5;
    second.prefill_ms = 1.0;
    second.decode_ms = 2.0;
    trtmc::add_canary_stage_times(combined, first);
    trtmc::add_canary_stage_times(combined, second);
    check(near(combined.setup_ms, 2.0) && near(combined.prefill_ms, 3.5) &&
              near(combined.decode_ms, 5.5),
          "merging segments sums setup, prefill and decode times");
}

} // namespace

int main() {
    test_elapsed_time_is_monotonic();
    test_single_segment_chunk_keeps_the_whole_wall();
    test_chunk_walls_are_shared_and_conserved();
    test_chunk_without_engine_segments_reports_only_the_frontend();
    test_merge_accumulates_all_three_stages();
    return failures;
}
