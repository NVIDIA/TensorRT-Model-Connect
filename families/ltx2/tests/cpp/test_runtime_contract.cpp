/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// LTX-2.5 host-side runtime contract: scheduler step, prompt padding, audio interleave.

#include "families/ltx2/runtime/progress_log.h"
#include "families/ltx2/runtime/runtime_math.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void test_euler_step_matches_flow_match_euler() {
    // diffusers: prev = x + (sigma_next - sigma) * v; the last distilled step lands on x0.
    std::vector<float> x{1.0F, -2.0F, 0.5F};
    const std::vector<float> v{0.5F, 1.0F, -4.0F};
    trtmc::ltx2_euler_step(x, v, 0.421875F, 0.0F);
    check(std::fabs(x[0] - (1.0F - 0.421875F * 0.5F)) < 1e-7F, "euler step value 0");
    check(std::fabs(x[1] - (-2.0F - 0.421875F)) < 1e-7F, "euler step value 1");
    check(std::fabs(x[2] - (0.5F + 0.421875F * 4.0F)) < 1e-7F, "euler step value 2");
    bool threw = false;
    try {
        std::vector<float> short_v{1.0F};
        trtmc::ltx2_euler_step(x, short_v, 1.0F, 0.5F);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "euler step rejects mismatched sizes");
}

void test_prompt_ids_left_pad_and_truncate() {
    std::vector<int32_t> ids;
    std::vector<int32_t> mask;
    trtmc::ltx2_prompt_ids({11, 12, 13}, 6, 0, ids, mask);
    check((ids == std::vector<int32_t>{0, 0, 0, 11, 12, 13}), "left padding");
    check((mask == std::vector<int32_t>{0, 0, 0, 1, 1, 1}), "mask marks tokens");
    trtmc::ltx2_prompt_ids({1, 2, 3, 4, 5}, 3, 0, ids, mask);
    check((ids == std::vector<int32_t>{1, 2, 3}), "right truncation keeps the first tokens");
    check((mask == std::vector<int32_t>{1, 1, 1}), "truncated mask is full");
}

void test_interleave_stereo() {
    const auto out = trtmc::ltx2_interleave({1.0F, 2.0F, 3.0F, -1.0F, -2.0F, -3.0F}, 2);
    check((out == std::vector<float>{1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}),
          "planar to interleaved");
}

void test_progress_line_format() {
    const auto line = trtmc::format_ltx2_progress(1, 12.5, "step", "step=2/8 step_ms=3.000");
    check(line == "[ltx-progress] rank=1 t_ms=12.500 event=step step=2/8 step_ms=3.000",
          "progress line keeps the LTX format");
}

} // namespace

int main() {
    test_euler_step_matches_flow_match_euler();
    test_prompt_ids_left_pad_and_truncate();
    test_interleave_stereo();
    test_progress_line_format();
    if (failures != 0)
        return EXIT_FAILURE;
    std::puts("ltx2 runtime contract: OK");
    return EXIT_SUCCESS;
}
