/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/personaplex/speech_performance.h"

#include <cmath>
#include <iostream>

namespace {

int g_failures = 0;

void check_close(double actual, double expected, const char* name) {
    if (std::fabs(actual - expected) <= 1.0e-9)
        return;
    std::cerr << "FAIL: " << name << " expected " << expected << ", got " << actual << '\n';
    ++g_failures;
}

} // namespace

int main() {
    trtmc::SpeechPerformanceTimings timings;
    timings.temporal_ms = 350.0;
    timings.depth_ms = 200.0;
    timings.codec_ms = 20.0;
    check_close(timings.host_ms(600.0), 30.0, "host residual");
    check_close(timings.host_ms(500.0), 0.0, "negative residual clamps to zero");
    return g_failures == 0 ? 0 : 1;
}
