/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Opt-in LTX-2.5 progress log (same line format as LTX-Video) with monotonic timestamps.
//
// Set TRTMC_LTX2_PROGRESS=1 to print one flushed stdout line per generation
// phase boundary and per denoising step, e.g.
//
//   [ltx-progress] rank=0 t_ms=1234.567 event=step step=3/50 step_ms=305.123
//
// t_ms is std::chrono::steady_clock time since the start of generate_image
// (text encoder -> denoise -> video and audio decode), after all engines are loaded. Unset,
// empty or "0" disables the log; nothing else about the run changes.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace trtmc {

constexpr const char* kLTX2ProgressEnv = "TRTMC_LTX2_PROGRESS";

inline bool ltx2_progress_enabled(const char* value) {
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

inline std::string format_ltx2_progress(int32_t rank, double t_ms, const std::string& event,
                                        const std::string& detail) {
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%.3f", t_ms);
    std::ostringstream line;
    line << "[ltx-progress] rank=" << rank << " t_ms=" << stamp << " event=" << event;
    if (!detail.empty())
        line << " " << detail;
    return line.str();
}

class LTX2ProgressLog {
  public:
    using Clock = std::chrono::steady_clock;

    explicit LTX2ProgressLog(int32_t rank = 0)
        : enabled_(ltx2_progress_enabled(std::getenv(kLTX2ProgressEnv))), rank_(rank),
          origin_(Clock::now()) {}

    bool enabled() const { return enabled_; }

    // Resets t=0 and emits event=start.
    void start(const std::string& detail = "") {
        origin_ = Clock::now();
        emit("start", detail);
    }

    void emit(const std::string& event, const std::string& detail = "") const {
        if (!enabled_)
            return;
        const double t_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - origin_).count();
        std::cout << format_ltx2_progress(rank_, t_ms, event, detail) << std::endl;
    }

  private:
    bool enabled_;
    int32_t rank_;
    Clock::time_point origin_;
};

} // namespace trtmc
