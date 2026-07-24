/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/measurement.h"

#include <chrono>
#include <optional>

namespace trtmc::runtime_measurement {
namespace {

using Clock = std::chrono::steady_clock;
thread_local std::optional<Clock::time_point> first_model_call;

} // namespace

void reset_model_call() noexcept {
    first_model_call.reset();
}

void record_model_call_start() noexcept {
    if (!first_model_call.has_value()) {
        first_model_call = Clock::now();
    }
}

bool model_call_started() noexcept {
    return first_model_call.has_value();
}

double model_call_wall_ms() noexcept {
    if (!first_model_call.has_value()) {
        return -1.0;
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - *first_model_call).count();
}

} // namespace trtmc::runtime_measurement
