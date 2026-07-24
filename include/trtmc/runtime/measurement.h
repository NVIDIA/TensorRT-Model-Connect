/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace trtmc::runtime_measurement {

// Per-thread benchmark instrumentation. The TensorRT backend records the first
// module invocation after reset; callers can then measure through the public
// operation return without including pipeline preprocessing before that call.
void reset_model_call() noexcept;
void record_model_call_start() noexcept;
bool model_call_started() noexcept;
double model_call_wall_ms() noexcept;

} // namespace trtmc::runtime_measurement
