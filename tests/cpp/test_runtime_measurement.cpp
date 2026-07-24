/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/measurement.h"

#include <cassert>

int main() {
    trtmc::runtime_measurement::reset_model_call();
    assert(!trtmc::runtime_measurement::model_call_started());

    trtmc::runtime_measurement::record_model_call_start();
    assert(trtmc::runtime_measurement::model_call_started());
    assert(trtmc::runtime_measurement::model_call_wall_ms() >= 0.0);

    trtmc::runtime_measurement::reset_model_call();
    assert(!trtmc::runtime_measurement::model_call_started());
    return 0;
}
