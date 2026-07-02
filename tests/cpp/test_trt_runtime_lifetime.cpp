/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-ENG-CPP-04
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         TRT runtime/logger lifetime: logger outlives runtime to prevent use-after-free
// Preconditions:  TRT headers and runtime available
// Postconditions: Runtime factory creates valid runtime, no crash from logger lifetime issues
// =============================================================================

// Regression test: TensorRT runtime/logger lifetime.
//
// This catches bugs where IRuntime is created with a short-lived logger.
// TensorRT stores ILogger by reference; if logger dies too early, later TRT
// calls can crash with "pure virtual method called".

#include "runtime/backend/trt_logger.h"

#include <cstddef>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void test_runtime_factory_logger_lifetime() {
    for (int i = 0; i < 20; ++i) {
        auto runtime = trtmc::create_trt_runtime();
        check(static_cast<bool>(runtime), "create_trt_runtime returns non-null");
        if (!runtime) {
            continue;
        }

        // Force TensorRT to emit an error log through ILogger.
        // If logger lifetime is invalid, this path can crash.
        auto engine = trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(nullptr, static_cast<std::size_t>(0)));
        check(!engine, "deserialize invalid blob returns null engine");
    }
}

} // namespace

int main() {
    test_runtime_factory_logger_lifetime();
    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All TRT runtime lifetime tests passed.\n";
    return 0;
}
