/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-LOG-CPP-01
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         TRT logger severity names, error storage, explicit config controls
// Preconditions:  TRT headers available
// Postconditions: Logger stores errors and respects severity settings
// =============================================================================

// =============================================================================
// test_trt_logger.cpp — Unit tests for TRT logger helpers
// =============================================================================
//
// Purpose:
//   Validates the helper functions from trt_common.h/cpp that control TRT
//   logging behavior. Runtime configuration flows through configure_trt_logger()
//   after platform.* schema resolution, not through process environment.
//
// Dependencies:
//   - runtime/core/trt_common.h (trt_log_to_stderr_enabled,
//     trt_log_stderr_min_severity, trt_severity_name)
//
// Environment:
//   CPU-only. No GPU or CUDA runtime required.
// =============================================================================

#include "runtime/backend/trt_logger.h"
#include "runtime/core/trt_common.h"

#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_severity_name_error() {
    const char* name = trtmc::trt_severity_name(trtmc::TrtLogSeverity::kError);
    check(std::string(name) == "ERROR", "severity_name ERROR");
}

static void test_severity_name_warning() {
    const char* name = trtmc::trt_severity_name(trtmc::TrtLogSeverity::kWarning);
    check(std::string(name) == "WARNING", "severity_name WARNING");
}

static void test_severity_name_info() {
    const char* name = trtmc::trt_severity_name(trtmc::TrtLogSeverity::kInfo);
    check(std::string(name) == "INFO", "severity_name INFO");
}

static void test_severity_name_verbose() {
    const char* name = trtmc::trt_severity_name(trtmc::TrtLogSeverity::kVerbose);
    check(std::string(name) == "VERBOSE", "severity_name VERBOSE");
}

static void test_severity_name_internal_error() {
    const char* name = trtmc::trt_severity_name(trtmc::TrtLogSeverity::kInternalError);
    check(std::string(name) == "INTERNAL_ERROR", "severity_name INTERNAL_ERROR");
}

static void test_logger_stores_error() {
    trtmc::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, "test error message");
    check(logger.last_error() == "test error message", "logger stores error");
}

static void test_logger_clear_error() {
    trtmc::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, "first error");
    logger.clear_error();
    check(logger.last_error().empty(), "logger clear_error");
}

static void test_logger_ignores_info() {
    trtmc::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kINFO, "info message");
    check(logger.last_error().empty(), "logger ignores info");
}

static void test_logger_overwrites_error() {
    trtmc::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, "first");
    logger.log(nvinfer1::ILogger::Severity::kERROR, "second");
    check(logger.last_error() == "second", "logger overwrites error");
}

static void test_logger_null_msg() {
    trtmc::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, nullptr);
    check(logger.last_error().empty(), "logger null msg ignored");
}

static void test_log_config_defaults_and_updates() {
    trtmc::configure_trt_logger(false, "INFO");
    check(!trtmc::trt_log_to_stderr_enabled(), "log_to_stderr default false");
    check(trtmc::trt_log_stderr_min_severity() == trtmc::TrtLogSeverity::kInfo,
          "min_severity default INFO");

    trtmc::configure_trt_logger(true, "warning");
    check(trtmc::trt_log_to_stderr_enabled(), "log_to_stderr configured true");
    check(trtmc::trt_log_stderr_min_severity() == trtmc::TrtLogSeverity::kWarning,
          "min_severity configured case-insensitive WARNING");

    trtmc::configure_trt_logger(true, "not-a-severity");
    check(trtmc::trt_log_stderr_min_severity() == trtmc::TrtLogSeverity::kInfo,
          "unknown min_severity falls back to INFO");
}

static void test_min_severity_returns_valid() {
    const auto sev = trtmc::trt_log_stderr_min_severity();
    // Verify the returned severity is one of the valid enum values.
    const bool valid =
        (sev == trtmc::TrtLogSeverity::kInternalError || sev == trtmc::TrtLogSeverity::kError ||
         sev == trtmc::TrtLogSeverity::kWarning || sev == trtmc::TrtLogSeverity::kInfo ||
         sev == trtmc::TrtLogSeverity::kVerbose);
    check(valid, "min_severity valid enum");
}

int main() {
    test_severity_name_error();
    test_severity_name_warning();
    test_severity_name_info();
    test_severity_name_verbose();
    test_severity_name_internal_error();
    test_logger_stores_error();
    test_logger_clear_error();
    test_logger_ignores_info();
    test_logger_overwrites_error();
    test_logger_null_msg();
    test_log_config_defaults_and_updates();
    test_min_severity_returns_valid();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All trt_logger tests passed.\n";
    return 0;
}
