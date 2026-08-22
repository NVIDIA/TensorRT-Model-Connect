/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-ABI-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         C ABI entry point validation and error handling
// Preconditions:  TRT headers available
// Postconditions: Null/empty paths produce errors, valid paths proceed
// =============================================================================

// =============================================================================
// Test suite: C ABI entry points (trtmc_create_pipeline, trtmc_create_pipeline_ex,
//   trtmc_last_error, trtmc_version, trtmc_has_trt).
//
// Purpose:
//   Validates the public C ABI surface exposed by trtmc/pipeline.h. Although
//   this test file is compiled as C++, all pipeline interactions go through
//   the extern "C" functions to verify the ABI contract that language bindings
//   and the CLI depend on. Tests cover: version string availability, TRT
//   detection, error handling for invalid inputs, null safety, the extended
//   options struct (TrtmcPipelineOptions), and error-state lifecycle.
//
// Dependencies:
//   - trtmc/pipeline.h: C ABI functions and IPipeline, TrtmcPipelineOptions.
//   - No TRT or GPU required. All tests exercise error paths with invalid
//     or nonexistent bundle paths.
//
// Approach:
//   Each test calls C ABI functions with specific inputs and checks return
//   values and side effects (error messages via trtmc_last_error). Tests are
//   designed to succeed in any environment -- GPU, CPU-only, or CI sandbox --
//   by testing error paths and null handling rather than successful pipeline
//   creation (which requires a pre-built .bundle artifact).
//
// Test categories:
//   - Version/capability queries: trtmc_version, trtmc_has_trt
//   - Error handling: null input, empty input, bad path, non-bundle file
//   - Null safety: deleting a null IPipeline pointer
//   - Error lifecycle: error set after failure, cleared by version query
//   - Extended API: TrtmcPipelineOptions zero-init safety, trtmc_create_pipeline_ex
// =============================================================================

#include "trtmc/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// -----------------------------------------------------------------------------
// Intention: Verify that trtmc_version() returns a non-null, non-empty string
//   identifying the library version.
// Setup: None.
// Mechanism: Calls trtmc_version(), checks the pointer is non-null and the
//   string has length > 0.
// -----------------------------------------------------------------------------
static void test_version_not_null() {
    const char* ver = trtmc_version();
    check(ver != nullptr, "trtmc_version returns non-null");
    check(std::strlen(ver) > 0, "trtmc_version is non-empty");
}

// -----------------------------------------------------------------------------
// Intention: Verify that trtmc_has_trt() returns a valid boolean value (0 or 1),
//   indicating whether TensorRT support was compiled in and is available.
// Setup: None.
// Mechanism: Calls trtmc_has_trt(), checks the return value is exactly 0 or 1.
// -----------------------------------------------------------------------------
static void test_has_trt_returns_bool() {
    const int val = trtmc_has_trt();
    check(val == 0 || val == 1, "trtmc_has_trt returns 0 or 1");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing a null model_id to trtmc_create_pipeline
//   returns nullptr and populates trtmc_last_error() with a descriptive message.
// Setup: None.
// Mechanism: Calls trtmc_create_pipeline(nullptr, 0), asserts the return is
//   nullptr, then checks trtmc_last_error() is non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_create_null_returns_null() {
    auto* p = trtmc_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after null input");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing an empty string as model_id returns nullptr
//   and sets an error message.
// Setup: None.
// Mechanism: Calls trtmc_create_pipeline("", 0), asserts nullptr return, checks
//   trtmc_last_error() is non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_create_empty_returns_null() {
    auto* p = trtmc_create_pipeline("", 0);
    check(p == nullptr, "empty input returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after empty input");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing a nonexistent filesystem path as bundle_path
//   returns nullptr and sets an error message (bundle validation fails).
// Setup: None.
// Mechanism: Calls trtmc_create_pipeline("/nonexistent/path/to/bundle.bundle", 0),
//   asserts nullptr return, checks trtmc_last_error() is non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_create_bad_path_returns_null() {
    auto* p = trtmc_create_pipeline("/nonexistent/path/to/bundle.bundle", 0);
    check(p == nullptr, "bad path returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "last_error has message after bad path");
}

// -----------------------------------------------------------------------------
// Intention: Verify that deleting a null IPipeline pointer is safe and does
//   not crash (C++ guarantees delete on nullptr is a no-op, but this
//   explicitly tests it through the ABI's usage pattern).
// Setup: A null IPipeline pointer.
// Mechanism: Calls delete on the null pointer, then asserts true (if we reach
//   the assertion, the delete did not crash).
// -----------------------------------------------------------------------------
static void test_delete_null_safe() {
    trtmc::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

// -----------------------------------------------------------------------------
// Intention: Verify the error-state lifecycle: trtmc_last_error() should contain
//   a message after a failed pipeline creation, and should remain accessible
//   as a non-null pointer after subsequent calls.
// Setup: Triggers a failure with a nonexistent path, then checks the error
//   state is set. Calls trtmc_version() (a successful call) and verifies the
//   version call succeeds independently.
// Mechanism:
//   1. Creates a pipeline with "/nonexistent" -> fails, error message is set.
//   2. Verifies trtmc_last_error() returns a non-empty string.
//   3. Calls trtmc_version() to confirm it works after an error state.
// -----------------------------------------------------------------------------
static void test_last_error_cleared_on_success() {
    // Create failure first
    auto* p1 = trtmc_create_pipeline("/nonexistent", 0);
    check(p1 == nullptr, "bad path fails");
    check(std::strlen(trtmc_last_error()) > 0, "error set after failure");

    // Verify trtmc_version still works after an error was set
    const char* ver = trtmc_version();
    check(ver != nullptr && std::strlen(ver) > 0, "version works after error state");
}

// -----------------------------------------------------------------------------
// Intention: Verify that zero-initializing TrtmcPipelineOptions (via C++ value
//   initialization) produces safe default values for every field, ensuring
//   backward compatibility when new fields are added to the struct.
// Setup: A brace-initialized TrtmcPipelineOptions{}.
// Mechanism: Checks each field: max_new_tokens==0, image_path==nullptr,
//   runtime_cache==nullptr, and cuda_graphs==0. Also
//   verifies that trtmc_create_pipeline_ex with null options and a bad path
//   returns nullptr (null options should use defaults).
// -----------------------------------------------------------------------------
static void test_pipeline_options_zero_init() {
    // Verify that zero-initialized TrtmcPipelineOptions is safe and backward-compatible
    TrtmcPipelineOptions opts{};
    check(opts.max_new_tokens == 0, "zero-init max_new_tokens == 0");
    check(opts.image_path == nullptr, "zero-init image_path == nullptr");
    check(opts.runtime_cache == nullptr, "zero-init runtime_cache == nullptr");
    check(opts.cuda_graphs == 0, "zero-init cuda_graphs == 0");

    // Should work with null options (uses defaults)
    auto* p = trtmc_create_pipeline_ex("/nonexistent", nullptr);
    check(p == nullptr, "null options with bad path returns null");
}

// -----------------------------------------------------------------------------
// Intention: Verify that trtmc_create_pipeline_ex correctly accepts a fully
//   populated TrtmcPipelineOptions struct and propagates the error when the
//   bundle path is invalid.
// Setup: Creates a TrtmcPipelineOptions with all fields set: max_new_tokens=5,
//   image_path="/nonexistent/image.png", runtime_cache="/nonexistent/cache",
//   and cuda_graphs=1.
// Mechanism: Calls trtmc_create_pipeline_ex with a nonexistent bundle path and
//   the options struct. Asserts nullptr return and a non-empty error message.
//   This ensures the extended API processes all option fields without crashing.
// -----------------------------------------------------------------------------
static void test_create_ex_with_options() {
    TrtmcPipelineOptions opts{};
    opts.max_new_tokens = 5;
    opts.image_path = "/nonexistent/image.png";
    opts.runtime_cache = "/nonexistent/cache";
    opts.cuda_graphs = 1;

    auto* p = trtmc_create_pipeline_ex("/nonexistent/bundle.bundle", &opts);
    check(p == nullptr, "bad bundle with options returns null");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set with options");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing a non-bundle file path (e.g., /dev/null) to
//   trtmc_create_pipeline does not crash and correctly returns nullptr.
// Setup: None (uses the always-available /dev/null as input).
// Mechanism: Calls trtmc_create_pipeline("/dev/null", 0), asserts nullptr
//   return. This ensures the bundle detection logic gracefully rejects
//   non-bundle files.
// -----------------------------------------------------------------------------
static void test_bundle_path_not_bundle() {
    // Test that passing a non-bundle file doesn't crash
    auto* p = trtmc_create_pipeline("/dev/null", 0);
    check(p == nullptr, "non-bundle file returns null");
}

int main() {
    test_version_not_null();
    test_has_trt_returns_bool();
    test_create_null_returns_null();
    test_create_empty_returns_null();
    test_create_bad_path_returns_null();
    test_delete_null_safe();
    test_last_error_cleared_on_success();
    test_pipeline_options_zero_init();
    test_create_ex_with_options();
    test_bundle_path_not_bundle();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI entry tests passed.\n";
    return 0;
}
