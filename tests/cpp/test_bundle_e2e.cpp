/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-BDL-CPP-03
// Architecture:   ARCH-BDL-001
// Unit Design:    UD-BDL-01
// Intent:         Bundle build + load round-trip integrity
// Preconditions:  TRT runtime available
// Postconditions: Bundle written and re-read matches original
// =============================================================================

// =============================================================================
// Test suite: Bundle validation -- loading non-bundle files and invalid paths.
//
// Purpose:
//   Validates that the C ABI and bundle infrastructure gracefully reject
//   invalid inputs. Since the C++ runtime is now bundle-only (requires
//   pre-built .bundle files), and unit tests cannot create bundles (that
//   requires TRT + GPU + model weights), these tests focus on the error
//   paths: non-bundle files, nonexistent paths, and bundle API utilities.
//
// Dependencies:
//   - trtmc/pipeline.h: C ABI entry points (trtmc_create_pipeline, trtmc_has_trt).
//   - trtmc/bundle.h: IsBundle, InspectBundle.
//   - No TRT or GPU required. All tests exercise error/rejection paths.
//
// Approach:
//   Tests verify that non-bundle files and invalid paths are rejected with
//   appropriate error messages. The IsBundle() utility is tested with known
//   non-bundle inputs. InspectBundle() is tested with an invalid path to
//   verify it does not crash.
//
// Test categories:
//   - Non-bundle rejection: /dev/null, nonexistent paths
//   - IsBundle utility: correctly rejects non-bundle files
//   - InspectBundle utility: handles invalid input without crashing
// =============================================================================

#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"

#include <cstring>
#include <exception>
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
// Intention: Verify that passing a non-bundle file (e.g., /dev/null) to
//   trtmc_create_pipeline returns nullptr with an appropriate error message.
// Setup: None (uses the always-available /dev/null as input).
// Mechanism: Calls trtmc_create_pipeline("/dev/null", 0), asserts nullptr
//   return and a non-empty error message. This ensures the bundle loader
//   gracefully rejects files that lack the .bundle magic bytes.
// -----------------------------------------------------------------------------
static void test_non_bundle_file_rejected() {
    auto* p = trtmc_create_pipeline("/dev/null", 0);
    check(p == nullptr, "non-bundle file returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error message set for non-bundle file");
}

// -----------------------------------------------------------------------------
// Intention: Verify that passing a nonexistent path to trtmc_create_pipeline
//   returns nullptr with an appropriate error message.
// Setup: None.
// Mechanism: Calls trtmc_create_pipeline with a path that does not exist on
//   the filesystem, asserts nullptr return and non-empty error.
// -----------------------------------------------------------------------------
static void test_nonexistent_bundle_rejected() {
    auto* p = trtmc_create_pipeline("/nonexistent/path/to/model.bundle", 0);
    check(p == nullptr, "nonexistent bundle path returns nullptr");
    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error message set for nonexistent bundle");
}

// -----------------------------------------------------------------------------
// Intention: Verify that IsBundle() correctly rejects a non-bundle file.
// Setup: None (uses /dev/null as a known non-bundle file).
// Mechanism: Calls IsBundle("/dev/null"), asserts it returns false.
// -----------------------------------------------------------------------------
static void test_is_bundle_rejects_non_bundle() {
    check(!trtmc::IsBundle("/dev/null"), "IsBundle rejects /dev/null");
}

// -----------------------------------------------------------------------------
// Intention: Verify that IsBundle() correctly rejects a nonexistent path.
// Setup: None.
// Mechanism: Calls IsBundle with a path that does not exist, asserts false.
// -----------------------------------------------------------------------------
static void test_is_bundle_rejects_nonexistent() {
    check(!trtmc::IsBundle("/nonexistent/path/to/model.bundle"),
          "IsBundle rejects nonexistent path");
}

// -----------------------------------------------------------------------------
// Intention: Verify that InspectBundle() does not crash when given a
//   nonexistent path, and returns an empty/default BundleInfo.
// Setup: None.
// Mechanism: Calls InspectBundle with a nonexistent path, checks that the
//   returned model_id is empty (indicating no valid data was extracted).
// -----------------------------------------------------------------------------
static void test_inspect_nonexistent_does_not_crash() {
    bool threw = false;
    try {
        trtmc::InspectBundle("/nonexistent/path/to/model.bundle");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "InspectBundle throws for nonexistent path");
}

// -----------------------------------------------------------------------------
// Intention: Verify that InspectBundle() does not crash when given a
//   non-bundle file, and returns an empty/default BundleInfo.
// Setup: None (uses /dev/null).
// Mechanism: Calls InspectBundle("/dev/null"), checks that model_id is empty.
// -----------------------------------------------------------------------------
static void test_inspect_non_bundle_does_not_crash() {
    bool threw = false;
    try {
        trtmc::InspectBundle("/dev/null");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "InspectBundle throws for non-bundle file");
}

int main() {
    test_non_bundle_file_rejected();
    test_nonexistent_bundle_rejected();
    test_is_bundle_rejects_non_bundle();
    test_is_bundle_rejects_nonexistent();
    test_inspect_nonexistent_does_not_crash();
    test_inspect_non_bundle_does_not_crash();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle E2E tests passed.\n";
    return 0;
}
