/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace trtmc::sam2 {

// Schema v2 adds the post-Q1 regular-receipt binding. Production pins were
// empty when this schema was introduced, so v1 is intentionally rejected.
inline constexpr std::int32_t kNativeQualificationRecordSchemaVersion = 2;
inline constexpr std::string_view kNativeQualificationRecordArtifactType =
    "sam2_native_qualification_record";
inline constexpr std::string_view kNativeSemanticAccuracyPolicyId = "sam2_semantic_accuracy_v1";
// Exact unrounded semantic gates named by kNativeSemanticAccuracyPolicyId.
// Keeping the values beside the authority identifier prevents a policy-name
// pin from silently drifting away from the benchmark that produced it.
inline constexpr double kNativeMinimumFrameMaskIou = 0.98;
inline constexpr double kNativeMinimumMacroMaskIou = 0.99;
inline constexpr double kNativeMinimumGlobalMaskIou = 0.99;
inline constexpr double kNativeMinimumBboxIou = 0.995;
inline constexpr double kNativeMaximumBboxCoordinateError = 0.5;
inline constexpr double kNativeMaximumBboxScoreError = 0.01;
inline constexpr std::string_view kGoldenMasksSha256 =
    "1c7830b37739e409fbb8dab2b81c31c63b3379e6c10ae9e6b4ca2cc48a656094";

struct NativeQualificationScope {
    std::string family;
    std::string model_id;
    std::uint32_t engine_contract_version{0U};
    std::string runtime_strategy;
    std::string precision;
    std::string gpu_name;
    std::string compute_capability;
    std::string tensorrt_version;
    std::string tensorrt_abi;
};

struct NativeQualificationPlanBinding {
    std::string section;
    std::string sha256;
};

struct NativeQualificationBundleBinding {
    std::string sha256;
    std::uint64_t size_bytes{0U};
    std::string embedded_config_sha256;
    std::string build_receipt_sha256;
    std::array<NativeQualificationPlanBinding, 6> plans;
};

struct NativeQualificationAccuracyEvidence {
    // Exact Q3-only receipt published before W3.
    std::string receipt_sha256;
    std::uint64_t receipt_size_bytes{0U};
    // Exact same-process regular receipt published after W3/N100/Q1. The
    // artifact-only generator verifies its Q3 linkage and post-Q1 gate before
    // emitting this record.
    std::string regular_receipt_sha256;
    std::uint64_t regular_receipt_size_bytes{0U};
    std::string mode;
    std::string policy_id;
    std::uint32_t replay_count{0U};
    std::uint32_t frames_per_replay{0U};
    bool reset_before_each_replay{false};
    bool all_semantic_gates_passed{false};
    bool timing_performed{true};
    std::string golden_manifest_sha256;
    std::string golden_masks_sha256;
    std::string benchmark_executable_sha256;
    std::string benchmark_source_manifest_sha256;
    std::string benchmark_source_closure_sha256;
};

struct NativeQualificationRecord {
    std::int32_t schema_version{kNativeQualificationRecordSchemaVersion};
    std::string artifact_type{std::string(kNativeQualificationRecordArtifactType)};
    std::string authority_id;
    std::uint64_t authority_serial{0U};
    bool self_authorizing{true};
    NativeQualificationScope scope;
    NativeQualificationBundleBinding bundle;
    NativeQualificationAccuracyEvidence accuracy_evidence;
    std::string generated_at_utc;
};

// Host-side canonical generator only. It does not publish a record and cannot
// add a production authority pin. Returned JSON is compact, key-sorted, and
// terminated by exactly one newline.
std::string makeCanonicalNativeQualificationRecord(const NativeQualificationRecord& record);

namespace qualification_internal {

struct AuthorizedNativeQualificationRecord {
    NativeQualificationRecord record;
    std::string record_sha256;
    std::uint64_t record_size_bytes{0U};
};

// Production pins are compiled into the implementation. The current registry
// is intentionally empty, so this function always fails closed today.
AuthorizedNativeQualificationRecord
authorizeProductionNativeQualificationRecord(const std::string& record_path);

#if defined(TRTMC_SAM2_TEST_QUALIFICATION_AUTHORITY)
// Compile-only test seam. This declaration and its loader entry point do not
// exist in production builds.
struct NativeQualificationTestPin {
    std::string authority_id;
    std::uint64_t minimum_authority_serial{0U};
    std::string record_sha256;
};

AuthorizedNativeQualificationRecord
authorizeNativeQualificationRecordForTest(const std::string& record_path,
                                          const NativeQualificationTestPin& pin);
#endif

} // namespace qualification_internal

} // namespace trtmc::sam2
