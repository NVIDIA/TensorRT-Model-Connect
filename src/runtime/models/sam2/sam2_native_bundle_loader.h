/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_native_video_processor.h"
#include "runtime/models/sam2/sam2_qualification_authority.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc::sam2 {

class NativeBundleLoadError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Values observed from the backend that will execute the serialized plans.
// Loading is restricted to the exact L4 qualification target and exact
// TensorRT build recorded in the bundle.
struct NativeBundleRuntimeTarget {
    std::string tensorrt_version;
    std::string tensorrt_abi;
    std::string gpu_name;
    std::string compute_capability;
};

// The factory must synchronously deserialize one owned plan view. The view is
// valid for the duration of the call. All bundle evidence is authenticated
// before the first factory invocation.
using NativePlanModuleFactory = std::function<std::unique_ptr<ITrtModule>(
    std::string_view section, const void* plan_data, std::size_t plan_size)>;

// Diagnostic-only unqualified path. It authenticates every internal bundle
// fact but never admits an artifact to production.
NativeVideoEngineSet
loadDiagnosticNativeVideoEngineSetFromBundle(const std::string& bundle_path,
                                             const NativeBundleRuntimeTarget& runtime_target,
                                             const NativePlanModuleFactory& module_factory);

// Diagnostic/build-time path with an additional exact full-bundle digest.
NativeVideoEngineSet loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
    const std::string& bundle_path, std::string_view expected_bundle_sha256,
    const NativeBundleRuntimeTarget& runtime_target, const NativePlanModuleFactory& module_factory);

// Production admission requires a separately stored, canonical qualification
// record whose exact SHA-256 is present in the compiled authority registry.
// The current registry is empty, so this API fails before bundle
// deserialization. The bundle itself must remain exactly unqualified.
NativeVideoEngineSet loadProductionQualifiedNativeVideoEngineSetFromBundle(
    const std::string& bundle_path, const std::string& qualification_record_path,
    const NativeBundleRuntimeTarget& runtime_target, const NativePlanModuleFactory& module_factory);

#if defined(TRTMC_SAM2_TEST_QUALIFICATION_AUTHORITY)
// Compile-only seam for CPU tests of the post-pin production path.
NativeVideoEngineSet loadProductionQualifiedNativeVideoEngineSetFromBundleForTest(
    const std::string& bundle_path, const std::string& qualification_record_path,
    const NativeBundleRuntimeTarget& runtime_target, const NativePlanModuleFactory& module_factory,
    const qualification_internal::NativeQualificationTestPin& pin);
#endif

} // namespace trtmc::sam2
