/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::cli {

// Private hand-off from the native `trtmc build` launcher to the Python
// builder. This is deliberately not a public CLI option: the native launcher
// resolves a product-owned executable and overwrites any inherited value.
inline constexpr char kInternalDynamicMemoryCalibratorEnv[] =
    "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR";
inline constexpr char kInternalDynamicMemoryCalibratorBuildIdentityEnv[] =
    "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_BUILD_IDENTITY";
inline constexpr char kInternalDynamicMemoryCalibratorName[] =
    "trtmc_dynamic_memory_qualify";
inline constexpr char kInternalDynamicMemoryCalibratorPackageDirectory[] =
    ".trtmc-internal";

const std::string& internal_dynamic_memory_calibrator_product_version();
const std::string& internal_dynamic_memory_calibrator_build_identity();
const std::string& internal_dynamic_memory_calibrator_identity_marker();

std::vector<std::filesystem::path>
internal_dynamic_memory_calibrator_candidates(const std::filesystem::path& trtmc_executable);

std::optional<std::filesystem::path>
find_internal_dynamic_memory_calibrator(const std::filesystem::path& trtmc_executable);

// Clears inherited private hand-off values before resolving the product-owned
// helper. A missing or build-incompatible helper is intentionally non-fatal:
// ordinary builds continue with both values unset, while the dynamic
// unknown-plan calibration path fails closed if it actually needs the helper.
// Returns false only when the process environment could not be updated.
bool configure_internal_dynamic_memory_calibrator(
    const std::filesystem::path& trtmc_executable, std::string& error);

} // namespace trtmc::cli
