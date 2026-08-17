/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle_writer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::sam2::native {

inline constexpr std::uint64_t kDefaultSam2WorkspaceBytes = std::uint64_t{8} << 30U;
inline constexpr std::uint64_t kMaximumSam2ConfigBytes = std::uint64_t{1} << 20U;
inline constexpr std::string_view kSam2ModelId = "sam2.1-hiera-small-bbox";

class Sam2EngineBuildError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class Sam2GraphKind : std::uint8_t {
    kImage,
    kPrompt,
    kRecurrent,
};

struct Sam2RuntimeBuildFacts {
    std::string tensorrt_version;
    std::string tensorrt_abi;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    std::string gpu_name;
    std::int32_t gpu_device{0};
    std::int32_t gpu_compute_major{0};
    std::int32_t gpu_compute_minor{0};
    std::uint64_t gpu_global_memory_bytes{0};
    bool strongly_typed{false};
    bool tf32_enabled{true};
};

struct Sam2GraphBuildFacts {
    std::string section;
    Sam2GraphKind kind{Sam2GraphKind::kImage};
    std::int32_t history_frames{0};
    std::int32_t input_count{0};
    std::int32_t output_count{0};
    std::int32_t layer_count{0};
    std::size_t referenced_tensor_count{0};
    bool graph_complete{false};
    std::int32_t convolution_layer_count{-1};
    std::int32_t activation_layer_count{-1};
    std::int32_t pooling_layer_count{-1};
    std::int32_t element_wise_layer_count{-1};
    std::int32_t shuffle_layer_count{-1};
    std::int32_t constant_layer_count{-1};
    std::int32_t slice_layer_count{-1};
    std::int32_t resize_layer_count{-1};
    std::int32_t normalization_layer_count{-1};
    std::int32_t cast_layer_count{-1};
    std::int32_t matrix_multiply_layer_count{-1};
    std::int32_t softmax_layer_count{-1};
    std::int32_t plugin_v3_layer_count{-1};
    std::int32_t attention_input_layer_count{-1};
    std::int32_t attention_output_layer_count{-1};
};

struct Sam2SerializedPlan {
    Sam2GraphBuildFacts graph;
    std::vector<std::uint8_t> bytes;
};

struct Sam2CompilationResult {
    Sam2RuntimeBuildFacts runtime;
    std::string plan_profiling_verbosity;
    std::vector<Sam2SerializedPlan> plans;
};

struct Sam2EngineBuildOptions {
    std::filesystem::path checkpoint_path;
    std::filesystem::path source_config_path;
    std::filesystem::path output_path;
    std::uint64_t workspace_bytes{kDefaultSam2WorkspaceBytes};
    std::int32_t gpu_device{0};
    std::string created_at_utc;
};

struct Sam2NativeBundleBuildResult {
    BundlePublicationFacts bundle;
    std::string build_receipt_json;
    std::string build_receipt_sha256;
    std::array<std::string, 6> plan_sha256;
};

// The raw delivered YAML is authenticated by SHA-256 and is never parsed. All
// executable configuration is the audited, compiled C++ graph contract; this
// canonical JSON is the exact configuration embedded in the bundle.
std::string_view sam2EmbeddedConfigJson() noexcept;
void verifySam2SourceConfig(const std::filesystem::path& path);

void validateSam2EngineBuildOptions(const Sam2EngineBuildOptions& options);
void validateSam2RuntimeBuildFacts(const Sam2RuntimeBuildFacts& runtime);
void validateSam2Compilation(const Sam2CompilationResult& compilation);

// Canonical JSON: stable key ordering, no host-locale formatting, and no
// qualification override. Native plans remain ineligible until a separate
// exact-golden qualification step updates a later, qualified artifact.
std::string makeSam2BuildReceipt(const Sam2EngineBuildOptions& options,
                                 const Sam2CompilationResult& compilation);

// Production entry point. It uses CheckpointReader::open() and native TensorRT
// IBuilder/INetworkDefinition construction implemented in the companion
// translation unit. The destination must not already exist.
Sam2NativeBundleBuildResult buildSam2NativeBundle(const Sam2EngineBuildOptions& options);

namespace detail {

// Kept TensorRT-free so the fail-closed assembly contract can be tested on a
// CPU-only host. Production callers use buildSam2NativeBundle() above.
Sam2NativeBundleBuildResult writeCompiledSam2NativeBundle(const Sam2EngineBuildOptions& options,
                                                          const Sam2CompilationResult& compilation);

} // namespace detail

} // namespace trtmc::sam2::native
