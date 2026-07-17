/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "bundle/bundle_format.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr const char* kSam3TrackerStepRuntimeManifestSection =
    "sam3_tracker_step_runtime_manifest.json";
inline constexpr const char* kSam3TrackerStepNativePluginSection =
    "sam3_tracker_step_native_plugin_so";
inline constexpr const char* kSam3TrackerStepScope = "meta_split_dynamic_encoder_static_decoder";
inline constexpr const char* kSam3TrackerMemoryAotiManifestSection =
    "sam3_tracker_memory_aoti_manifest.json";
inline constexpr const char* kSam3TrackerMemoryScope = "fixed_memory_encoder_soft_hard_b1_b2";
inline constexpr const char* kSam3HardMaskResizeAotiManifestSection =
    "sam3_hard_mask_resize_aoti_manifest.json";
inline constexpr const char* kSam3HardMaskResizeScope = "torch_bilinear_288_to_1008_b1_b2";

struct Sam3TrackerStepPackageSpec {
    std::string stage;
    std::string package_global;
    std::string section;
    std::string sha256;
    int32_t batch_size{0};
};

struct Sam3TrackerStepPipelineSpec {
    std::string global_name;
    std::string encoder_sha256;
    std::string decoder_sha256;
    int32_t batch_size{0};
};

struct Sam3TrackerStepRuntimeManifest {
    int32_t schema_version{0};
    std::string step_scope;
    std::string plugin_section;
    std::string plugin_sha256;
    std::string plugin_type;
    std::string plugin_version;
    std::string torch_version;
    std::string transformers_version;
    std::string tvm_ffi_version;
    std::string tensorrt_version;
    std::string cuda_version;
    std::string host_architecture;
    bool torch_cxx11_abi{false};
    uint64_t aoti_abi_version{0};
    int32_t compute_capability_major{0};
    int32_t compute_capability_minor{0};
    std::array<Sam3TrackerStepPackageSpec, 4> packages;
    std::array<Sam3TrackerStepPipelineSpec, 2> pipelines;
};

struct Sam3TrackerMemoryPackageSpec {
    std::string policy;
    std::string package_global;
    std::string section;
    std::string sha256;
    int32_t batch_size{0};
    bool hard_mask{false};
};

struct Sam3TrackerMemoryAotiManifest {
    int32_t schema_version{0};
    std::string scope;
    std::string artifact_format;
    std::string model_sha256;
    std::string exporter_sha256;
    std::string torch_version;
    std::string transformers_version;
    std::string cuda_version;
    std::string host_architecture;
    bool torch_cxx11_abi{false};
    uint64_t aoti_abi_version{0};
    int32_t compute_capability_major{0};
    int32_t compute_capability_minor{0};
    std::array<Sam3TrackerMemoryPackageSpec, 4> packages;
};

struct Sam3HardMaskResizePackageSpec {
    std::string package_global;
    std::string section;
    std::string sha256;
    int32_t batch_size{0};
};

struct Sam3HardMaskResizeAotiManifest {
    int32_t schema_version{0};
    std::string scope;
    std::string artifact_format;
    std::string exporter_sha256;
    std::string torch_version;
    std::string transformers_version;
    std::string cuda_version;
    std::string host_architecture;
    bool torch_cxx11_abi{false};
    uint64_t aoti_abi_version{0};
    int32_t compute_capability_major{0};
    int32_t compute_capability_minor{0};
    std::array<Sam3HardMaskResizePackageSpec, 2> packages;
};

// Parse and fully validate the content contract, including every artifact
// SHA-256. This function is side-effect-free and is used by CPU unit tests.
Sam3TrackerStepRuntimeManifest
validate_sam3_tracker_step_runtime_manifest(const BundleFile& bundle);

// Parse the separate memory-exporter manifest, validate the exact fixed tensor
// contracts and four soft/hard B1/B2 content-addressed packages, then bind its
// producer ABI and target architecture to the tracker-step manifest.
Sam3TrackerMemoryAotiManifest
validate_sam3_tracker_memory_aoti_manifest(const BundleFile& bundle,
                                           const Sam3TrackerStepRuntimeManifest& step_manifest);

Sam3HardMaskResizeAotiManifest
validate_sam3_hard_mask_resize_aoti_manifest(const BundleFile& bundle,
                                             const Sam3TrackerStepRuntimeManifest& step_manifest);

// Extract, ABI-check, load, and register the model-owned native plugin and
// all ten AOTI packages, register the memory and resize functions, and register
// both split pipelines. Must run before TensorRT deserializes any step or
// memory wrapper plan.
// Repeated loads of the same content-addressed manifest are idempotent.
void load_sam3_tracker_step_runtime(const BundleFile& bundle);

// Exposed for deterministic bundle construction and known-answer unit tests.
std::string sam3_tracker_step_sha256_hex(const std::vector<char>& data);

} // namespace trtmc
