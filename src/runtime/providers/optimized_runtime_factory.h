/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * Private integration contract between the Model Connect optimized-runtime
 * host and a model-owned adapter DSO.
 *
 * This is deliberately not an operation ABI. The DSO implements the existing
 * trtmc::IPipeline interface and therefore owns every model- and
 * modality-specific operation, request translation, and result translation.
 * The shared host only authenticates and materializes the capsule, loads its
 * exact DSO, and asks the DSO to construct a pipeline.
 *
 * All request pointers are borrowed for the duration of create(). No exception
 * may cross the exported entry point or create callback. A successful create
 * transfers ownership of the returned pipeline to Model Connect. The host
 * keeps the DSO loaded for the remainder of the process so that the pipeline's
 * code and vtable remain valid through destruction.
 */

#include "trtmc/pipeline.h"

#include <cstddef>
#include <cstdint>

namespace trtmc::internal {

inline constexpr std::uint32_t kOptimizedRuntimeFactoryAbiVersionV1 = 1U;
inline constexpr std::uint32_t kOptimizedRuntimePipelineAbiVersionV1 = 1U;
inline constexpr char kOptimizedRuntimeFactoryEntrypointV1[] =
    "trtmc_get_optimized_runtime_factory_v1";

// IPipeline passes C++ standard-library objects across the DSO boundary. Keep
// the versioned operation ABI above, and additionally require both sides to
// have been built with a compatible compiler and standard-library ABI.
struct OptimizedRuntimeToolchainAbiV1 {
    std::uint32_t compiler_family;
    std::uint32_t compiler_major_version;
    std::uint32_t cxx_abi_family;
    std::uint32_t cxx_abi_version;
    std::uint32_t cxx_language_standard;
    std::uint32_t standard_library_family;
    std::uint32_t standard_library_version;
    std::uint32_t standard_library_abi;
    std::uint32_t pointer_size;
    std::uint32_t string_size;
    std::uint32_t string_alignment;
};

#if defined(__clang__)
inline constexpr std::uint32_t kOptimizedRuntimeCompilerFamilyV1 = 2U;
inline constexpr std::uint32_t kOptimizedRuntimeCompilerMajorVersionV1 = __clang_major__;
#elif defined(__GNUC__)
inline constexpr std::uint32_t kOptimizedRuntimeCompilerFamilyV1 = 1U;
inline constexpr std::uint32_t kOptimizedRuntimeCompilerMajorVersionV1 = __GNUC__;
#elif defined(_MSC_VER)
inline constexpr std::uint32_t kOptimizedRuntimeCompilerFamilyV1 = 3U;
inline constexpr std::uint32_t kOptimizedRuntimeCompilerMajorVersionV1 = _MSC_VER;
#else
inline constexpr std::uint32_t kOptimizedRuntimeCompilerFamilyV1 = 0U;
inline constexpr std::uint32_t kOptimizedRuntimeCompilerMajorVersionV1 = 0U;
#endif

#if defined(_MSC_VER)
inline constexpr std::uint32_t kOptimizedRuntimeCxxAbiFamilyV1 = 2U;
inline constexpr std::uint32_t kOptimizedRuntimeCxxAbiVersionV1 = _MSC_VER;
#elif defined(__GXX_ABI_VERSION)
inline constexpr std::uint32_t kOptimizedRuntimeCxxAbiFamilyV1 = 1U;
inline constexpr std::uint32_t kOptimizedRuntimeCxxAbiVersionV1 = __GXX_ABI_VERSION;
#else
inline constexpr std::uint32_t kOptimizedRuntimeCxxAbiFamilyV1 = 0U;
inline constexpr std::uint32_t kOptimizedRuntimeCxxAbiVersionV1 = 0U;
#endif

#if defined(_LIBCPP_VERSION)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryFamilyV1 = 2U;
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryVersionV1 = _LIBCPP_VERSION;
#if defined(_LIBCPP_ABI_VERSION)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = _LIBCPP_ABI_VERSION;
#else
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = 0U;
#endif
#elif defined(__GLIBCXX__)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryFamilyV1 = 1U;
#if defined(_GLIBCXX_RELEASE)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryVersionV1 = _GLIBCXX_RELEASE;
#else
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryVersionV1 = 0U;
#endif
#if defined(_GLIBCXX_USE_CXX11_ABI)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = _GLIBCXX_USE_CXX11_ABI;
#else
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = 0U;
#endif
#elif defined(_MSVC_STL_VERSION)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryFamilyV1 = 3U;
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryVersionV1 = _MSVC_STL_VERSION;
#if defined(_ITERATOR_DEBUG_LEVEL)
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = _ITERATOR_DEBUG_LEVEL;
#else
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = 0U;
#endif
#else
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryFamilyV1 = 0U;
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryVersionV1 = 0U;
inline constexpr std::uint32_t kOptimizedRuntimeStandardLibraryAbiV1 = 0U;
#endif

inline constexpr OptimizedRuntimeToolchainAbiV1 kCurrentOptimizedRuntimeToolchainAbiV1 = {
    kOptimizedRuntimeCompilerFamilyV1,
    kOptimizedRuntimeCompilerMajorVersionV1,
    kOptimizedRuntimeCxxAbiFamilyV1,
    kOptimizedRuntimeCxxAbiVersionV1,
    static_cast<std::uint32_t>(__cplusplus),
    kOptimizedRuntimeStandardLibraryFamilyV1,
    kOptimizedRuntimeStandardLibraryVersionV1,
    kOptimizedRuntimeStandardLibraryAbiV1,
    static_cast<std::uint32_t>(sizeof(void*)),
    static_cast<std::uint32_t>(sizeof(std::string)),
    static_cast<std::uint32_t>(alignof(std::string)),
};

struct OptimizedRuntimePipelineCreateRequestV1 {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    const char* implementation_id;
    const char* model_id;
    const char* profile_id;
    const char* bundle_path;
    const char* artifact_path;
    const char* implementation_metadata;
    std::size_t implementation_metadata_size;
    const LoadOptions* load_options;
};

using CreateOptimizedRuntimePipelineV1 =
    IPipeline* (*)(const OptimizedRuntimePipelineCreateRequestV1* request, char* error_message,
                   std::size_t error_message_capacity) noexcept;

struct OptimizedRuntimeFactoryV1 {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    const char* implementation_id;
    const char* runtime_name;
    const char* runtime_version;
    const char* runtime_commit;
    CreateOptimizedRuntimePipelineV1 create;
    // Bump this explicit contract version only when IPipeline changes in a
    // binary-incompatible way. Source-only edits do not invalidate adapters.
    std::uint32_t pipeline_abi_version{kOptimizedRuntimePipelineAbiVersionV1};
    OptimizedRuntimeToolchainAbiV1 toolchain_abi{kCurrentOptimizedRuntimeToolchainAbiV1};

    // Optional process-global compatibility claim. A runtime that mutates a
    // process-global registry during create() names that registry and supplies
    // a deterministic fingerprint of the state it installs. Factories that
    // do not make such mutations leave both pointers null. The host allows
    // multiple factories to claim one namespace only when their fingerprints
    // are identical.
    //
    // These fields extend the end of the V1 table. A factory compiled against
    // the original V1 layout may report kOptimizedRuntimeFactoryV1BaseSize and
    // is treated as making no claim.
    const char* process_compatibility_namespace{nullptr};
    const char* process_compatibility_fingerprint{nullptr};
};

inline constexpr std::uint32_t kOptimizedRuntimeFactoryV1BaseSize = static_cast<std::uint32_t>(
    offsetof(OptimizedRuntimeFactoryV1, process_compatibility_namespace));

using GetOptimizedRuntimeFactoryV1 = const OptimizedRuntimeFactoryV1* (*)() noexcept;

} // namespace trtmc::internal
