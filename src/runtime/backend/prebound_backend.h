/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct ModuleExternalBinding {
    std::string tensor_name;
    void* device_ptr{nullptr};
    std::size_t capacity_bytes{0};
};

// Internal standard-TRT capability used by staged model runtimes. Keeping it
// separate from IBackend preserves the installed backend interface and ABI.
class IPreboundBackend {
  public:
    virtual ~IPreboundBackend();
    virtual std::unique_ptr<ITrtModule>
    create_module_prebound(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<ModuleExternalBinding>& external_bindings) = 0;
};

// Optional TensorRT-RTX capability for large staged plans. This is a sibling
// interface so extending file-backed behavior never changes IPreboundBackend's
// established cross-DSO vtable.
class IFileBackedBackend {
  public:
    virtual ~IFileBackedBackend();
    virtual std::unique_ptr<ITrtModule>
    create_module_from_file(const char* plan_path, std::uint64_t plan_offset,
                            std::uint64_t plan_size, const char* expected_sha256,
                            const ModuleCreateOptions& options,
                            const std::vector<ModuleExternalBinding>& external_bindings,
                            std::int64_t weight_streaming_budget_bytes, bool retain_engine) = 0;
};

// Optional capability for backends that own a process-shared JIT runtime
// cache. Windows deliberately keeps backend DSOs alive until process exit, so
// callers need an explicit, loader-lock-free persistence point.
class IRuntimeCacheBackend {
  public:
    virtual ~IRuntimeCacheBackend();
    virtual std::uint64_t acquire_runtime_cache_lease(const char* path) = 0;
    // Releasing the final lease persists the cache. It may throw so an
    // explicit pipeline finalization can report serialization or I/O failure.
    virtual void release_runtime_cache_lease(std::uint64_t lease) = 0;
};

} // namespace trtmc
