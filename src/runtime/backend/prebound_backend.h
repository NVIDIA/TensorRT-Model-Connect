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

// Optional TensorRT-RTX capability for large staged plans. It remains separate
// from IPreboundBackend because file-backed deserialization has different
// lifetime and execution-context requirements.
class IFileBackedBackend {
  public:
    virtual ~IFileBackedBackend();
    virtual std::unique_ptr<ITrtModule>
    create_module_from_file(const char* plan_path, std::uint64_t plan_offset,
                            std::uint64_t plan_size, const ModuleCreateOptions& options,
                            const std::vector<ModuleExternalBinding>& external_bindings,
                            std::int64_t weight_streaming_budget_bytes, bool retain_engine,
                            bool serial_execution_context) = 0;
    // Releasing the final lease persists the process-shared JIT cache. These
    // explicit calls avoid serializing it from the Windows loader lock.
    virtual std::uint64_t acquire_runtime_cache_lease(const char* path) = 0;
    virtual void release_runtime_cache_lease(std::uint64_t lease) = 0;
};

} // namespace trtmc
