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

    // Optional TensorRT-RTX path for large staged plans. The caller supplies a
    // validated byte range and digest inside a local bundle so the backend can
    // verify and deserialize without copying the complete plan into host memory.
    virtual std::unique_ptr<ITrtModule>
    create_module_from_file(const char* plan_path, std::uint64_t plan_offset,
                            std::uint64_t plan_size, const char* expected_sha256,
                            const ModuleCreateOptions& options,
                            const std::vector<ModuleExternalBinding>& external_bindings,
                            std::int64_t weight_streaming_budget_bytes) {
        (void)plan_path;
        (void)plan_offset;
        (void)plan_size;
        (void)expected_sha256;
        (void)options;
        (void)external_bindings;
        (void)weight_streaming_budget_bytes;
        return nullptr;
    }
};

} // namespace trtmc
