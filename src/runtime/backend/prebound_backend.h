/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_backend.h"

#include <cstddef>
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

} // namespace trtmc
