/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// QwenDebugWindows: optional float32 decoder outputs named debug_hidden_<i>.
//
// The family builder can mark the residual stream after selected decoder
// layers as engine outputs. The runtime discovers them by name and binds each
// one to a device buffer owned here; the TensorRT module's forward() skips
// externally bound outputs, so they are never copied to host on the generation
// path. Values reach the host only through append_rows().

#include "trtmc/runtime/device_tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;

class QwenDebugWindows {
  public:
    struct Window {
        std::string name;
        int32_t layer{0};
        int64_t hidden{0};
        DeviceTensor buffer; // [rows, hidden] float32, bound as the engine output
    };

    // Bind every debug_hidden_<i> output of the module to an owned buffer of the
    // module's maximum output shape, which must hold at least min_rows rows.
    // Engines without debug outputs bind nothing.
    void bind(ITrtModule& module, int64_t min_rows);

    bool empty() const { return windows_.empty(); }
    const std::vector<Window>& windows() const { return windows_; }

    // Append the first `rows` rows of windows()[index] to `out` (synchronous D2H
    // on the module stream).
    void append_rows(std::size_t index, int64_t rows, std::vector<float>& out) const;

  private:
    std::vector<Window> windows_; // sorted by layer
    cudaStream_t stream_{nullptr};
};

} // namespace trtmc
