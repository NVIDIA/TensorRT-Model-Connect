/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Minimal forward declarations for TRT-RTX APIs not in standard NvInfer.h.
// These types exist in the RTX shared library's vtable — we just need
// declarations so the compiler can generate the correct virtual calls.
//
// This header is ONLY included by rtx_backend.cpp (compiled into the RTX DSO).

#include <NvInfer.h>

namespace nvinfer1 {

//! Strategy for CUDA graph capture.
enum class CudaGraphStrategy : int32_t {
    kWHOLE_GRAPH_CAPTURE = 0, //!< Capture the entire inference graph.
};

//! Strategy for dynamic shape kernel specialization.
enum class DynamicShapesKernelSpecializationStrategy : int32_t {
    kLAZY = 0,  //!< Async background JIT (default).
    kEAGER = 1, //!< Blocking JIT.
    kNONE = 2,  //!< Fallback kernels only.
};

//! Opaque runtime cache for JIT-compiled kernels.
class IRuntimeCache {
  public:
    virtual ~IRuntimeCache() noexcept = default;

    //! Deserialize a previously saved cache.
    virtual bool deserialize(const void* data, size_t size) noexcept = 0;

    //! Serialize the cache to a host memory blob. Caller deletes the result.
    virtual IHostMemory* serialize() noexcept = 0;
};

} // namespace nvinfer1

// Extension methods on IRuntimeConfig that exist in TRT-RTX but not standard TRT.
// We access them via the apiv pimpl — the RTX library's VRuntimeConfig vtable
// includes these additional methods.
//
// IMPORTANT: These are NOT safe to call when linked against standard libnvinfer.
// They are ONLY valid when linked against libtensorrt_rtx.

namespace trtmc {
namespace rtx_ext {

// These free functions cast and call through the pimpl.  The RTX IRuntimeConfig
// has the same base layout as standard TRT (it extends VRuntimeConfig), so the
// pimpl pointer is valid.  We use the Python bindings API signatures as reference.

// Note: In practice, the RTX SDK ships its own NvInfer.h with these methods
// declared directly on IRuntimeConfig.  When the RTX headers become available,
// replace this shim with a direct #include of the RTX NvInfer.h.

} // namespace rtx_ext
} // namespace trtmc
