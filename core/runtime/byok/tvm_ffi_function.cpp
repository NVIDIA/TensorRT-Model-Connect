/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/byok/tvm_ffi_function.h"

#if TRTMC_HAS_TVM_FFI

#include <cstdint>
#include <tvm/ffi/c_api.h>

namespace trtmc {

TvmFfiBoundFunction::~TvmFfiBoundFunction() {
    if (function_ != nullptr)
        TVMFFIObjectDecRef(function_);
}

TvmFfiBoundFunctionPtr resolve_global_tvm_ffi_function(std::string_view name) noexcept {
    TVMFFIByteArray key{name.data(), name.size()};
    TVMFFIObjectHandle function = nullptr;
    if (TVMFFIFunctionGetGlobal(&key, &function) != 0 || function == nullptr)
        return nullptr;
    return std::make_shared<const TvmFfiBoundFunction>(function);
}

} // namespace trtmc

#endif
