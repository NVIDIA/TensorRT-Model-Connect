/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if TRTMC_HAS_TVM_FFI

#include <memory>
#include <string_view>

namespace trtmc {

class TvmFfiBoundFunction {
  public:
    explicit TvmFfiBoundFunction(void* function) noexcept : function_(function) {}
    ~TvmFfiBoundFunction();

    TvmFfiBoundFunction(const TvmFfiBoundFunction&) = delete;
    TvmFfiBoundFunction& operator=(const TvmFfiBoundFunction&) = delete;

    void* handle() const noexcept { return function_; }

  private:
    void* function_{nullptr};
};

using TvmFfiBoundFunctionPtr = std::shared_ptr<const TvmFfiBoundFunction>;

TvmFfiBoundFunctionPtr resolve_global_tvm_ffi_function(std::string_view name) noexcept;

} // namespace trtmc

#endif
