/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/primitives/trt_common.h"

#include <NvInfer.h>
#include <memory>
#include <string>

namespace trtmc {

class TrtLogger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
    const std::string& last_error() const;
    void clear_error();

  private:
    std::string mLastError;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr == nullptr)
            return;
        delete ptr;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime();

} // namespace trtmc
