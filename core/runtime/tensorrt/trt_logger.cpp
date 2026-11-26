/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/tensorrt/trt_logger.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

namespace trtmc {

namespace {

TrtLogSeverity from_trt_severity(nvinfer1::ILogger::Severity severity) {
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return TrtLogSeverity::kInternalError;
    case nvinfer1::ILogger::Severity::kERROR:
        return TrtLogSeverity::kError;
    case nvinfer1::ILogger::Severity::kWARNING:
        return TrtLogSeverity::kWarning;
    case nvinfer1::ILogger::Severity::kINFO:
        return TrtLogSeverity::kInfo;
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return TrtLogSeverity::kVerbose;
    }
    return TrtLogSeverity::kInfo;
}

} // namespace

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kERROR && msg != nullptr)
        mLastError = msg;

    if (msg == nullptr)
        return;

    const TrtLogSeverity generic = from_trt_severity(severity);
    if (trt_log_to_stderr_enabled() && severity <= Severity::kVERBOSE &&
        generic <= trt_log_stderr_min_severity()) {
        std::cerr << "TRT_LOG[" << trt_severity_name(generic) << "] " << msg << '\n';
    } else if (severity <= Severity::kWARNING) {
        std::cerr << "[trt] " << trt_severity_name(generic) << ": " << msg << '\n';
    }
}

const std::string& TrtLogger::last_error() const {
    return mLastError;
}

void TrtLogger::clear_error() {
    mLastError.clear();
}

TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime() {
    // When CUDA cannot initialize (no device, or no memory left for the context), TensorRT 11.4's
    // createInferRuntime logs an API usage error but returns a non-null, unusable pointer, so a
    // null check passes and the first deserialization crashes. Initialize CUDA first and report
    // its error instead. createInferRuntime creates the current device's context anyway.
    const cudaError_t status = cudaFree(nullptr);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("[trtmc] CUDA initialization failed: ") +
                                 cudaGetErrorName(status) + ": " + cudaGetErrorString(status));
    static TrtLogger logger;
    return TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
}

} // namespace trtmc
