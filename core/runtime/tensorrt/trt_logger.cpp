/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/tensorrt/trt_logger.h"

#include <iostream>

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
    static TrtLogger logger;
    return TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
}

} // namespace trtmc
