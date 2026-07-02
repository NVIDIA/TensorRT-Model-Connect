/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// trt_common.cpp — TensorRT logger/runtime helpers plus CUDA graph support.
// CudaStream/CudaBuffer moved to cuda_common.cpp.

#include "runtime/core/trt_common.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace trtmc {

const char* trt_severity_name(TrtLogSeverity severity) {
    switch (severity) {
    case TrtLogSeverity::kInternalError:
        return "INTERNAL_ERROR";
    case TrtLogSeverity::kError:
        return "ERROR";
    case TrtLogSeverity::kWarning:
        return "WARNING";
    case TrtLogSeverity::kInfo:
        return "INFO";
    case TrtLogSeverity::kVerbose:
        return "VERBOSE";
    }
    return "UNKNOWN";
}

// Process-wide TRT-logger state. Populated by configure_trt_logger() after
// pipeline_factory resolves a ConfigBundle that has the platform.* namespace.
// Defaults: verbose stderr stream is off; warnings and errors always
// surface via the short "[trt] ..." path in TrtLogger::log.
namespace {
struct TrtLogState {
    bool verbose_stderr_enabled{false};
    TrtLogSeverity verbose_min_severity{TrtLogSeverity::kInfo};
};
TrtLogState& mutable_trt_log_state() {
    static TrtLogState state;
    return state;
}
} // namespace

bool trt_log_to_stderr_enabled() {
    return mutable_trt_log_state().verbose_stderr_enabled;
}

TrtLogSeverity trt_log_stderr_min_severity() {
    return mutable_trt_log_state().verbose_min_severity;
}

// Public setter used by the pipeline factory after resolving the runtime
// config. Replaces the old TRTMC_TRT_LOG_{STDERR,MIN_SEVERITY} env vars.
// Severity values: INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE.
void configure_trt_logger(bool verbose_stderr, const std::string& min_severity) {
    TrtLogState& state = mutable_trt_log_state();
    state.verbose_stderr_enabled = verbose_stderr;
    std::string upper(min_severity);
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (upper == "INTERNAL_ERROR")
        state.verbose_min_severity = TrtLogSeverity::kInternalError;
    else if (upper == "ERROR")
        state.verbose_min_severity = TrtLogSeverity::kError;
    else if (upper == "WARNING")
        state.verbose_min_severity = TrtLogSeverity::kWarning;
    else if (upper == "VERBOSE")
        state.verbose_min_severity = TrtLogSeverity::kVerbose;
    else
        state.verbose_min_severity = TrtLogSeverity::kInfo;
}

CudaGraphExec::~CudaGraphExec() {
    reset();
}

CudaGraphExec::CudaGraphExec(CudaGraphExec&& other) noexcept
    : graph_(other.graph_), exec_(other.exec_) {
    other.graph_ = nullptr;
    other.exec_ = nullptr;
}

CudaGraphExec& CudaGraphExec::operator=(CudaGraphExec&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = other.graph_;
        exec_ = other.exec_;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
    }
    return *this;
}

bool CudaGraphExec::begin_capture(cudaStream_t stream) {
    reset();
    auto err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_graph] begin_capture failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    return true;
}

bool CudaGraphExec::end_capture(cudaStream_t stream) {
    auto err = cudaStreamEndCapture(stream, &graph_);
    if (err != cudaSuccess || graph_ == nullptr) {
        std::cerr << "[cuda_graph] end_capture failed: "
                  << (err != cudaSuccess ? cudaGetErrorString(err) : "null graph") << '\n';
        graph_ = nullptr;
        return false;
    }

    err = cudaGraphInstantiate(&exec_, graph_, 0);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_graph] instantiate failed: " << cudaGetErrorString(err) << '\n';
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        return false;
    }
    return true;
}

bool CudaGraphExec::launch(cudaStream_t stream) const {
    if (exec_ == nullptr) {
        return false;
    }
    return cudaGraphLaunch(exec_, stream) == cudaSuccess;
}

bool CudaGraphExec::ready() const {
    return exec_ != nullptr;
}

void CudaGraphExec::reset() {
    if (exec_ != nullptr) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

} // namespace trtmc
