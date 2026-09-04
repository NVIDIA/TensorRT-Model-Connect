/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/primitives/cuda_common.h"

#include <cuda_runtime_api.h>
#include <string>

namespace trtmc {

enum class TrtLogSeverity {
    kInternalError,
    kError,
    kWarning,
    kInfo,
    kVerbose,
};

const char* trt_severity_name(TrtLogSeverity severity);
bool trt_log_to_stderr_enabled();
TrtLogSeverity trt_log_stderr_min_severity();

// Configure the process-wide TRT logger. Called by pipeline_factory after
// resolving the platform.* namespace (replaces the old TRTMC_TRT_LOG_STDERR
// and TRTMC_TRT_LOG_MIN_SEVERITY env vars).
void configure_trt_logger(bool verbose_stderr, const std::string& min_severity);

// RAII wrapper for CUDA graph + executable graph.
// Captures a stream region and replays it without per-kernel launch overhead.
class CudaGraphExec final {
  public:
    CudaGraphExec() = default;
    ~CudaGraphExec();

    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;
    CudaGraphExec(CudaGraphExec&& other) noexcept;
    CudaGraphExec& operator=(CudaGraphExec&& other) noexcept;

    // Begin capturing on the given stream. All subsequent CUDA operations
    // on this stream will be recorded into the graph until end_capture().
    bool begin_capture(cudaStream_t stream);

    // End capture, instantiate the executable graph. Returns true on success.
    bool end_capture(cudaStream_t stream);

    // Launch the captured graph on the given stream.
    bool launch(cudaStream_t stream) const;

    // True if a graph has been successfully captured and instantiated.
    bool ready() const;

    // Reset — destroy captured graph and executable.
    void reset();

  private:
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t exec_{nullptr};
};

} // namespace trtmc
