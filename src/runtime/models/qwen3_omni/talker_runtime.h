/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct Qwen3OmniTalkerRuntimeResult {
    int exit_code{-1};
    std::vector<int32_t> frame_major_codes;
    std::string stderr_data;
    double worker_start_ms{0.0};
    double talker_ms{0.0};
    double ipc_ms{0.0};
    double output_materialization_ms{0.0};
};

struct Qwen3OmniTalkerRuntimeStats {
    int64_t worker_starts{0};
    int64_t requests{0};
    bool worker_running{false};
};

class Qwen3OmniTalkerRuntime {
  public:
    Qwen3OmniTalkerRuntime(std::string hf_python, std::string model_id, std::string model_revision,
                           int32_t n_codebooks, int32_t max_frames,
                           std::vector<std::string> worker_argv = {});
    ~Qwen3OmniTalkerRuntime();

    Qwen3OmniTalkerRuntime(Qwen3OmniTalkerRuntime&&) noexcept;
    Qwen3OmniTalkerRuntime& operator=(Qwen3OmniTalkerRuntime&&) noexcept;
    Qwen3OmniTalkerRuntime(const Qwen3OmniTalkerRuntime&) = delete;
    Qwen3OmniTalkerRuntime& operator=(const Qwen3OmniTalkerRuntime&) = delete;

    // Load the official Talker worker and wait until its CUDA model is resident.
    // Idempotent: run() reuses the ready worker without another model load.
    void start();
    Qwen3OmniTalkerRuntimeResult run(const std::string& prompt, const std::string& assistant_text);
    void shutdown();
    Qwen3OmniTalkerRuntimeStats stats() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace trtmc
