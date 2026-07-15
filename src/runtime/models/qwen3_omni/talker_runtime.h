/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct Qwen3OmniTalkerRuntimeResult {
    int exit_code{-1};
    std::vector<int32_t> frame_major_codes;
    std::string stderr_data;
};

Qwen3OmniTalkerRuntimeResult
run_qwen3_omni_official_talker(const std::string& hf_python, const std::string& model_id,
                               const std::string& model_revision, const std::string& prompt,
                               const std::string& assistant_text, int32_t n_codebooks,
                               int32_t max_frames);

} // namespace trtmc
