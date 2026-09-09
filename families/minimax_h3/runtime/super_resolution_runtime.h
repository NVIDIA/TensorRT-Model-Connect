/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::minimax_h3 {

struct SuperResolutionConfig {
    bool enabled{false};
    std::string section;
    std::string input_name;
    std::string output_name;
    int32_t source_height{0};
    int32_t source_width{0};
    int32_t target_height{0};
    int32_t target_width{0};
    int32_t batch_min{0};
    int32_t batch_opt{0};
    int32_t batch_max{0};
};

void validate_super_resolution_config(const SuperResolutionConfig& config);
void validate_super_resolution_plan(ITrtModule& module, const SuperResolutionConfig& config);

// Runs the frame-independent native TensorRT super-resolution plan in bounded
// batches. Both input and output are frame-major RGB (NHWC) FP32.
std::vector<float> run_super_resolution(ITrtModule& module, const std::vector<float>& pixels,
                                        int32_t num_frames, const SuperResolutionConfig& config);

} // namespace trtmc::minimax_h3
