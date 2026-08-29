/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/cosmos3/options.h"
#include "trtmc/pipeline.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        const auto options = trtmc::parse_cosmos3_options(
            R"({"negative_prompt":"blurry, distorted, low quality, jittery, deformed","num_inference_steps":35,"guidance_scale":6.0,"flow_shift":10.0,"video_height":720,"video_width":1280,"video_num_frames":189,"frame_rate":24,"text_seq_len":4096,"seed":42})");
        trtmc::GenerateConfig config;
        config.seed = 7;
        const auto request = trtmc::resolve_cosmos3_request(options, config);
        if (request.seed != 7 || request.video_num_frames != 189 || request.video_height != 720 ||
            request.video_width != 1280 || request.num_inference_steps != 35 ||
            request.text_seq_len != 4096 ||
            request.negative_prompt != "blurry, distorted, low quality, jittery, deformed") {
            throw std::runtime_error("Cosmos3 request resolution changed the fixed profile");
        }
        config.num_steps = 34;
        try {
            (void)trtmc::resolve_cosmos3_request(options, config);
            throw std::runtime_error("Cosmos3 accepted a reduced-quality step override");
        } catch (const std::invalid_argument&) {
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
