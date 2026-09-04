/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/cosmos3/runtime/runtime_config.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr const char* kRuntime =
    R"({"negative_prompt":"blurry, distorted","num_inference_steps":35,"guidance_scale":6.0,"flow_shift":10.0,"seed":42,"video_height":720,"video_width":1280,"video_num_frames":189,"frame_rate":24,"text_seq_len":4096,"context_parallel_size":2})";

bool parse_throws(const std::string& text) {
    try {
        (void)trtmc::cosmos3::parse_runtime_config(text);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

} // namespace

int main() {
    try {
        const auto runtime = trtmc::cosmos3::parse_runtime_config(kRuntime);
        if (runtime.context_parallel_size != 2 || runtime.seed != 42)
            throw std::runtime_error("Cosmos3 runtime values were not retained");

        trtmc::ImageGenerationConfig request;
        request.seed = 7;
        const auto resolved = trtmc::cosmos3::resolve_request(runtime, request);
        if (resolved.seed != 7 || resolved.num_inference_steps != 35 ||
            resolved.negative_prompt != "blurry, distorted") {
            throw std::runtime_error("Cosmos3 request resolution changed the fixed profile");
        }

        request.num_steps = 34;
        try {
            (void)trtmc::cosmos3::resolve_request(runtime, request);
            throw std::runtime_error("Cosmos3 accepted a different step count");
        } catch (const std::invalid_argument&) {
        }

        if (!parse_throws(
                R"({"negative_prompt":"x","num_inference_steps":35,"guidance_scale":6.0,"flow_shift":10.0,"seed":42,"video_height":720,"video_width":1280,"video_num_frames":189,"frame_rate":24,"text_seq_len":4096,"context_parallel_size":4})")) {
            throw std::runtime_error("Cosmos3 accepted unsupported CP4");
        }
        if (!parse_throws(
                R"({"negative_prompt":"x","num_inference_steps":35,"guidance_scale":6.0,"flow_shift":10.0,"seed":42,"video_height":720,"video_width":1280,"video_num_frames":189,"frame_rate":24,"text_seq_len":4096,"context_parallel_size":1,"legacy_mode":"single"})")) {
            throw std::runtime_error("Cosmos3 accepted a legacy runtime field");
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
