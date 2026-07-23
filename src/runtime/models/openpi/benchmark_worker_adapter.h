/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/openpi/api.h"
#include "trtmc/pipeline.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::openpi::benchmark {

inline nlohmann::json run_predict_actions(IPipeline& pipeline, const nlohmann::json& request,
                                          int warmup, int iterations) {
    constexpr std::size_t kPixelsPerCamera = 224U * 224U * 3U;
    constexpr std::array<const char*, 3> kCameraNames = {"base_0_rgb", "left_wrist_0_rgb",
                                                         "right_wrist_0_rgb"};
    constexpr std::array<bool, 3> kCameraValidity = {true, true, false};

    ActionRequest input;
    input.prompt = request.at("prompt").get<std::string>();
    input.seed = request.at("seed").get<int32_t>();
    input.denoise_steps = 10;
    input.state.assign(8U, 0.0F);
    input.initial_noise.assign(15U * 32U, 0.0F);
    for (std::size_t index = 0; index < kCameraNames.size(); ++index) {
        RobotImage camera;
        camera.name = kCameraNames[index];
        camera.height = 224;
        camera.width = 224;
        camera.channels = 3;
        camera.valid = kCameraValidity[index];
        camera.pixels.assign(kPixelsPerCamera, camera.valid ? 0.5F : 0.0F);
        input.cameras.push_back(std::move(camera));
    }

    auto* action_pipeline = dynamic_cast<IOpenPIActionPipeline*>(&pipeline);
    if (action_pipeline == nullptr) {
        throw std::runtime_error(std::string(pipeline.pipeline_type()) +
                                 " does not support predict_actions");
    }
    ActionResult last;
    for (int index = 0; index < warmup; ++index) {
        last = action_pipeline->predict_actions(input);
    }
    nlohmann::json observations = nlohmann::json::array();
    using Clock = std::chrono::steady_clock;
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        last = action_pipeline->predict_actions(input);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        observations.push_back({
            {"iteration", index},
            {"runtime_e2e_wall_ms", wall_ms},
            {"action_chunks", 1},
            {"action_steps", last.horizon},
            {"preprocess_ms", last.timings.preprocess_ms},
            {"prefill_ms", last.timings.prefill_ms},
            {"denoise_ms", last.timings.denoise_ms},
            {"postprocess_ms", last.timings.postprocess_ms},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"horizon", last.horizon},
             {"action_dim", last.action_dim},
             {"element_count", last.actions.size()},
         }},
    };
}

} // namespace trtmc::openpi::benchmark
