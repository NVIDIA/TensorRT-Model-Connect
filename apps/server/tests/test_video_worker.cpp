/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cli/io.h"
#include "server/native_worker.h"
#include "trtmc/task.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

using Json = nlohmann::json;
class Video final : public trtmc::IImageGeneration {
  public:
    int calls{0};
    const char* task() const noexcept override { return kTask; }
    trtmc::ImageResult generate_image(const std::string& prompt,
                                      const trtmc::ImageGenerationConfig& config) override {
        if (prompt != "three frames" || config.seed != 42 || config.num_steps != 30)
            throw std::runtime_error("native request changed");
        ++calls;
        trtmc::ImageResult result;
        result.width = 8;
        result.height = 4;
        result.channels = 3;
        result.num_frames = 3;
        result.pixels.resize(8 * 4 * 3 * 3);
        for (int frame = 0; frame < 3; ++frame)
            for (int pixel = 0; pixel < 8 * 4; ++pixel)
                result.pixels[frame * 8 * 4 * 3 + pixel * 3 + frame] = 1.0F;
        return result;
    }
};
int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("trtmc-video-worker-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        Video task;
        const Json request{
            {"id", "clip"},
            {"op", "generate_video"},
            {"prompt", "three frames"},
            {"output_path", root.string()},
            {"config", {{"width", 8}, {"height", 4}, {"num_steps", 30}, {"seed", 42}}}};
        auto bad = request;
        bad["config"]["width"] = 16;
        std::stringstream input;
        input << bad.dump() << '\n' << request.dump() << '\n';
        std::ostringstream output;
        trtmc::server::run_task_worker(task, input, output, {8, 4, 3});
        std::istringstream lines(output.str());
        std::string line;
        std::getline(lines, line);
        const auto ready = Json::parse(line);
        std::getline(lines, line);
        const auto rejected = Json::parse(line);
        std::getline(lines, line);
        const auto result = Json::parse(line).at("result");
        if (ready["capabilities"] != Json::array({"video_generation"}) ||
            ready["video"]["num_frames"] != 3 || task.calls != 1 ||
            rejected["error"]["type"] != "invalid_request_error" || result["num_frames"] != 3 ||
            result["timing_scope"] != "public_task_call_wall")
            throw std::runtime_error("video capability, admission, or timing failed");
        for (int frame = 0; frame < 3; ++frame) {
            const auto image = trtmc::cli::io::read_image(
                (root / ("frame-00000" + std::to_string(frame) + ".png")).string());
            if (image.width != 8 || image.height != 4 || image.pixels.at(frame) != 1.0F)
                throw std::runtime_error("complete ordered frame codec failed");
        }
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
