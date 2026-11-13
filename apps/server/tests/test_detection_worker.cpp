/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cli/io.h"
#include "server/native_worker.h"
#include "trtmc/task.h"

#include <filesystem>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
class Detector final : public trtmc::IObjectDetection {
  public:
    int calls{0};
    bool invalid{false}, empty{false};
    trtmc::ObjectDetectionResult detect(const float* pixels, std::int32_t height,
                                        std::int32_t width) override {
        require(height == 4 && width == 8 && pixels[0] == 1.0F && pixels[1] == 0.0F,
                "original dimensions and decoded RGB pixels reach the public Task");
        ++calls;
        trtmc::ObjectDetectionResult result;
        result.image_height = height;
        result.image_width = width;
        if (!empty)
            result.boxes = {{-2.0F, 1.0F, 10.0F, 4.0F,
                             invalid ? std::numeric_limits<float>::quiet_NaN() : 0.75F, 42}};
        return result;
    }
};
std::vector<Json> messages(const std::string& value) {
    std::vector<Json> result;
    std::istringstream input(value);
    for (std::string line; std::getline(input, line);)
        result.push_back(Json::parse(line));
    return result;
}
void verify(const std::string& path) {
    Detector task;
    std::stringstream input;
    input << Json{{"id", "bad"},
                  {"op", "detect"},
                  {"image_path", path},
                  {"config", {{"threshold", .5}}}}
                 .dump()
          << '\n';
    input << Json{{"id", "image"}, {"op", "detect"}, {"image_path", path}}.dump() << '\n';
    std::ostringstream output;
    require(trtmc::server::run_task_worker(task, input, output) == 0, "worker completes");
    const auto replies = messages(output.str());
    require(task.calls == 1 && replies.size() == 3 &&
                replies[0].at("capabilities") == Json::array({"object_detection"}),
            "declared Task is advertised");
    require(replies[1].at("error").at("type") == "invalid_request_error",
            "unsupported controls never reach the Task");
    const auto result = replies[2].at("result");
    require(replies[2].at("id") == "image" && result.at("image_width") == 8 &&
                result.at("image_height") == 4 && result.at("boxes")[0].at("x_min") == -2 &&
                result.at("boxes")[0].at("x_max") == 10 &&
                result.at("boxes")[0].at("class_id") == 42 &&
                result.at("boxes")[0].at("confidence") == .75 &&
                result.at("model_call_ms").get<double>() >= 0 &&
                result.at("timing_scope") == "public_task_call_wall",
            "native boxes, labels, confidence and Task timing are preserved without clipping");
    for (bool empty : {true, false}) {
        task.empty = empty;
        task.invalid = !empty;
        std::istringstream request(
            Json{{"id", "next"}, {"op", "detect"}, {"image_path", path}}.dump() + "\n");
        std::ostringstream reply;
        const auto code = trtmc::server::run_task_worker(task, request, reply);
        const auto response = messages(reply.str())[1];
        require(empty ? code == 0 && response.at("result").at("boxes").empty()
                      : code == 1 &&
                            response.at("error").at("message") == "native worker operation failed",
                "empty detections succeed; invalid native results retire and redact");
    }
}
} // namespace
int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("trtmc-detection-" + std::to_string(std::random_device{}()) + ".png");
    try {
        std::vector<float> pixels(8 * 4 * 3, 0.0F);
        for (std::size_t i = 0; i < pixels.size(); i += 3)
            pixels[i] = 1.0F;
        trtmc::cli::io::save_png(path.string(), pixels, 8, 4);
        verify(path.string());
        std::filesystem::remove(path);
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
