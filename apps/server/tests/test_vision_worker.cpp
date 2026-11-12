/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cli/io.h"
#include "server/native_worker.h"
#include "trtmc/task.h"

#include <filesystem>
#include <iostream>
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
class Vision final : public trtmc::IVisionLanguageGeneration, public trtmc::ITextGeneration {
  public:
    int calls{0};
    const char* task() const noexcept override { return IVisionLanguageGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return 10; }
    trtmc::TextResult generate(const std::string&, const trtmc::TextGenerationConfig&) override {
        throw std::runtime_error("image request must never take the text-only interface");
    }
    trtmc::TextResult generate(const std::string& prompt, const float* pixels, std::int32_t height,
                               std::int32_t width,
                               const trtmc::TextGenerationConfig& config) override {
        require(prompt == "Color?" && height == 4 && width == 8 && pixels[0] == 1.0F &&
                    pixels[1] == 0.0F,
                "original RGB geometry and question reach the Task");
        require(config.max_new_tokens == 10 && config.top_k == 1 && config.seed == 42,
                "generation controls reach the owning family");
        ++calls;
        trtmc::TextResult result;
        result.text = "Red. Complete answer.";
        result.token_ids = {1, 2, 3};
        return result;
    }
};
void verify(const std::string& path) {
    Vision task;
    const Json valid{{"id", "vqa"},
                     {"op", "generate_vision"},
                     {"prompt", "Color?"},
                     {"image_path", path},
                     {"config", {{"max_new_tokens", 10}, {"top_k", 1}, {"seed", 42}}}};
    std::stringstream input;
    auto missing = valid;
    missing.erase("image_path");
    input << missing.dump() << '\n';
    auto unreadable = valid;
    unreadable["id"] = "unreadable-image";
    unreadable["image_path"] = path + "/missing.png";
    input << unreadable.dump() << '\n';
    auto bad_config = valid;
    bad_config["config"]["unknown"] = true;
    input << bad_config.dump() << '\n';
    input << Json{{"id", "text"}, {"op", "generate"}, {"prompt", "Color?"}}.dump() << '\n';
    input << valid.dump() << '\n';
    auto default_config = valid;
    default_config["id"] = "default-tokens";
    default_config["config"].erase("max_new_tokens");
    input << default_config.dump() << '\n';
    std::ostringstream output;
    require(trtmc::server::run_task_worker(task, input, output) == 0, "worker completes");
    std::vector<Json> replies;
    std::istringstream stream(output.str());
    for (std::string line; std::getline(stream, line);)
        replies.push_back(Json::parse(line));
    require(replies.size() == 7 && task.calls == 2 &&
                replies[0].at("capabilities") == Json::array({"vision_language_generation"}) &&
                replies[0].at("default_max_new_tokens") == 10,
            "declared bundle capability prevents inherited text dispatch");
    for (std::size_t i = 1; i < 5; ++i)
        require(replies[i].at("error").at("type") == "invalid_request_error",
                "invalid requests remain recoverable and never invoke the Task");
    const auto result = replies[5].at("result");
    require(result.at("text") == "Red. Complete answer." && result.at("completion_tokens") == 3 &&
                result.at("image_height") == 4 && result.at("image_width") == 8 &&
                result.at("model_call_ms").get<double>() >= 0 &&
                result.at("timing_scope") == "public_task_call_wall",
            "complete result and timing survive");
}
} // namespace
int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("trtmc-vision-" + std::to_string(std::random_device{}()) + ".png");
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
