/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cli/io.h"
#include "server/native_worker.h"
#include "trtmc/task.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>

using Json = nlohmann::json;
namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
class Generator final : public trtmc::IImageGeneration {
  public:
    int calls{0};
    const char* task() const noexcept override { return kTask; }
    trtmc::ImageResult generate_image(const std::string& prompt,
                                      const trtmc::ImageGenerationConfig& config) override {
        require(prompt == "A red rectangle" && config.num_steps == 10 && config.seed == 42 &&
                    config.guidance_scale == 7.5F && config.cfg_scale == 1.0F &&
                    config.negative_prompt == "blur" && config.height == 4 && config.width == 8,
                "explicit controls reach the existing Task unchanged");
        require(config.initial_latents == std::vector<float>({0.5F, -0.5F}),
                "replay latents reach the Task unchanged");
        ++calls;
        trtmc::ImageResult result;
        result.height = 4;
        result.width = 8;
        result.pixels.assign(4 * 8 * 3, 0.0F);
        for (std::size_t i = 0; i < result.pixels.size(); i += 3)
            result.pixels[i] = 1.0F;
        return result;
    }
};
void verify(const std::filesystem::path& root) {
    const auto noise = root / "noise.f32";
    const float values[]{0.5F, -0.5F};
    std::ofstream(noise, std::ios::binary)
        .write(reinterpret_cast<const char*>(values), sizeof(values));
    const Json request{{"id", "image"},
                       {"op", "generate_image"},
                       {"prompt", "A red rectangle"},
                       {"output_path", (root / "image.png").string()},
                       {"initial_latents_path", noise.string()},
                       {"config",
                        {{"num_steps", 10},
                         {"seed", 42},
                         {"guidance_scale", 7.5},
                         {"cfg_scale", 1.0},
                         {"height", 4},
                         {"width", 8},
                         {"negative_prompt", "blur"}}}};
    std::stringstream input;
    auto bad = request;
    bad["config"]["seed"] = 4294967296ULL;
    input << bad.dump() << '\n';
    bad = request;
    bad["config"]["unknown"] = 1;
    input << bad.dump() << '\n';
    bad = request;
    bad["initial_latents_path"] = (root / "missing.f32").string();
    input << bad.dump() << '\n';
    input << request.dump() << '\n' << request.dump() << '\n';
    Generator task;
    std::ostringstream output;
    require(trtmc::server::run_task_worker(task, input, output) == 0 && task.calls == 2,
            "persistent Task recovers from invalid requests");
    std::vector<Json> messages;
    std::istringstream lines(output.str());
    for (std::string line; std::getline(lines, line);)
        messages.push_back(Json::parse(line));
    require(messages.size() == 6 &&
                messages[0]["capabilities"] == Json::array({"image_generation"}),
            "image capability is explicit");
    for (std::size_t i = 1; i <= 3; ++i)
        require(messages[i]["error"]["type"] == "invalid_request_error",
                "input errors are recoverable");
    const auto result = messages.back().at("result");
    require(result["width"] == 8 && result["height"] == 4 && result["output_count"] == 1 &&
                result["model_call_ms"].get<double>() >= 0 &&
                result["timing_scope"] == "public_task_call_wall",
            "real output and Task timing");
    const auto image = trtmc::cli::io::read_image((root / "image.png").string());
    require(image.width == 8 && image.height == 4 && image.pixels[0] == 1.0F &&
                image.pixels[1] == 0.0F,
            "existing CLI PNG codec preserves generated pixels");
}
} // namespace
int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("trtmc-image-" + std::to_string(std::random_device{}()));
    try {
        std::filesystem::create_directory(root);
        verify(root);
        std::filesystem::remove_all(root);
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
