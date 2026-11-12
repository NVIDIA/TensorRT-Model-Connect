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

namespace {
using Json = nlohmann::json;
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
class Editor final : public trtmc::IImageEditing, public trtmc::IImageGeneration {
  public:
    int calls{0};
    const char* task() const noexcept override { return trtmc::IImageEditing::kTask; }
    trtmc::ImageResult generate_image(const std::string&,
                                      const trtmc::ImageGenerationConfig&) override {
        throw std::runtime_error("editing must consume the reference image");
    }
    trtmc::ImageResult generate_image(const std::string& prompt, const float* pixels,
                                      std::int32_t height, std::int32_t width,
                                      const trtmc::ImageGenerationConfig& config) override {
        require(prompt == "Watercolor" && height == 1 && width == 1 && pixels != nullptr,
                "the actual decoded reference and instruction reach the edit Task");
        require(config.seed == 42 && config.num_steps == 20 && config.guidance_scale == 4 &&
                    config.cfg_scale == 4 && config.height == 1 && config.width == 1 &&
                    config.negative_prompt == "blur" &&
                    config.initial_latents == std::vector<float>({0.5F, -0.5F}),
                "all explicit controls and replay reach the public Task");
        ++calls;
        trtmc::ImageResult result;
        result.height = result.width = 1;
        result.pixels.assign(pixels, pixels + 3);
        return result;
    }
};
void verify(const std::filesystem::path& root) {
    const auto red = root / "red.png", blue = root / "blue.png", noise = root / "noise.f32";
    trtmc::cli::io::save_png(red.string(), std::vector<float>{1, 0, 0}, 1, 1);
    trtmc::cli::io::save_png(blue.string(), std::vector<float>{0, 0, 1}, 1, 1);
    const float values[]{0.5F, -0.5F};
    std::ofstream(noise, std::ios::binary)
        .write(reinterpret_cast<const char*>(values), sizeof(values));
    const Json request{{"id", "edit"},
                       {"op", "edit_image"},
                       {"prompt", "Watercolor"},
                       {"image_path", red.string()},
                       {"output_path", (root / "first.png").string()},
                       {"initial_latents_path", noise.string()},
                       {"config",
                        {{"seed", 42},
                         {"num_steps", 20},
                         {"height", 1},
                         {"width", 1},
                         {"guidance_scale", 4},
                         {"cfg_scale", 4},
                         {"negative_prompt", "blur"}}}};
    std::stringstream input;
    auto bad = request;
    bad["op"] = "generate_image";
    input << bad.dump() << '\n';
    bad = request;
    bad["image_path"] = (root / "missing.png").string();
    input << bad.dump() << '\n';
    input << request.dump() << '\n';
    auto second = request;
    second["image_path"] = blue.string();
    second["output_path"] = (root / "second.png").string();
    input << second.dump() << '\n';
    Editor editor;
    std::ostringstream output;
    require(trtmc::server::run_task_worker(editor, input, output) == 0 && editor.calls == 2,
            "persistent editing recovers from invalid requests");
    std::istringstream lines(output.str());
    std::vector<Json> messages;
    for (std::string line; std::getline(lines, line);)
        messages.push_back(Json::parse(line));
    require(messages.size() == 5 && messages[0]["capabilities"] == Json::array({"image_edit"}),
            "declared editing is distinct from inherited generation");
    for (std::size_t i = 1; i < 3; ++i)
        require(messages[i]["error"]["type"] == "invalid_request_error",
                "input failures are recoverable");
    const auto& result = messages.back().at("result");
    require(result["image_width"] == 1 && result["image_height"] == 1 &&
                result["output_count"] == 1 && result["model_call_ms"].get<double>() >= 0 &&
                result["timing_scope"] == "public_task_call_wall",
            "record actual input, output and Task timing");
    require(trtmc::cli::io::read_image((root / "first.png").string()).pixels ==
                    std::vector<float>({1, 0, 0}) &&
                trtmc::cli::io::read_image((root / "second.png").string()).pixels ==
                    std::vector<float>({0, 0, 1}),
            "different uploaded references influence complete generated PNG pixels");
}
} // namespace
int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("trtmc-edit-" + std::to_string(std::random_device{}()));
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
