/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/runtime/pipeline.h"
#include "families/lighton_ocr/tests/cpp/image_io.h"
#include "trtmc/runtime/family_loader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

namespace {
void save(const std::filesystem::path& directory, nlohmann::json& metadata, const std::string& name,
          const std::vector<float>& values, const std::vector<std::int64_t>& shape) {
    const auto file = name + ".f32";
    std::ofstream stream(directory / file, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream)
        throw std::runtime_error("Cannot write diagnostic tensor");
    metadata["tensors"][name] = {{"file", file}, {"shape", shape}, {"dtype", "float32"}};
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6)
        return 2;
    try {
        int width = 0, height = 0;
        std::unique_ptr<unsigned char, decltype(&lighton_ocr_free_rgb)> pixels(
            lighton_ocr_load_rgb(argv[3], &width, &height), lighton_ocr_free_rgb);
        if (!pixels)
            throw std::runtime_error("Cannot read diagnostic RGB image");
        std::ifstream input(argv[4]);
        nlohmann::json request;
        input >> request;
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* pipeline = dynamic_cast<trtmc::lighton_ocr::Pipeline*>(task.get());
        if (!pipeline)
            throw std::runtime_error("Diagnostic requires the real LightOnOCR native family");
        const trtmc::internal::ImageView image{pixels.get(),
                                               static_cast<std::size_t>(height) * width * 3,
                                               static_cast<std::uint32_t>(height),
                                               static_cast<std::uint32_t>(width),
                                               3,
                                               trtmc::internal::ImageFormat::UInt8};
        const auto observed = pipeline->observe(
            image, request.at("teacher_forced_ids").get<std::vector<std::int32_t>>());
        const std::filesystem::path directory(argv[5]);
        std::filesystem::create_directories(directory);
        nlohmann::json metadata{{"precision", "fp32"},
                                {"prompt_ids", observed.prompt_ids},
                                {"cache_prefix_unchanged", observed.cache_prefix_unchanged}};
        save(directory, metadata, "pixel_values", observed.image.pixels,
             {1, 3, observed.image.height, observed.image.width});
        save(directory, metadata, "image_features", observed.image_features,
             {static_cast<std::int64_t>(observed.image_features.size() / 1024), 1024});
        for (std::size_t step = 0; step < observed.logits.size(); ++step)
            save(directory, metadata, "step_" + std::to_string(step) + "_logits",
                 observed.logits[step],
                 {1, static_cast<std::int64_t>(observed.logits[step].size())});
        std::ofstream(directory / "metadata.json") << metadata.dump(2) << '\n';
        std::cout << "LightOnOCR native observations written\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
