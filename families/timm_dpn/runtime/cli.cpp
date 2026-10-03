/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/cli.h"

#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

std::vector<float> read_image(const std::string& path, int& width, int& height) {
    int channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> image(
        stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!image || width <= 0 || height <= 0)
        throw std::invalid_argument("unable to decode classification image");
    std::vector<float> pixels(static_cast<std::size_t>(width) * height * 3);
    std::transform(image.get(), image.get() + pixels.size(), pixels.begin(),
                   [](stbi_uc value) { return value / 255.0F; });
    return pixels;
}

Json classify(const Json& values, const char* default_runtime_root) {
    const trtmc::BundleReader reader(values.at("bundle").get<std::string>());
    if (reader.info().family != "timm_dpn")
        throw std::invalid_argument("timm_dpn CLI requires its own family bundle");
    int width = 0, height = 0;
    const auto pixels = read_image(values.at("image").get<std::string>(), width, height);
    const auto runtime_root = values.value("runtime_root", std::string(default_runtime_root));
    const auto runtime_cache = values.value("runtime_cache", std::string{});
    const auto cuda_graphs = values.value("cuda_graphs", false);
    auto task = trtmc::load_task(reader, runtime_root, 0, runtime_cache, cuda_graphs);
    auto* classifier = dynamic_cast<trtmc::IImageClassification*>(task.get());
    if (!classifier)
        throw std::invalid_argument("bundle does not implement image classification");
    const auto result = classifier->classify(pixels.data(), height, width);
    for (const auto value : result.logits) {
        if (!std::isfinite(value))
            throw std::runtime_error("classification returned non-finite logits");
    }
    if (!std::isfinite(result.top_score))
        throw std::runtime_error("classification returned a non-finite top score");
    return {{"logits", result.logits},
            {"top_class", result.top_class},
            {"top_score", result.top_score}};
}
} // namespace

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        if (std::string(handler) != "classify")
            throw std::invalid_argument("unknown timm_dpn CLI handler");
        const auto result = classify(Json::parse(values_json), default_runtime_root).dump() + '\n';
        output(context, result.data(), result.size());
        return 0;
    } catch (const std::exception& exception) {
        const auto message = std::string("Error: ") + exception.what() + '\n';
        error(context, message.data(), message.size());
        return 1;
    }
}
