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
#include <cmath>
#include <memory>
#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Image {
    std::vector<float> pixels;
    int width = 0, height = 0;
};

Image read_image(const std::string& path) {
    Image result;
    int channels = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> bytes(
        stbi_load(path.c_str(), &result.width, &result.height, &channels, 3), stbi_image_free);
    if (!bytes || result.width <= 0 || result.height <= 0)
        throw std::runtime_error("unable to decode image: " + path);
    const auto count = static_cast<std::size_t>(result.width) * result.height * 3U;
    result.pixels.resize(count);
    for (std::size_t i = 0; i < count; ++i)
        result.pixels[i] = static_cast<float>(bytes.get()[i]) / 255.0F;
    return result;
}

void require_finite(const std::vector<float>& values) {
    for (const auto value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error("dinov3 returned a non-finite result");
    }
}

nlohmann::json execute(const std::string& handler, const nlohmann::json& values,
                       const char* default_runtime_root) {
    if (handler != "extract_features")
        throw std::invalid_argument("unknown dinov3 CLI handler: " + handler);
    const trtmc::BundleReader reader(values.at("bundle").get<std::string>());
    if (reader.info().family != "dinov3")
        throw std::invalid_argument("dinov3 CLI requires a dinov3 bundle");
    const auto image = read_image(values.at("image").get<std::string>());
    auto task = trtmc::load_task(
        reader, values.value("runtime_root", std::string(default_runtime_root)), 0,
        values.value("runtime_cache", std::string()), values.value("cuda_graphs", false));
    auto* model = dynamic_cast<trtmc::IImageFeatureExtractor*>(task.get());
    if (!model)
        throw std::invalid_argument("dinov3 bundle does not implement IImageFeatureExtractor");
    const auto result =
        model->extract_image_features(image.pixels.data(), image.height, image.width);
    require_finite(result.last_hidden_state);
    require_finite(result.pooler_output);
    return {{"last_hidden_state", result.last_hidden_state},
            {"last_hidden_state_shape", result.last_hidden_state_shape},
            {"pooler_output", result.pooler_output},
            {"pooler_output_shape", result.pooler_output_shape}};
}
} // namespace

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        const auto result =
            execute(handler, nlohmann::json::parse(values_json), default_runtime_root).dump() +
            '\n';
        output(context, result.data(), result.size());
        return 0;
    } catch (const std::exception& exception) {
        const auto message = std::string("Error: ") + exception.what() + '\n';
        error(context, message.data(), message.size());
        return 1;
    } catch (...) {
        const std::string message = "Error: dinov3 CLI failed\n";
        error(context, message.data(), message.size());
        return 1;
    }
}
