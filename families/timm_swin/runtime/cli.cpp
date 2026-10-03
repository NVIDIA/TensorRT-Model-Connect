/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/cli.h"

#include "trtmc/bundle.h"
#include "trtmc/core.hpp"
#include "trtmc/features.hpp"

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
    if (reader.info().family != "timm_swin")
        throw std::invalid_argument("timm_swin CLI requires its own family bundle");
    int width = 0, height = 0;
    const auto pixels = read_image(values.at("image").get<std::string>(), width, height);
    const auto runtime_root = values.value("runtime_root", std::string(default_runtime_root));
    const auto runtime_cache = values.value("runtime_cache", std::string{});
    const auto cuda_graphs = values.value("cuda_graphs", false);
    const auto model =
        trtmc::Model::load(reader.path(), {runtime_root, 0, runtime_cache, cuda_graphs});
    const auto task = model.task<trtmc::ImageToClassScores>();
    const auto result = task.run(
        {trtmc::ImageInput{trtmc::Span<const float>{pixels.data(), pixels.size()},
                           static_cast<std::uint32_t>(height), static_cast<std::uint32_t>(width)}},
        {});
    std::vector<float> scores(result.scores().begin(), result.scores().end());
    for (const auto value : scores) {
        if (!std::isfinite(value))
            throw std::runtime_error("classification returned a non-finite score");
    }
    const char* kind = nullptr;
    switch (result.kind()) {
    case TRTMC_SCORE_LOGIT:
        kind = "logit";
        break;
    case TRTMC_SCORE_PROBABILITY:
        kind = "probability";
        break;
    case TRTMC_SCORE_UNBOUNDED:
        kind = "unbounded";
        break;
    default:
        throw std::runtime_error("classification returned an unknown score kind");
    }
    auto labels = Json::array();
    for (const auto label : result.labels())
        labels.push_back(std::string(label));
    Json output{{"scores", scores},
                {"score_kind", kind},
                {"labels", labels},
                {"vocabulary_id", std::string(result.vocabulary_id())},
                {"task", trtmc::ImageToClassScores::kTask}};
    if (result.kind() == TRTMC_SCORE_LOGIT)
        output["logits"] = scores;
    if (scores.empty()) {
        output["top_class"] = -1;
        output["top_score"] = nullptr;
    } else {
        const auto best = std::max_element(scores.begin(), scores.end());
        output["top_class"] = best - scores.begin();
        output["top_score"] = *best;
    }
    return output;
}
} // namespace

extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        if (std::string(handler) != "classify")
            throw std::invalid_argument("unknown timm_swin CLI handler");
        const auto result = classify(Json::parse(values_json), default_runtime_root).dump() + '\n';
        output(context, result.data(), result.size());
        return 0;
    } catch (const std::exception& exception) {
        const auto message = std::string("Error: ") + exception.what() + '\n';
        error(context, message.data(), message.size());
        return 1;
    }
}
