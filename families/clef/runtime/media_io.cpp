/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clef/runtime/media_io.h"

#include "families/clef/runtime/record.h"

#include <fstream>
#include <memory>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb/stb_image.h"

namespace trtmc::clef {
StructuredDecisionRequest read_request(const std::filesystem::path& path) {
    std::ifstream source(path);
    if (!source)
        throw std::invalid_argument("cannot read Clef record: " + path.string());
    auto record = Json::parse(source);
    StructuredDecisionRequest request;
    auto image = [&](const Json& name) {
        const auto image_path = path.parent_path() / name.get<std::string>();
        ImageResult result;
        int channels;
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> data(
            stbi_load(image_path.c_str(), &result.width, &result.height, &channels, 3),
            stbi_image_free);
        if (!data)
            throw std::invalid_argument("cannot decode Clef image: " + image_path.string());
        result.channels = 3;
        result.pixels.assign(data.get(), data.get() + static_cast<std::size_t>(result.height) *
                                                          result.width * 3);
        return result;
    };
    for (const auto& name : record.value("images", Json::array()))
        request.images.push_back(image(name));
    for (const auto& frames : record.value("videos", Json::array())) {
        if (!frames.is_array() || frames.empty())
            throw std::invalid_argument("video must contain image frame paths");
        ImageResult video;
        video.num_frames = 0;
        for (const auto& name : frames) {
            auto frame = image(name);
            if (video.num_frames > 0 &&
                (frame.height != video.height || frame.width != video.width))
                throw std::invalid_argument("video frame dimensions must match");
            video.height = frame.height;
            video.width = frame.width;
            video.pixels.insert(video.pixels.end(), frame.pixels.begin(), frame.pixels.end());
            ++video.num_frames;
        }
        request.videos.push_back(std::move(video));
    }
    record.erase("images");
    record.erase("videos");
    request.document = record.dump();
    return request;
}
} // namespace trtmc::clef
