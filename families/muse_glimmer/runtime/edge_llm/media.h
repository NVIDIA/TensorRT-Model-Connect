/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "families/muse_glimmer/runtime/edge_llm/request.h"
#include "families/muse_glimmer/runtime/media_request.h"
#include "families/muse_glimmer/runtime/vision_pixels.h"

#include <algorithm>
#include <limits>

namespace trtmc::muse_glimmer::edge_llm {
inline trt_edgellm::rt::imageUtils::ImageData pixels(Span<const internal::ImageView> frames,
                                                     bool video, double fps, int capacity) {
    if (!frames.size() || !frames.data() || frames.size() > 2 * static_cast<size_t>(capacity))
        throw std::invalid_argument("Invalid Muse Edge visual frame count");
    const auto& first = frames.data()[0];
    // Validate storage and dimensions before allocating an Edge tensor.
    auto rgb = visual_rgb(first);
    if (frames.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / rgb.size())
        throw std::invalid_argument("Muse Edge frame stack exceeds tensor capacity");
    using namespace trt_edgellm::rt;
    using Dtype = decltype(Tensor{}.getDataType());
    Tensor tensor({static_cast<int64_t>(frames.size()), first.height, first.width, 3},
                  DeviceType::kCPU, Dtype::kUINT8);
    auto* out = static_cast<uint8_t*>(tensor.rawPointer());
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i != 0) {
            const auto& frame = frames.data()[i];
            if (frame.width != first.width || frame.height != first.height)
                throw std::invalid_argument("Muse Edge video requires equal frame dimensions");
            rgb = visual_rgb(frame);
        }
        out = std::copy(rgb.begin(), rgb.end(), out);
    }
    imageUtils::ImageData data(std::move(tensor));
    data.isVideo = video;
    data.fps = fps;
    return data;
}

inline trt_edgellm::rt::LLMGenerationRequest media_request(const MediaTurns& turns,
                                                           const TextGenerationConfig& cfg,
                                                           int budget, int capacity, double fps) {
    auto request = make_request("", cfg, budget);
    auto& row = request.requests.front();
    row.messages.clear();
    for (const auto& turn : turns) {
        auto& msg = row.messages.emplace_back();
        msg.role = turn.role;
        msg.contentIsArray = true;
        for (const auto& part : turn.parts) {
            if (const auto* text = std::get_if<std::string>(&part)) {
                msg.contents.push_back({"text", *text});
            } else if (const auto* image = std::get_if<internal::ImageView>(&part)) {
                msg.contents.push_back({"image", ""});
                row.imageBuffers.push_back(pixels({image, 1}, false, fps, capacity));
            } else {
                const auto& video = std::get<internal::VideoView>(part);
                auto data = pixels(video.frames, true, fps, capacity);
                for (size_t i = 0; i < video.timestamps_seconds.size(); ++i)
                    data.timestamps.push_back(video.timestamps_seconds.data()[i]);
                msg.contents.push_back({"video", ""});
                row.imageBuffers.push_back(std::move(data));
            }
        }
    }
    return request;
}
} // namespace trtmc::muse_glimmer::edge_llm
