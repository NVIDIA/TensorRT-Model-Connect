/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kHeight = 1280;
constexpr std::int32_t kWidth = 1088;
constexpr std::size_t kFrameCount = 5;

using Frames = std::array<std::vector<std::uint8_t>, kFrameCount>;

Frames make_frames() {
    Frames frames;
    for (std::size_t index = 0; index < frames.size(); ++index) {
        auto& frame = frames[index];
        frame.resize(static_cast<std::size_t>(kHeight) * kWidth * 3U);
        for (std::int32_t row = 0; row < kHeight; ++row) {
            for (std::int32_t column = 0; column < kWidth; ++column) {
                const auto offset = (static_cast<std::size_t>(row) * kWidth + column) * 3U;
                frame[offset] = static_cast<std::uint8_t>(column + 17 * index);
                frame[offset + 1U] = static_cast<std::uint8_t>(row + 31 * index);
                frame[offset + 2U] = static_cast<std::uint8_t>(
                    (column / 8) ^ (row / 8) ^ (53 * static_cast<std::int32_t>(index)));
            }
        }
        const std::int32_t top = 300 + 20 * static_cast<std::int32_t>(index);
        const std::int32_t left = 250 + 30 * static_cast<std::int32_t>(index);
        for (std::int32_t row = top; row < top + 500; ++row) {
            for (std::int32_t column = left; column < left + 400; ++column) {
                const auto offset = (static_cast<std::size_t>(row) * kWidth + column) * 3U;
                frame[offset] = 235;
                frame[offset + 1U] = static_cast<std::uint8_t>(45 + 20 * index);
                frame[offset + 2U] = 25;
            }
        }
    }
    return frames;
}

Frames load_frames(char** paths) {
    constexpr auto kFrameBytes = static_cast<std::size_t>(kHeight) * kWidth * 3U;
    Frames frames;
    for (std::size_t index = 0; index < frames.size(); ++index) {
        const std::filesystem::path path(paths[index]);
        const auto status = std::filesystem::symlink_status(path);
        if (std::filesystem::is_symlink(status) || !std::filesystem::is_regular_file(status))
            throw std::runtime_error("SAM2 local RGB8 input must be a regular non-symlink");
        if (std::filesystem::file_size(path) != kFrameBytes)
            throw std::runtime_error("SAM2 local RGB8 input has the wrong size");
        std::ifstream input(path, std::ios::binary);
        frames[index].resize(kFrameBytes);
        input.read(reinterpret_cast<char*>(frames[index].data()),
                   static_cast<std::streamsize>(frames[index].size()));
        if (!input || input.peek() != std::ifstream::traits_type::eof())
            throw std::runtime_error("SAM2 local RGB8 input could not be read exactly");
    }
    return frames;
}

trtmc::VideoSegmentationRequest request(const Frames& frames) {
    trtmc::VideoSegmentationRequest value;
    for (const auto& frame : frames) {
        value.frames.push_back(
            {frame.data(), frame.size(), kHeight, kWidth, trtmc::VideoFrameFormat::kRgb8});
    }
    return value;
}

bool equal(const trtmc::VideoSegmentationResult& left,
           const trtmc::VideoSegmentationResult& right) {
    if (left.frames.size() != right.frames.size())
        return false;
    for (std::size_t index = 0; index < left.frames.size(); ++index) {
        const auto& a = left.frames[index];
        const auto& b = right.frames[index];
        if (a.masks != b.masks || a.object_ids != b.object_ids ||
            a.detection_scores != b.detection_scores || a.tracking_scores != b.tracking_scores ||
            a.boxes != b.boxes || a.num_objects != b.num_objects || a.height != b.height ||
            a.width != b.width) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 9) {
            throw std::invalid_argument(
                "usage: operational_probe BUNDLE RUNTIME_ROOT MASKS_U8 [FRAME0 ... FRAME4]");
        }
        auto task = trtmc::load_task(argv[1], argv[2]);
        auto* video = dynamic_cast<trtmc::IVideoSegmentation*>(task.get());
        if (video == nullptr)
            throw std::runtime_error("SAM2 bundle does not implement video segmentation");
        auto session = video->create_video_segmentation_session();
        if (session == nullptr)
            throw std::runtime_error("SAM2 did not create a video session");
        const auto frames = argc == 9 ? load_frames(argv + 4) : make_frames();
        const auto input = request(frames);
        const auto first = session->segment(input);
        const auto second = session->segment(input);
        if (first.frames.size() != kFrameCount)
            throw std::runtime_error("SAM2 did not return five frames");
        if (first.frames.front().boxes.empty() || first.frames.front().detection_scores.empty() ||
            first.frames.front().object_ids.empty()) {
            throw std::runtime_error("SAM2 did not return detector metadata");
        }

        const std::vector<float> expected_box{136.0F, 160.0F, 952.0F, 1120.0F};
        bool metadata_exact = true;
        bool binary = true;
        std::vector<std::int64_t> counts;
        std::vector<std::vector<std::uint8_t>> masks;
        for (const auto& frame : first.frames) {
            metadata_exact = metadata_exact && frame.num_objects == 1 &&
                             frame.object_ids == std::vector<std::int32_t>{1} &&
                             frame.detection_scores == std::vector<float>{1.0F} &&
                             frame.boxes == expected_box;
            binary = binary && frame.masks.size() == static_cast<std::size_t>(kHeight) * kWidth &&
                     std::all_of(frame.masks.begin(), frame.masks.end(),
                                 [](std::uint8_t value) { return value <= 1; });
            counts.push_back(std::count(frame.masks.begin(), frame.masks.end(), std::uint8_t{1}));
            masks.push_back(frame.masks);
        }
        const bool temporal = std::any_of(masks.begin() + 1, masks.end(),
                                          [&](const auto& mask) { return mask != masks.front(); });
        std::ofstream mask_output(argv[3], std::ios::binary | std::ios::trunc);
        if (!mask_output)
            throw std::runtime_error("cannot create SAM2 mask output");
        for (const auto& mask : masks) {
            mask_output.write(reinterpret_cast<const char*>(mask.data()),
                              static_cast<std::streamsize>(mask.size()));
        }
        if (!mask_output)
            throw std::runtime_error("cannot write SAM2 mask output");
        nlohmann::json receipt{
            {"same_session_repeat_exact", equal(first, second)},
            {"bbox_xyxy", first.frames.front().boxes},
            {"detector_score", first.frames.front().detection_scores.front()},
            {"label", first.frames.front().object_ids.front()},
            {"binary_masks", binary},
            {"mask_foreground_pixels", counts},
            {"temporally_distinct_masks", temporal},
            {"metadata_exact", metadata_exact},
        };
        std::cout << receipt.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
