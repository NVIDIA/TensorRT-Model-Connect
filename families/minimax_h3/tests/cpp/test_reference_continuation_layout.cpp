/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minimax_h3/runtime/reference_layout.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace trtmc::minimax_h3;
namespace {
void require(bool value, const char* label) {
    if (!value)
        throw std::runtime_error(label);
}
template <class F>
void rejects(F&& operation) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("invalid continuation input accepted");
}
template <class T>
void dump(const std::vector<T>& values) {
    std::cout.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}
void dump_case(const std::string& field) {
    const auto input = nlohmann::json::parse(std::cin);
    const auto target = input.at("target").get<std::vector<int32_t>>();
    const auto guide = input.at("guide").get<std::vector<int32_t>>();
    require(target.size() == 4 && guide.size() == 2, "fixture shape");
    std::vector<ImageReferenceLatentShape> images;
    for (const auto& value : input.at("images"))
        images.push_back({value.at(0), value.at(1)});
    const auto origin = input.contains("media_origin")
                            ? std::optional<int32_t>(input.at("media_origin"))
                            : std::nullopt;
    const auto tags = input.at("text_tags").get<std::vector<int32_t>>();
    const ReferenceTargetLatentShape shape{target[0], target[1], target[2], target[3]};
    const auto layout =
        guide == std::vector<int32_t>{0, 0}
            ? make_image_reference_layout(tags, images, shape, input.at("max_rows"), origin)
            : make_continuation_reference_layout(tags, images, shape, input.at("max_rows"),
                                                 {guide[0], guide[1]}, input.at("offset"), origin);
    if (field == "positions")
        dump(layout.positions);
    else if (field == "tags")
        dump(layout.token_tags);
    else if (field == "text")
        dump(layout.text_indices);
    else if (field == "audio")
        dump(layout.audio_indices);
    else if (field == "video")
        dump(layout.video_indices);
    else if (field == "target_audio")
        dump(layout.target_audio_indices);
    else if (field == "target_video")
        dump(layout.target_video_indices);
    else if (field == "packed")
        dump(layout.packed_indices);
    else {
        const auto times =
            make_image_reference_timesteps(layout, input.at("video_time"), input.at("audio_time"));
        if (field == "timesteps")
            dump(times.timesteps);
        else if (field == "timestep_indices")
            dump(times.timestep_indices);
        else if (field == "adaln_indices")
            dump(times.adaln_indices);
        else
            throw std::invalid_argument("unknown dump field");
    }
}
void tests() {
    const auto value = make_continuation_reference_layout({1, 0, 1}, {{4, 6}, {2, 4}}, {2, 4, 6, 3},
                                                          37, {1, 1}, 170.0);
    require(value.token_tags.size() == 37 && value.condition_video_rows == 14 &&
                value.condition_audio_rows == 2 && value.continuation_video_rows == 6,
            "guide row counts");
    require(value.audio_indices == std::vector<int32_t>({11, 12, 19, 20, 21, 22, 23, 24}),
            "stereo guide precedes target audio");
    require(value.target_audio_indices.front() == 19 && value.target_video_indices.front() == 25,
            "output heads select only generated rows");
    require(value.positions[3 * 3] == 3 && value.positions[9 * 3] == 4,
            "static references remain unshifted");
    require(value.positions[11 * 3] == 175 && value.positions[13 * 3] == 175 &&
                value.positions[19 * 3] == 175 && value.positions[25 * 3] == 175,
            "guide and target share the global media origin");
    auto packed = value.packed_indices;
    std::sort(packed.begin(), packed.end());
    for (int i = 0; i < 37; ++i)
        require(packed[i] == i, "packed indices form a complete permutation");
    const auto times = make_image_reference_timesteps(value, .25F, .5F);
    require(times.timesteps == std::vector<float>({.25F, .5F, .999F, 1.0F}),
            "clean audio has a separate timestep slot");
    require(times.timestep_indices[11] == 3 && times.timestep_indices[12] == 3 &&
                times.timestep_indices[19] == 1 && times.timestep_indices[13] == 2,
            "reference and generated AV timesteps remain independent");
    const auto request = [](ReferenceContinuationGuideShape guide, double offset, int cap,
                            std::optional<int32_t> origin = std::nullopt) {
        return make_continuation_reference_layout({1, 0, 1}, {{4, 6}, {2, 4}}, {2, 4, 6, 3}, cap,
                                                  guide, offset, origin);
    };
    rejects([&] { request({0, 1}, 0, 37); });
    rejects([&] { request({1, 0}, 0, 37); });
    rejects([&] { request({3, 1}, 0, 100); });
    rejects([&] { request({1, 4}, 0, 100); });
    rejects([&] { request({1, 1}, -1, 37); });
    rejects([&] { request({1, 1}, std::numeric_limits<double>::infinity(), 37); });
    rejects([&] { request({1, 1}, std::numeric_limits<double>::quiet_NaN(), 37); });
    rejects([&] { request({1, 1}, 0, 36); });
    rejects([&] { request({1, 1}, 0, 37, 2); });
    rejects([&] { make_image_reference_layout({1, 0, 1}, {{4, 6}}, {2, 4, 6, 3}, 100, 2); });
    rejects([&] { make_image_reference_layout({1, 0, 1}, {{4, 6}}, {2, 4, 6, 3}, 100, -1); });
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--dump")
            dump_case(argv[2]);
        else {
            require(argc == 1, "unexpected arguments");
            tests();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
