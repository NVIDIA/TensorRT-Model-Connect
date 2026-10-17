/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_layout.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace trtmc::minimax_h3;

void require(bool value, const char* label) {
    if (!value)
        throw std::runtime_error(label);
}

template <class Exception, class Callable>
void rejects(Callable&& operation) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("invalid input was accepted");
}

void test_layout() {
    const auto value = make_image_reference_layout({1, 0, 1}, {{4, 6}, {2, 4}}, {2, 4, 6, 3}, 29);
    require(value.token_tags.size() == 29 && value.positions.size() == 87, "layout shape");
    require(value.text_indices == std::vector<int32_t>({0, 1, 2}), "text row prefix");
    require(value.condition_video_rows == 8, "conditions keep their own geometry");
    require(value.video_indices.size() == 20 && value.video_indices[8] == 17,
            "video target suffix");
    require(value.target_audio_indices == value.audio_indices && value.audio_indices.size() == 6,
            "image-only references do not invent a reference soundtrack");
    require(value.target_video_indices.front() == 17 && value.target_video_indices.back() == 28,
            "only generated rows reach the output head");
    require(value.positions[3 * 3] == 3.0 && value.positions[9 * 3] == 4.0,
            "each image advances rotary time by one");
    require(value.positions[11 * 3] == 5.0 && value.positions[17 * 3] == 5.0,
            "generated audio/video share their origin after all references");
    require(value.positions[23 * 3] == 5.0 + 5.0 / 3.0,
            "generated frame time adds origin after accumulating frame spans");
    std::vector<int32_t> concat = value.text_indices;
    concat.insert(concat.end(), value.audio_indices.begin(), value.audio_indices.end());
    concat.insert(concat.end(), value.video_indices.begin(), value.video_indices.end());
    for (std::size_t row = 0; row < concat.size(); ++row)
        require(concat[static_cast<std::size_t>(value.packed_indices[row])] ==
                    static_cast<int32_t>(row),
                "inverse gather preserves condition placement");
    const auto first = make_image_reference_timesteps(value, 0.0F, 0.0F);
    require(first.timesteps == std::vector<float>({0.0F, 0.999F}), "equal times collapse");
    require(first.timestep_indices[1] == 0 && first.adaln_indices[1] == 0,
            "Qwen video-tagged text uses the target video time, not the condition time");
    require(first.timestep_indices[3] == 1 && first.adaln_indices[3] == 3,
            "reference latent uses the condition time and video modality");
    require(first.adaln_indices[11] == 2, "audio tag does not change its timestep");
    const auto middle = make_image_reference_timesteps(value, 0.2F, 0.4F);
    require(middle.timesteps == std::vector<float>({0.2F, 0.4F, 0.999F}),
            "sorted unique time table");
    const auto late = make_image_reference_timesteps(value, 0.9995F, 0.7F);
    require(late.timesteps == std::vector<float>({0.7F, 0.9995F}),
            "condition time follows late target time");
    require(late.timestep_indices[3] == late.timestep_indices[17],
            "late reference and target slots merge");
}

void test_rejections() {
    const ReferenceTargetLatentShape target{2, 4, 6, 3};
    rejects<std::length_error>(
        [&] { make_image_reference_layout({1, 0, 1}, {{4, 6}, {2, 4}}, target, 28); });
    rejects<std::invalid_argument>([&] { make_image_reference_layout({1}, {}, target, 100); });
    rejects<std::invalid_argument>([&] { make_image_reference_layout({}, {{4, 6}}, target, 100); });
    rejects<std::invalid_argument>(
        [&] { make_image_reference_layout({2}, {{4, 6}}, target, 100); });
    rejects<std::invalid_argument>(
        [&] { make_image_reference_layout({1}, {{3, 6}}, target, 100); });
    rejects<std::invalid_argument>(
        [&] { make_image_reference_layout({1}, {{4, 6}}, {2, 4, 6, 0}, 100); });
    rejects<std::invalid_argument>([&] {
        make_image_reference_layout({1}, std::vector<ImageReferenceLatentShape>(10, {2, 2}), target,
                                    100);
    });
    rejects<std::length_error>([&] {
        make_image_reference_layout({1}, {{4, 6}}, {std::numeric_limits<int32_t>::max(), 4, 6, 3},
                                    100);
    });
    auto layout = make_image_reference_layout({1}, {{2, 2}}, {1, 2, 2, 1}, 10);
    for (const float bad : {-0.1F, 1.1F, std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
        rejects<std::invalid_argument>([&] { make_image_reference_timesteps(layout, bad, 0.2F); });
        rejects<std::invalid_argument>([&] { make_image_reference_timesteps(layout, 0.2F, bad); });
    }
    layout.token_tags[0] = -1;
    rejects<std::invalid_argument>([&] { make_image_reference_timesteps(layout, 0.1F, 0.2F); });
    rejects<std::invalid_argument>([] { pack_reference_video_rows({}, 1, 4, 6); });
}

std::vector<float> fixture_latents(int32_t frames, int32_t height, int32_t width) {
    std::vector<float> values(static_cast<std::size_t>(24) * frames * height * width);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<float>(static_cast<int32_t>(index % 16381) - 8190) / 16.0F;
    return values;
}

void test_patchify() {
    const auto input = fixture_latents(3, 4, 6);
    const auto packed = pack_reference_video_rows(input, 3, 4, 6);
    require(input.size() == packed.size(), "patchify preserves all values");
    std::size_t row = 0;
    for (int32_t frame = 0; frame < 3; ++frame)
        for (int32_t y = 0; y < 4; y += 2)
            for (int32_t x = 0; x < 6; x += 2) {
                for (int32_t channel = 0; channel < 24; ++channel)
                    for (int32_t dy = 0; dy < 2; ++dy)
                        for (int32_t dx = 0; dx < 2; ++dx)
                            require(packed[row * 96 + channel * 4 + dy * 2 + dx] ==
                                        input[((channel * 3 + frame) * 4 + y + dy) * 6 + x + dx],
                                    "C/T/H/W patchify permutation");
                ++row;
            }
}

template <class T>
void dump(const std::vector<T>& values) {
    std::cout.write(reinterpret_cast<const char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void dump_case(const std::string& field) {
    const auto input = nlohmann::json::parse(std::cin);
    const auto tags = input.at("text_tags").get<std::vector<int32_t>>();
    const auto dims = input.at("target").get<std::vector<int32_t>>();
    require(dims.size() == 4, "target fixture shape");
    std::vector<ImageReferenceLatentShape> images;
    for (const auto& image : input.at("images"))
        images.push_back({image.at(0).get<int32_t>(), image.at(1).get<int32_t>()});
    const auto layout = make_image_reference_layout(
        tags, images, {dims[0], dims[1], dims[2], dims[3]}, input.at("max_rows").get<int32_t>());
    if (field == "positions")
        dump(layout.positions);
    else if (field == "tags")
        dump(layout.token_tags);
    else if (field == "text")
        dump(layout.text_indices);
    else if (field == "video")
        dump(layout.video_indices);
    else if (field == "audio")
        dump(layout.audio_indices);
    else if (field == "packed")
        dump(layout.packed_indices);
    else if (field == "target_video")
        dump(layout.target_video_indices);
    else if (field == "target_audio")
        dump(layout.target_audio_indices);
    else if (field == "timesteps" || field == "timestep_indices" || field == "adaln_indices") {
        const auto plan = make_image_reference_timesteps(
            layout, input.at("video_time").get<float>(), input.at("audio_time").get<float>());
        if (field == "timesteps")
            dump(plan.timesteps);
        else if (field == "timestep_indices")
            dump(plan.timestep_indices);
        else
            dump(plan.adaln_indices);
    } else if (field == "patches") {
        dump(pack_reference_video_rows(fixture_latents(dims[0], dims[1], dims[2]), dims[0], dims[1],
                                       dims[2]));
    } else
        throw std::invalid_argument("unknown fixture field");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--dump") {
            dump_case(argv[2]);
            return 0;
        }
        require(argc == 1, "unexpected test arguments");
        test_layout();
        test_patchify();
        test_rejections();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
