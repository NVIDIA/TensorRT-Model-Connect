/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/visual_reference.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool valid, const char* message) {
    if (!valid)
        throw std::runtime_error(message);
}

void test_axis() {
    using trtmc::minimax_h3::make_visual_reference_axis;
    for (int32_t length = 32; length <= 4096; length += 16) {
        const auto axis = make_visual_reference_axis(length);
        require(axis.starts.front() == 0, "axis starts at zero");
        require(axis.starts.back() + axis.lengths.back() == length, "axis covers exact extent");
        require(axis.overlaps.size() + 1 == axis.starts.size(), "axis overlap count");
        for (std::size_t index = 0; index < axis.overlaps.size(); ++index) {
            require(axis.starts[index] % 16 == 0, "latent-aligned start");
            require(axis.overlaps[index] >= 64 && axis.overlaps[index] < 256,
                    "original overlap bounds");
            require(axis.starts[index + 1] ==
                        axis.starts[index] + axis.lengths[index] - axis.overlaps[index],
                    "adjacent overlap");
        }
    }
    require(make_visual_reference_axis(768).overlaps == std::vector<int32_t>({96, 80, 80}),
            "round-robin slack distribution");
    require(make_visual_reference_axis(1344).starts ==
                std::vector<int32_t>({0, 176, 352, 528, 704, 896, 1088}),
            "1344 reference starts");
    for (const int32_t bad : {-16, 0, 16, 31, 257}) {
        bool rejected = false;
        try {
            (void)make_visual_reference_axis(bad);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "invalid geometry rejected");
    }
}

void test_stitch() {
    using trtmc::minimax_h3::stitch_visual_reference_moments;
    constexpr std::size_t count = 48 * 16 * 16;
    const std::vector<std::vector<std::vector<float>>> tiles = {
        {std::vector<float>(count, 10.0F), std::vector<float>(count, 20.0F)},
        {std::vector<float>(count, 30.0F), std::vector<float>(count, 40.0F)},
    };
    const auto output = stitch_visual_reference_moments(tiles, 272, 288, 1);
    require(output.size() == 48 * 17 * 18, "stitched moments shape");
    for (std::size_t channel = 0; channel < 48; ++channel) {
        const auto offset = channel * 17 * 18;
        require(output[offset] == 10.0F, "top-left original value");
        require(output[offset + 1 * 18 + 2] == 30.0F,
                "original left tile owns corner after vertical blend");
        const float vertical = 20.0F * (1.0F - 1.0F / 15) + 40.0F / 15;
        const float expected = 30.0F * (1.0F - 1.0F / 14) + vertical / 14;
        require(std::abs(output[offset + 2 * 18 + 3] - expected) < 1.0e-5F,
                "vertical then horizontal blend");
        require(output[offset + 16 * 18 + 17] == 40.0F, "last tile is not cropped");
    }
    require(tiles[1][1][0] == 40.0F, "raw moment tiles are immutable");
    const std::vector<std::vector<std::vector<float>>> single = {
        {std::vector<float>(48 * 5 * 2 * 3, 7.0F)}};
    require(stitch_visual_reference_moments(single, 32, 48, 5) == single[0][0],
            "single tile retains all moments");
    auto bad = tiles;
    bad[1][1].pop_back();
    bool rejected = false;
    try {
        (void)stitch_visual_reference_moments(bad, 272, 288, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "malformed payload rejected before indexing");
}

void dump_upstream_fixture() {
    // Exactly representable nonconstant inputs exercise channels, seams and
    // four-way corners in the independent upstream Python comparison.
    std::vector<std::vector<std::vector<float>>> tiles(2);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 2; ++col) {
            std::vector<float> values(48 * 16 * 16);
            for (std::size_t index = 0; index < values.size(); ++index)
                values[index] =
                    static_cast<float>(
                        static_cast<int32_t>((index * 13 + row * 37 + col * 101) % 251) - 125) /
                    32.0F;
            tiles[row].push_back(std::move(values));
        }
    }
    const auto output = trtmc::minimax_h3::stitch_visual_reference_moments(tiles, 272, 288, 1);
    std::cout.write(reinterpret_cast<const char*>(output.data()),
                    static_cast<std::streamsize>(output.size() * sizeof(float)));
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_axis();
        test_stitch();
        if (argc == 2 && std::string(argv[1]) == "--dump-upstream-fixture")
            dump_upstream_fixture();
        else if (argc != 1)
            throw std::invalid_argument("expected no arguments or --dump-upstream-fixture");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
