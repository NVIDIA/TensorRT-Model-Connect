/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/video_decode_math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace trtmc::minimax_h3;
constexpr std::size_t kTileValues = 3 * 28 * 256 * 256;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Fn>
void must_throw(Fn&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "expected validation failure");
}

DecodedH3Video fixture(int32_t frames, int32_t height, int32_t width) {
    const auto g = make_video_decode_geometry(frames, height, width);
    VideoClipAssembler assembler(g);
    std::vector<float> tile(kTileValues);
    for (int32_t clip = 0; clip < g.clips; ++clip)
        for (std::size_t item = 0; item < g.tiles(); ++item) {
            for (std::size_t i = 0; i < tile.size(); ++i)
                tile[i] = static_cast<float>(
                              static_cast<int32_t>((i * 13 + item * 37 + clip * 101) % 251) - 125) /
                          32.0F;
            assembler.append(clip, item, tile.data(), tile.size());
        }
    return assembler.finish();
}

void test_rounding() {
    for (uint32_t i = 0; i < 65536; ++i) {
        const auto half = static_cast<uint16_t>(i);
        const float value = video_half_to_float(half);
        if ((half & 0x7c00U) == 0x7c00U && (half & 0x3ffU))
            require(std::isnan(value), "all half NaNs widen to NaN");
        else
            require(video_float_to_half(value) == half,
                    "all finite/inf/signed-zero half values round-trip");
    }
    require(video_float_to_half(1.0F + std::ldexp(1.0F, -11)) == 0x3c00,
            "even half midpoint rounds down");
    require(video_float_to_half(1.0F + 3 * std::ldexp(1.0F, -11)) == 0x3c02,
            "odd half midpoint rounds up");
    require(video_float_to_half(std::ldexp(1.0F, -25)) == 0, "subnormal midpoint rounds to zero");
    require(video_float_to_half(3 * std::ldexp(1.0F, -25)) == 2,
            "subnormal odd midpoint rounds up");
    require(video_float_to_half(65504.0F) == 0x7bff && video_float_to_half(65520.0F) == 0x7c00,
            "half overflow boundary");
}

void test_geometry_and_validation() {
    const auto portrait = make_video_decode_geometry(37, 84, 48);
    const auto landscape = make_video_decode_geometry(37, 48, 84);
    require(portrait.frames == 124 && portrait.clips == 7 && portrait.pad_tokens == 0 &&
                portrait.tiles() == 28,
            "full original portrait geometry");
    require(portrait.vertical.starts == landscape.horizontal.starts &&
                portrait.horizontal.starts == landscape.vertical.starts,
            "portrait is a different grid, not a tile reshape");
    const auto master = make_video_decode_geometry(37, 96, 54);
    require(master.height == 1536 && master.width == 864 && master.tiles() == 40,
            "exact 9:16 without reducing the 768-pixel short edge");
    must_throw([] { make_video_decode_geometry(2, 16, 16); });
    must_throw([] { make_video_decode_geometry(37, 17, 16); });
    must_throw([] { make_video_decode_geometry(37, 16, 15); });
    auto g = make_video_decode_geometry(3, 16, 16);
    require(g.frames == 9 && g.pad_tokens == 4 && g.clips == 1, "original repeated-tail crop");
    VideoClipAssembler incomplete(g);
    must_throw([&] { incomplete.finish(); });
    std::vector<float> tile(kTileValues, 0.0F);
    must_throw([&] { incomplete.append(1, 0, tile.data(), tile.size()); });
    tile[0] = 1.0F + std::ldexp(1.0F, -11);
    must_throw([&] { incomplete.append(0, 0, tile.data(), tile.size()); });
    must_throw([&] { incomplete.finish(); });
    const auto result = fixture(3, 16, 16);
    require(result.frames == 9 && result.height == 256 && result.width == 256 &&
                result.rgb.size() == 9 * 256 * 256 * 3,
            "owned THWC frame shape");
    require(std::all_of(result.rgb.begin(), result.rgb.end(),
                        [](float x) { return std::isfinite(x) && x >= 0 && x <= 1; }),
            "ImageNet denormalization/clamp");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--float-round") {
            float value;
            while (std::cin.read(reinterpret_cast<char*>(&value), sizeof(value))) {
                const uint16_t out = trtmc::minimax_h3::video_float_to_half(value);
                std::cout.write(reinterpret_cast<const char*>(&out), sizeof(out));
            }
            if (!std::cin.eof() || std::cin.gcount() != 0)
                throw std::runtime_error("truncated FP32 fixture");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--half-widen") {
            uint16_t value;
            while (std::cin.read(reinterpret_cast<char*>(&value), sizeof(value))) {
                const float out = trtmc::minimax_h3::video_half_to_float(value);
                std::cout.write(reinterpret_cast<const char*>(&out), sizeof(out));
            }
            if (!std::cin.eof() || std::cin.gcount() != 0)
                throw std::runtime_error("truncated FP16 fixture");
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--assemble") {
            const auto result = fixture(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
            std::cout.write(reinterpret_cast<const char*>(result.rgb.data()),
                            static_cast<std::streamsize>(result.rgb.size() * sizeof(float)));
            return std::cout ? 0 : 1;
        }
        if (argc != 1)
            throw std::invalid_argument("invalid video assembly fixture arguments");
        test_rounding();
        test_geometry_and_validation();
        std::cout << "MiniMax-H3 video decode math checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
