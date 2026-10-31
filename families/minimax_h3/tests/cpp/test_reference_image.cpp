/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_image.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using namespace trtmc::minimax_h3;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void rejected(Function function) {
    bool failed = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed, "malformed reference input must be rejected");
}

std::vector<uint8_t> fixture(int32_t height, int32_t width, int32_t pattern) {
    std::vector<uint8_t> result(static_cast<std::size_t>(height) * width * 3);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto y = i / (static_cast<std::size_t>(width) * 3);
        const auto x = i / 3 % width;
        if (pattern == 0)
            result[i] = static_cast<uint8_t>((i * 37 + i / 7 * 13) % 256);
        else if (pattern == 1)
            result[i] = (x + y) % 2 ? 255 : 0;
        else if (pattern == 2)
            result[i] = x == static_cast<std::size_t>(width / 2) &&
                                y == static_cast<std::size_t>(height / 2)
                            ? 255
                            : 0;
        else
            result[i] = 173;
    }
    return result;
}

void geometry() {
    const auto portrait = reference_image_shape(880, 1184);
    require(portrait.height == 2048 && portrait.width == 2752, "original sample geometry");
    const auto limit = reference_image_shape(1, 4);
    require(limit.height == 2048 && limit.width == 8192, "no image-reference area reduction");
    const auto even = reference_image_shape(128, 129);
    const auto odd = reference_image_shape(128, 131);
    require(even.width == 2048 && odd.width == 2112, "Python ties-to-even, not lround");
    const auto large = reference_image_shape(std::numeric_limits<int32_t>::max(),
                                             std::numeric_limits<int32_t>::max());
    require(large.height == 2048 && large.width == 2048, "wide geometry arithmetic");
    for (const auto shape : {ReferenceImageShape{0, 4}, {-1, 4}, {1, 5}, {5, 1}})
        rejected([&] { (void)reference_image_shape(shape.height, shape.width); });
}

void pixels() {
    auto rgb = fixture(32, 48, 0);
    require(resize_reference_rgb_lanczos(rgb.data(), rgb.size(), 32, 48, {32, 48}) == rgb,
            "identity resize preserves exact RGB bytes");
    const auto flat = fixture(4, 7, 3);
    const auto expanded = resize_reference_rgb_lanczos(flat.data(), flat.size(), 4, 7, {37, 93});
    require(std::all_of(expanded.begin(), expanded.end(), [](uint8_t x) { return x == 173; }),
            "constant image remains constant");
    rejected([&] { (void)resize_reference_rgb_lanczos(nullptr, rgb.size(), 32, 48, {64, 64}); });
    rejected(
        [&] { (void)resize_reference_rgb_lanczos(rgb.data(), rgb.size() - 1, 32, 48, {64, 64}); });
    rejected([&] { (void)resize_reference_rgb_lanczos(rgb.data(), rgb.size(), 32, 48, {0, 64}); });
    PreparedReferenceImage image{{64, 96}, fixture(64, 96, 0)};
    const auto tile = pack_visual_reference_rgb_tile(image, 16, 32, 32, 48);
    require(tile.size() == 3 * 32 * 48, "one image is one VAE temporal frame");
    require(std::isfinite(tile.front()) && tile.front() != -1.0F,
            "ImageNet, not Qwen normalization");
    for (const int32_t top : {-16, 1, 48})
        rejected([&] { (void)pack_visual_reference_rgb_tile(image, top, 0, 32, 32); });
    rejected([&] { (void)pack_visual_reference_rgb_tile(image, 0, 0, 16, 32); });
    image.rgb.pop_back();
    rejected([&] { (void)pack_visual_reference_rgb_tile(image, 0, 0, 32, 32); });
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            geometry();
            pixels();
        } else if (argc == 7 && std::string(argv[1]) == "--resize-fixture") {
            const int32_t h = std::stoi(argv[2]), w = std::stoi(argv[3]);
            const ReferenceImageShape target{std::stoi(argv[4]), std::stoi(argv[5])};
            require(h > 0 && w > 0 && h <= 8192 && w <= 8192, "fixture dimensions");
            const auto rgb = fixture(h, w, std::stoi(argv[6]));
            const auto result = resize_reference_rgb_lanczos(rgb.data(), rgb.size(), h, w, target);
            std::cout.write(reinterpret_cast<const char*>(result.data()),
                            static_cast<std::streamsize>(result.size()));
        } else if (argc == 5 && std::string(argv[1]) == "--prepare-fixture") {
            const int32_t h = std::stoi(argv[2]), w = std::stoi(argv[3]);
            require(h > 0 && w > 0 && h <= 8192 && w <= 8192, "fixture dimensions");
            const auto rgb = fixture(h, w, std::stoi(argv[4]));
            const auto prepared = prepare_reference_image(rgb.data(), rgb.size(), h, w);
            const int32_t shape[] = {prepared.shape.height, prepared.shape.width};
            std::cout.write(reinterpret_cast<const char*>(shape), sizeof(shape));
            std::cout.write(reinterpret_cast<const char*>(prepared.rgb.data()),
                            static_cast<std::streamsize>(prepared.rgb.size()));
        } else if (argc == 2 && std::string(argv[1]) == "--tile-fixture") {
            const PreparedReferenceImage image{{64, 96}, fixture(64, 96, 0)};
            const auto tile = pack_visual_reference_rgb_tile(image, 16, 32, 32, 48);
            std::cout.write(reinterpret_cast<const char*>(tile.data()),
                            static_cast<std::streamsize>(tile.size() * sizeof(float)));
        } else {
            throw std::invalid_argument(
                "expected no arguments, --resize-fixture H W OH OW PATTERN, "
                "--prepare-fixture H W PATTERN, or --tile-fixture");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
