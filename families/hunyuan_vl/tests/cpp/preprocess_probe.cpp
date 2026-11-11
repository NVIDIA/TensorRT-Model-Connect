/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Dumps the family's host preprocessing for the Python parity test against the reference processor.
//   image <rgb> <height> <width> <runtime.json> <out-prefix>   -> <out>.f32 pixels, <out>.json grid
//   tokens <tokenizer.json> <texts.json> <out.json>
#include "families/hunyuan_vl/runtime/media.h"
#include "families/hunyuan_vl/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
std::string read(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("cannot read ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

int image(char** argv) {
    using namespace trtmc::hunyuan_vl;
    const auto pixels = read(argv[2]);
    const auto metadata = nlohmann::json::parse(read(argv[5]));
    ImageSettings settings;
    settings.patch = metadata.at("patch_size");
    settings.merge = metadata.at("merge_size");
    settings.min_pixels = metadata.at("image").at("min_pixels");
    settings.max_pixels = metadata.at("image").at("max_pixels");
    settings.mean = metadata.at("image").at("mean").get<std::array<float, 3>>();
    settings.std = metadata.at("image").at("std").get<std::array<float, 3>>();
    settings.rescale_factor = metadata.at("image").at("rescale_factor");
    trtmc::internal::ImageView view{pixels.data(),
                                    pixels.size(),
                                    uint32_t(std::stoul(argv[3])),
                                    uint32_t(std::stoul(argv[4])),
                                    3,
                                    trtmc::internal::ImageFormat::UInt8};
    const auto plan = plan_image(view, settings, metadata.at("max_patches"));
    const auto values = patchify(plan, settings);
    std::vector<int32_t> ids;
    std::vector<float> lambdas;
    position_corners(plan, metadata.at("position_edge"), ids, lambdas);
    const std::string prefix = argv[6];
    std::ofstream(prefix + ".f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * 4));
    std::ofstream(prefix + ".json") << nlohmann::json{
        {"height", plan.height},      {"width", plan.width},     {"rows", plan.rows()},
        {"cols", plan.cols()},        {"tokens", plan.tokens()}, {"position_ids", ids},
        {"position_lambdas", lambdas}}.dump();
    return 0;
}

int tokens(char** argv) {
    const trtmc::hunyuan_vl::Tokenizer tokenizer(read(argv[2]));
    nlohmann::json result = nlohmann::json::array();
    for (const auto& text : nlohmann::json::parse(read(argv[3]))) {
        const auto ids = tokenizer.encode(text.get<std::string>());
        result.push_back({{"ids", ids}, {"decoded", tokenizer.decode(ids)}});
    }
    std::ofstream(argv[4]) << result.dump();
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string(argv[1]) == "image")
            return image(argv);
        if (argc == 5 && std::string(argv[1]) == "tokens")
            return tokens(argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cerr << "usage: see the header of preprocess_probe.cpp\n";
    return 2;
}
