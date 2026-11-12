/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// E2E helper: run the runtime's host-side request preparation for one conversation and dump it
// for comparison with the provider processor.
//
//   minicpmv4_7_preprocess_dump BUNDLE REQUEST_JSON OUTPUT_DIR
//
// REQUEST_JSON: {"enable_thinking": bool, "messages": [{"role": "user", "parts": [
//                  {"rgb": "<raw HWC uint8 file>", "height": H, "width": W} | {"text": "..."}]}]}
// OUTPUT_DIR receives dump.json (prompt, input_ids, positions, crop grids and position ids) and
// patches.f32 (every crop's normalized patches in prompt order, [patches, 3 * p * p]).

#include "families/minicpmv4_7/runtime/image_preprocess.h"
#include "families/minicpmv4_7/runtime/prompt.h"
#include "families/minicpmv4_7/runtime/tokenizer.h"
#include "trtmc/bundle.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trtmc::minicpmv4_7;
using Json = nlohmann::json;

namespace {

std::string section(const trtmc::BundleReader& bundle, const char* name) {
    const auto data = bundle.read_section(name);
    return {data.begin(), data.end()};
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE REQUEST_JSON OUTPUT_DIR\n";
        return 2;
    }
    try {
        const trtmc::BundleReader bundle(argv[1]);
        const auto runtime = Json::parse(section(bundle, "runtime.json"));
        const auto tokenizer_json = bundle.read_section("tokenizer.json");
        const auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size());
        const std::filesystem::path request_path(argv[2]);
        const auto request = Json::parse(std::ifstream(request_path));
        const auto& vision = runtime.at("vision");
        ProcessorConfig processor;
        processor.patch_size = vision.at("patch_size");
        processor.patches_per_side = vision.at("patches_per_side");
        processor.scale_resolution = vision.at("scale_resolution");
        processor.max_slice_nums = vision.at("max_slice_nums");
        processor.max_patches = vision.at("max_patches");
        processor.mean = vision.at("image_mean").get<std::array<float, 3>>();
        processor.stddev = vision.at("image_std").get<std::array<float, 3>>();
        const auto& ids = runtime.at("token_ids");
        TokenIds tokens;
        tokens.image_pad = ids.at("image_pad");
        tokens.image_start = ids.at("image_start");
        tokens.image_end = ids.at("image_end");
        tokens.slice_start = ids.at("slice_start");
        tokens.slice_end = ids.at("slice_end");
        tokens.image_id_start = ids.at("image_id_start");
        tokens.image_id_end = ids.at("image_id_end");
        tokens.newline = ids.at("newline");

        std::vector<ChatMessage> messages;
        std::vector<ImageCrops> images;
        for (const auto& message : request.at("messages")) {
            const auto role = message.at("role").get<std::string>();
            ChatMessage item;
            item.role = role == "system" ? Role::System
                        : role == "user" ? Role::User
                                         : Role::Assistant;
            std::vector<std::string> parts;
            for (const auto& part : message.at("parts")) {
                if (part.contains("text")) {
                    parts.push_back(part.at("text").get<std::string>());
                    continue;
                }
                const auto pixels =
                    read_file(request_path.parent_path() / part.at("rgb").get<std::string>());
                const int32_t height = part.at("height"), width = part.at("width");
                if (pixels.size() != static_cast<std::size_t>(height) * width * 3)
                    throw std::runtime_error("raw RGB size disagrees with its shape");
                images.push_back(preprocess_image(pixels.data(), height, width, processor));
                parts.push_back(kImagePlaceholder);
            }
            item.content = join_parts(parts);
            messages.push_back(std::move(item));
        }
        const bool thinking = request.value("enable_thinking", false);
        const auto prompt =
            format_conversation(runtime.at("prompts").at(thinking ? "true" : "false"), messages);
        const auto expanded = expand_images(prompt, images, processor.merge_factor);
        const auto input_ids = tokenizer->encode(expanded);
        std::vector<std::array<int32_t, 2>> grids;
        Json crops = Json::array();
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        std::ofstream patches(output / "patches.f32", std::ios::binary);
        for (const auto& image : images)
            for (const auto& crop : image.crops) {
                grids.push_back({crop.grid_h, crop.grid_w});
                crops.push_back({{"grid", {crop.grid_h, crop.grid_w}},
                                 {"position_ids", crop.position_ids},
                                 {"window_index", crop.window_index},
                                 {"merge_index", crop.merge_index}});
                patches.write(reinterpret_cast<const char*>(crop.patches.data()),
                              static_cast<std::streamsize>(crop.patches.size() * sizeof(float)));
            }
        const auto canvas = canvas_positions(input_ids, tokens, grids, processor.merge_factor);
        Json dump{{"prompt", prompt},
                  {"input_ids", input_ids},
                  {"positions", canvas.positions},
                  {"decode_position", canvas.decode_position},
                  {"crops", crops},
                  {"slice_grids", Json::array()}};
        for (const auto& image : images)
            dump["slice_grids"].push_back({image.rows, image.cols});
        std::ofstream(output / "dump.json") << dump.dump();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "minicpmv4_7_preprocess_dump: " << error.what() << "\n";
        return 1;
    }
}
