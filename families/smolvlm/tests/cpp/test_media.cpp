/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Native image processor and prompt rendering against reference fixtures
// (tests/data/media_cases.json, prompt_cases.json; see tests/test_fixtures.py).
// Usage: test_smolvlm_media MEDIA_CASES.json PROMPT_CASES.json
#include "families/smolvlm/runtime/media.h"
#include "families/smolvlm/runtime/prompt.h"

#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::smolvlm::Rgb8Image;
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class Function>
void check_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    check(false, message + " did not throw");
}

// Same formula as tests/test_fixtures.py:pattern.
Rgb8Image pattern(int height, int width) {
    Rgb8Image image{height, width, std::vector<uint8_t>(size_t(height) * width * 3)};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                image.pixels[(size_t(y) * width + x) * 3 + c] =
                    uint8_t((y * 37 + x * 11 + c * 83 + (x * y % 13) * 7) % 256);
    return image;
}

trtmc::smolvlm::ImageProcessing processing(int longest_edge, int tile) {
    trtmc::smolvlm::ImageProcessing p;
    p.longest_edge = longest_edge;
    p.tile = tile;
    p.rescale_factor = 0.00392156862745098;
    p.mean = {0.5F, 0.5F, 0.5F};
    p.std = {0.5F, 0.5F, 0.5F};
    p.image_seq_len = 64;
    return p;
}

nlohmann::json read(const char* path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(std::string("cannot read ") + path);
    nlohmann::json document;
    input >> document;
    return document;
}

void test_media(const nlohmann::json& cases) {
    for (const auto& item : cases.at("resize")) {
        const auto out =
            trtmc::smolvlm::resize_lanczos(pattern(item.at("height"), item.at("width")),
                                           item.at("out_height"), item.at("out_width"));
        const auto expected = item.at("expected").get<std::vector<int>>();
        bool same = out.pixels.size() == expected.size();
        for (size_t i = 0; same && i < expected.size(); ++i)
            same = out.pixels[i] == expected[i];
        check(same, "Lanczos " + item.dump().substr(0, 80));
    }
    for (const auto& item : cases.at("geometry")) {
        const int height = item.at("height"), width = item.at("width");
        const auto resized = trtmc::smolvlm::resize_to_longest_edge(height, width, 2048);
        const auto tiled = trtmc::smolvlm::resize_for_tiles(resized.first, resized.second, 512);
        check(resized.first == item.at("resized")[0] && resized.second == item.at("resized")[1] &&
                  tiled.first == item.at("tiled")[0] && tiled.second == item.at("tiled")[1],
              "geometry " + std::to_string(height) + "x" + std::to_string(width));
    }
    const auto& pipeline = cases.at("pipeline");
    const auto tiled =
        trtmc::smolvlm::tile_image(pattern(pipeline.at("height"), pipeline.at("width")),
                                   processing(pipeline.at("longest_edge"), pipeline.at("tile")));
    const auto expected = pipeline.at("pixels").get<std::vector<double>>();
    bool same = tiled.rows == pipeline.at("rows") && tiled.cols == pipeline.at("cols") &&
                tiled.frames == pipeline.at("frames") && tiled.pixels.size() == expected.size();
    for (size_t i = 0; same && i < expected.size(); ++i)
        same = tiled.pixels[i] == static_cast<float>(expected[i]);
    check(same, "tiling pipeline (bit-exact normalized tiles)");
}

void test_inputs() {
    // Float32 x / 255 inputs round back to the same 8-bit image.
    const auto source = pattern(5, 7);
    std::vector<float> scaled(source.pixels.size());
    for (size_t i = 0; i < scaled.size(); ++i)
        scaled[i] = float(source.pixels[i]) / 255.0F;
    trtmc::internal::ImageView view{scaled.data(),
                                    scaled.size() * sizeof(float),
                                    5,
                                    7,
                                    3,
                                    trtmc::internal::ImageFormat::Float32};
    check(trtmc::smolvlm::to_rgb8(view).pixels == source.pixels, "Float32 [0, 1] round trip");
    trtmc::internal::ImageView bytes{
        source.pixels.data(), source.pixels.size(), 5, 7, 3, trtmc::internal::ImageFormat::UInt8};
    check(trtmc::smolvlm::to_rgb8(bytes).pixels == source.pixels, "UInt8 copy");
    auto rgba = bytes;
    rgba.channels = 4;
    check_throws([&] { trtmc::smolvlm::to_rgb8(rgba); }, "4-channel input");
    auto short_view = bytes;
    short_view.byte_size -= 1;
    check_throws([&] { trtmc::smolvlm::to_rgb8(short_view); }, "short UInt8 buffer");
    scaled[3] = 1.5F;
    check_throws([&] { trtmc::smolvlm::to_rgb8(view); }, "Float32 pixel above 1");
    check_throws([&] { trtmc::smolvlm::resize_lanczos(source, 0, 3); }, "empty resize target");
}

void test_prompts(const nlohmann::json& cases) {
    using trtmc::internal::ImagesTextMessage;
    using trtmc::internal::ImagesTextPart;
    using trtmc::internal::MessageRole;
    for (const auto& item : cases) {
        std::deque<std::vector<ImagesTextPart>> parts;
        std::deque<std::string> texts;
        std::vector<ImagesTextMessage> messages;
        std::vector<std::pair<int, int>> grids;
        for (const auto& message : item.at("messages")) {
            parts.emplace_back();
            for (const auto& part : message.at("content")) {
                if (part.at("type") == "text") {
                    texts.push_back(part.at("text"));
                    parts.back().emplace_back(trtmc::internal::TextPartView{texts.back()});
                } else {
                    grids.emplace_back(part.at("rows"), part.at("cols"));
                    parts.back().emplace_back(trtmc::internal::ImageView{});
                }
            }
            const std::string role = message.at("role");
            messages.push_back({role == "system"      ? MessageRole::System
                                : role == "assistant" ? MessageRole::Assistant
                                                      : MessageRole::User,
                                {parts.back().data(), parts.back().size()}});
        }
        size_t next = 0;
        const auto rendered = trtmc::smolvlm::render_chat(
            {messages.data(), messages.size()}, [&](const trtmc::internal::ImageView&) {
                const auto grid = grids.at(next++);
                return trtmc::smolvlm::image_prompt(grid.first, grid.second,
                                                    item.at("image_seq_len"));
            });
        check(rendered == item.at("expected").get<std::string>() && next == grids.size(),
              "chat template " + item.at("messages").dump().substr(0, 80));
    }
    std::vector<ImagesTextPart> injected{trtmc::internal::TextPartView{"look <image> here"}};
    ImagesTextMessage message{MessageRole::User, {injected.data(), injected.size()}};
    check_throws(
        [&] { trtmc::smolvlm::render_chat({&message, 1}, [](const auto&) { return ""; }); },
        "literal <image> in text");
    std::vector<ImagesTextPart> none;
    ImagesTextMessage empty{MessageRole::User, {none.data(), none.size()}};
    check_throws([&] { trtmc::smolvlm::render_chat({&empty, 1}, [](const auto&) { return ""; }); },
                 "empty message");
    ImagesTextMessage tool{MessageRole::Tool, {injected.data(), injected.size()}};
    check_throws([&] { trtmc::smolvlm::render_chat({&tool, 1}, [](const auto&) { return ""; }); },
                 "tool role");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_smolvlm_media MEDIA_CASES.json PROMPT_CASES.json\n";
        return 2;
    }
    try {
        test_media(read(argv[1]));
        test_inputs();
        test_prompts(read(argv[2]));
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures)
        return 1;
    std::cout << "test_smolvlm_media passed\n";
    return 0;
}
