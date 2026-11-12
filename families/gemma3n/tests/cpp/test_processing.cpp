/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Source-only checks of the Gemma 3n host processing against goldens produced
// by the reference stack (tests/data/goldens.json): the tokenizer pipeline on a
// small fixture, the antialiased bilinear resize, and chat-template rendering.

#include "families/gemma3n/runtime/image.h"
#include "families/gemma3n/runtime/prompt.h"
#include "families/gemma3n/runtime/sampler.h"
#include "families/gemma3n/runtime/tokenizer.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace trtmc;
using namespace trtmc::gemma3n;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << label << '\n';
    }
}

void expect_throws(const std::function<void()>& body, const std::string& label) {
    try {
        body();
    } catch (const std::exception&) {
        return;
    }
    expect(false, label + " did not throw");
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::vector<uint8_t> synthetic(int height, int width) {
    std::vector<uint8_t> pixels(std::size_t(height) * std::size_t(width) * 3U);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                pixels[(std::size_t(y) * std::size_t(width) + std::size_t(x)) * 3U +
                       std::size_t(c)] =
                    static_cast<uint8_t>((x * 7 + y * 13 + c * 31 + (x * y) % 17) % 256);
    return pixels;
}

void check_tokenizer(const nlohmann::json& goldens, const std::string& data) {
    const Tokenizer tokenizer(read_file(data + "/tokenizer_fixture.json"));
    for (const auto& golden : goldens) {
        const auto ids = golden.at("ids").get<std::vector<int32_t>>();
        if (!golden.at("text").is_null()) {
            const auto text = golden.at("text").get<std::string>();
            expect(tokenizer.encode(text) == ids, "encode " + text);
        }
        expect(tokenizer.decode(ids) == golden.at("decoded").get<std::string>(),
               "decode " + golden.at("decoded").get<std::string>());
    }
    expect(tokenizer.token_id("<image_soft_token>") >= 0, "token_id");
    expect_throws([&] { tokenizer.token_id("<missing>"); }, "unknown token");
}

void check_resize(const nlohmann::json& goldens) {
    for (const auto& golden : goldens) {
        const int height = golden.at("height"), width = golden.at("width");
        const int out_height = golden.at("out_height"), out_width = golden.at("out_width");
        const auto resized =
            resize_bilinear_aa(synthetic(height, width), height, width, out_height, out_width);
        const auto label = "resize " + std::to_string(height) + "x" + std::to_string(width) +
                           " -> " + std::to_string(out_height) + "x" + std::to_string(out_width);
        if (golden.contains("pixels"))
            expect(resized == golden.at("pixels").get<std::vector<uint8_t>>(), label);
        int64_t sum = 0;
        for (const auto value : resized)
            sum += value;
        expect(sum == golden.at("sum").get<int64_t>(), label + " sum");
        for (const auto& sample : golden.at("samples"))
            expect(resized.at(sample.at(0).get<std::size_t>()) == sample.at(1).get<int>(),
                   label + " sample");
    }
}

void check_pixels() {
    // Float32 pixels from 8-bit sources recover the exact 8-bit values.
    std::vector<float> levels(256 * 3);
    for (int i = 0; i < 256 * 3; ++i)
        levels[std::size_t(i)] = static_cast<float>(i / 3) / 255.0F;
    internal::ImageView image{
        levels.data(), levels.size() * sizeof(float), 1, 256, 3, internal::ImageFormat::Float32};
    const auto rgb = to_rgb8(image);
    bool exact = true;
    for (int i = 0; i < 256 * 3; ++i)
        exact = exact && rgb.pixels[std::size_t(i)] == i / 3;
    expect(exact, "float32 to uint8 levels");
    const auto values = pixel_values(Rgb8{std::vector<uint8_t>(16 * 16 * 3, 255), 16, 16}, 8,
                                     static_cast<float>(1.0 / 255.0));
    expect(values.size() == 3U * 8U * 8U && values[0] == 1.0F, "rescaled planar pixels");
    levels[0] = 1.5F;
    expect_throws([&] { to_rgb8(image); }, "float32 out of range");
    internal::ImageView rgba{
        levels.data(), levels.size() * sizeof(float), 1, 192, 4, internal::ImageFormat::Float32};
    expect_throws([&] { to_rgb8(rgba); }, "four channels");
}

internal::MessageRole role(const std::string& name) {
    if (name == "system")
        return internal::MessageRole::System;
    if (name == "assistant")
        return internal::MessageRole::Assistant;
    return internal::MessageRole::User;
}

// Renders one conversation given in the reference processor's message format
// ({"role", "content": [{"type": "text", "text"} | {"type": "image"}]}).
std::string render_conversation(const nlohmann::json& conversation) {
    static const uint8_t pixel[3] = {1, 2, 3};
    static const internal::ImageView image{pixel, 3, 1, 1, 3, internal::ImageFormat::UInt8};
    std::vector<std::string> texts;
    for (const auto& message : conversation)
        for (const auto& item : message.at("content"))
            if (item.at("type") == "text")
                texts.push_back(item.at("text").get<std::string>());
    std::vector<std::vector<internal::ImagesTextPart>> parts;
    std::size_t next_text = 0;
    for (const auto& message : conversation) {
        parts.emplace_back();
        for (const auto& item : message.at("content")) {
            if (item.at("type") == "text")
                parts.back().emplace_back(internal::TextPartView{texts[next_text++]});
            else
                parts.back().emplace_back(image);
        }
    }
    std::vector<internal::ImagesTextMessage> messages;
    for (std::size_t index = 0; index < parts.size(); ++index)
        messages.push_back(
            {role(conversation[index].at("role")), {parts[index].data(), parts[index].size()}});
    return render_prompt({messages.data(), messages.size()}, image_sequence(256)).text;
}

void check_prompts(const nlohmann::json& goldens) {
    const uint8_t pixel[3] = {1, 2, 3};
    const internal::ImageView image{pixel, 3, 1, 1, 3, internal::ImageFormat::UInt8};
    const auto sequence = image_sequence(256);
    for (const auto& golden : goldens)
        expect(render_conversation(golden.at("messages")) ==
                   golden.at("rendered").get<std::string>(),
               "rendered prompt");
    std::vector<internal::ImagesTextPart> user{internal::TextPartView{"hi <image_soft_token>"}};
    std::vector<internal::ImagesTextMessage> literal{
        {internal::MessageRole::User, {user.data(), 1}}};
    expect_throws([&] { render_prompt({literal.data(), 1}, sequence); }, "literal placeholder");
    std::vector<internal::ImagesTextPart> text{internal::TextPartView{"hi"}};
    std::vector<internal::ImagesTextMessage> swapped{
        {internal::MessageRole::Assistant, {text.data(), 1}}};
    expect_throws([&] { render_prompt({swapped.data(), 1}, sequence); }, "assistant first");
    std::vector<internal::ImagesTextPart> system_image{image};
    std::vector<internal::ImagesTextMessage> system{
        {internal::MessageRole::System, {system_image.data(), 1}},
        {internal::MessageRole::User, {text.data(), 1}}};
    expect_throws([&] { render_prompt({system.data(), 2}, sequence); }, "system image");
    expect(strip("\xE3\x80\x80 a b\t\xC2\xA0") == "a b", "unicode strip");
}

void check_sampler() {
    const std::vector<float> logits{0.5F, 2.0F, -1.0F, 2.0F, 1.0F};
    Sampler greedy(Generation{});
    expect(greedy.next(logits) == 1, "greedy takes the first maximum");
    Generation top1;
    top1.temperature = 1.0;
    top1.top_k = 1;
    top1.seed = 7;
    Sampler first(top1);
    bool argmax = true;
    for (int i = 0; i < 16; ++i)
        argmax = argmax && first.next(logits) == 1;
    expect(argmax, "top_k 1 keeps the first maximum");
    Generation sampled;
    sampled.temperature = 0.8;
    sampled.top_p = 0.9;
    sampled.seed = 268;
    Sampler a(sampled), b(sampled);
    bool same = true;
    for (int i = 0; i < 32; ++i)
        same = same && a.next(logits) == b.next(logits);
    expect(same, "a seed fixes the sampled sequence");
    Generation nucleus;
    nucleus.temperature = 1.0;
    nucleus.top_p = 1e-6;
    nucleus.seed = 3;
    Sampler narrow(nucleus);
    expect(narrow.next(logits) == 1, "a tiny nucleus keeps only the most likely token");
    const std::array<internal::ConfigEntry, 1> bad{{{"top_p", 0.0}}};
    expect_throws([&] { read_generation({bad.data(), bad.size()}); }, "top_p 0");
    const std::array<internal::ConfigEntry, 1> unknown{{{"min_p", 0.1}}};
    expect_throws([&] { read_generation({unknown.data(), unknown.size()}); }, "unknown field");
}
} // namespace

// With --prompt-ids TOKENIZER CONVERSATIONS, print the native prompt token ids
// of each conversation as JSON (the family E2E compares them with the
// reference processor on the checkpoint's tokenizer). Without arguments, run
// the source-only checks.
int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--prompt-ids") {
            const Tokenizer tokenizer(read_file(argv[2]));
            auto ids = nlohmann::json::array();
            for (const auto& conversation : nlohmann::json::parse(read_file(argv[3])))
                ids.push_back(tokenizer.encode(render_conversation(conversation)));
            std::cout << ids.dump() << '\n';
            return 0;
        }
        const std::string data = GEMMA3N_TEST_DATA;
        const auto goldens = nlohmann::json::parse(read_file(data + "/goldens.json"));
        check_tokenizer(goldens.at("tokenizer"), data);
        check_resize(goldens.at("resize"));
        check_pixels();
        check_prompts(goldens.at("prompts"));
        check_sampler();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0)
        return 1;
    std::cout << "gemma3n processing checks passed\n";
    return 0;
}
