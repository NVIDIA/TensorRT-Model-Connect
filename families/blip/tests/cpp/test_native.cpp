/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/image_preprocess.h"
#include "families/blip/runtime/tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/runtime/family_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 6)
            throw std::invalid_argument(
                "expected MODE BUNDLE_OR_TOKENIZER RUNTIME_OR_OUTPUT IMAGE PREFIX");
        int width = 0, height = 0, channels = 0;
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
            stbi_load(argv[4], &width, &height, &channels, 3), &stbi_image_free);
        if (!image)
            throw std::runtime_error("image could not be read");
        const auto count = static_cast<std::size_t>(width) * height * 3;
        if (std::string(argv[1]) == "preprocess") {
            std::ifstream stream(argv[2], std::ios::binary);
            const std::string tokenizer_data{std::istreambuf_iterator<char>(stream), {}};
            auto tokenizer =
                trtmc::blip::CreateWordPieceTokenizer(tokenizer_data.data(), tokenizer_data.size());
            const std::vector<std::string> texts{"", "a photography of", "Hello, WORLD!",
                                                 "Caf\xc3\xa9 isn't a dog.", "[MASK] beach"};
            nlohmann::json tokens = nlohmann::json::array();
            for (const auto& text : texts) {
                const auto ids = tokenizer->encode(text);
                tokens.push_back(
                    {{"text", text}, {"ids", ids}, {"decoded", tokenizer->decode(ids)}});
            }
            trtmc::blip::BlipPreprocessConfig cfg;
            cfg.input_image_h = cfg.input_image_w = 384;
            cfg.image_mean = {0.48145466F, 0.4578275F, 0.40821073F};
            cfg.image_std = {0.26862954F, 0.26130258F, 0.27577711F};
            std::vector<float> source(count);
            for (std::size_t i = 0; i < count; ++i)
                source[i] = image.get()[i] / 255.0F;
            const auto output =
                trtmc::blip::preprocess_blip_image(source.data(), height, width, cfg);
            std::ofstream destination(argv[3], std::ios::binary);
            destination.write(reinterpret_cast<const char*>(output.data()),
                              output.size() * sizeof(float));
            if (!destination)
                throw std::runtime_error("failed to write preprocessing output");
            std::cout << tokens.dump() << '\n';
            return 0;
        }
        if (std::string(argv[1]) != "inference")
            throw std::invalid_argument("unknown test mode");
        auto task = trtmc::load_task(argv[2], argv[3]);
        auto* caption = dynamic_cast<trtmc::internal::IImagesTextToText*>(task.get());
        if (!caption)
            throw std::runtime_error("loaded family does not implement image captioning");
        trtmc::internal::ImageView view{image.get(),
                                        count,
                                        static_cast<std::uint32_t>(height),
                                        static_cast<std::uint32_t>(width),
                                        3,
                                        trtmc::internal::ImageFormat::UInt8};
        std::vector<trtmc::internal::ImagesTextPart> parts{view,
                                                           trtmc::internal::TextPartView{argv[5]}};
        trtmc::internal::ImagesTextMessage message{trtmc::internal::MessageRole::User,
                                                   {parts.data(), parts.size()}};
        trtmc::internal::ImagesTextToTextRequest request{{&message, 1}};
        trtmc::internal::ConfigEntry entry{"max_new_tokens", std::int64_t{32}};
        const auto first = caption->run(request, {&entry, 1});
        auto rejects = [&](auto callback) {
            try {
                callback();
            } catch (const std::invalid_argument&) {
                return;
            }
            throw std::runtime_error("invalid request was accepted");
        };
        entry.value = std::int64_t{-1};
        rejects([&] { caption->run(request, {&entry, 1}); });
        entry.value = std::int64_t{32};
        rejects([&] { caption->run({}, {&entry, 1}); });
        message.role = trtmc::internal::MessageRole::Assistant;
        rejects([&] { caption->run(request, {&entry, 1}); });
        message.role = trtmc::internal::MessageRole::User;
        parts[1] = view;
        rejects([&] { caption->run(request, {&entry, 1}); });
        parts[1] = trtmc::internal::TextPartView{"a picture of"};
        caption->run(request, {&entry, 1});
        parts[1] = trtmc::internal::TextPartView{argv[5]};
        const auto again = caption->run(request, {&entry, 1});
        if (first.token_ids != again.token_ids || first.text != again.text)
            throw std::runtime_error("caption leaked state across requests");
        std::cout << nlohmann::json{{"text", first.text},
                                    {"token_ids", first.token_ids},
                                    {"repeated_request", "passed"},
                                    {"invalid_requests", 4}}
                         .dump()
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
