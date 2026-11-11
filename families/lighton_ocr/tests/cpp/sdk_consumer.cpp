/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/tests/cpp/image_io.h"
#include "trtmc/language.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 6)
        return 2;
    try {
        int width = 0, height = 0;
        std::unique_ptr<unsigned char, decltype(&lighton_ocr_free_rgb)> pixels(
            lighton_ocr_load_rgb(argv[3], &width, &height), lighton_ocr_free_rgb);
        if (!pixels)
            throw std::runtime_error("Cannot read SDK RGB image");
        std::ifstream input(argv[4]);
        nlohmann::json request;
        input >> request;
        trtmc::Config config{{"max_new_tokens", request.value("max_new_tokens", 1024)}};
        std::vector<trtmc::ImagesTextPart> parts{
            trtmc::ImageInput{{pixels.get(), static_cast<std::size_t>(height) * width * 3},
                              static_cast<std::uint32_t>(height),
                              static_cast<std::uint32_t>(width)}};
        const auto media = trtmc::ImagesTextToTextRequest::from_parts(parts);
        auto result = [&]() {
            const auto model = trtmc::Model::load(argv[1], {argv[2]});
            const auto task = model.task<trtmc::ImagesTextToText>();
            auto retained = task.run(media, config);
            const auto original = retained.token_ids();
            const std::vector<std::int32_t> ids(original.begin(), original.end());
            const std::string text(retained.text());
            bool rejected = false;
            try {
                auto invalid = parts;
                invalid.emplace_back(trtmc::TextPart{"Unsupported instruction"});
                task.run(trtmc::ImagesTextToTextRequest::from_parts(std::move(invalid)), config);
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("SDK accepted an unsupported instruction");
            const auto repeated = task.run(media, config);
            const auto repeated_ids = repeated.token_ids();
            if (std::string(repeated.text()) != text ||
                std::vector<std::int32_t>(repeated_ids.begin(), repeated_ids.end()) != ids ||
                std::string(retained.text()) != text)
                throw std::runtime_error("SDK request reuse or retained result changed generation");
            return retained;
        }();
        const auto retained_ids = result.token_ids();
        const std::vector<std::int32_t> ids(retained_ids.begin(), retained_ids.end());
        const std::string text(result.text());
        nlohmann::json output{{"text", text},
                              {"token_ids", ids},
                              {"prefill_ms", result.prefill_ms()},
                              {"decode_ms", result.decode_ms()},
                              {"repeat_matches", true},
                              {"rejection_recovery", true},
                              {"result_retained_after_model_release", true}};
        std::ofstream(argv[5]) << output.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
