/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <trtmc/text.hpp>

namespace {
void json_string(std::string_view value) {
    std::cout << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\')
            std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32)
            std::cout << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                      << static_cast<unsigned>(byte) << std::dec << std::setfill(' ');
        else
            std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " BUNDLE RUNTIME_ROOT TEXT [MAX_NEW_TOKENS]\n";
        return 2;
    }
    try {
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            const auto task = model.task<trtmc::ConditionalTextGeneration>();
            if (task.config_fields().size() != 3)
                throw std::runtime_error("bart must expose exactly 3 config fields");
            trtmc::RunConfig config;
            if (argc >= 5) {
                config.set("max_new_tokens", std::int64_t(std::stoll(argv[4])));
            }
            return task.run({trtmc::TextSource{std::string(argv[3])}}, config);
        }();
        std::cout << "{\"task\":\"conditional_text_generation\",\"text\":";
        json_string(result.text());
        std::cout << ",\"tokens\":[";
        const auto tokens = result.token_ids();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << tokens[i];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
