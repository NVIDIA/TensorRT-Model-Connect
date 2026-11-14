/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phonon/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
    using trtmc::phonon::Tokenizer;
    if (Tokenizer::prepare("  hello   world ") != "Hello world.")
        throw std::runtime_error("text preparation regression");
    for (const auto* text : {"", "   ", "hello; world", "h\xc3\xa9llo"}) {
        bool rejected = false;
        try {
            Tokenizer::prepare(text);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid text was accepted");
    }
    if (argc == 3) {
        std::ifstream file(argv[1]);
        std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        for (auto id : Tokenizer(data).encode(argv[2]))
            std::cout << id << ' ';
        std::cout << '\n';
    }
}
