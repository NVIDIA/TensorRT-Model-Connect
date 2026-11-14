/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* A direct header-only public SDK caller. No family or application headers. */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/trtmc.hpp>
#include <vector>

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

trtmc::ConfigValue value(const std::string& input, trtmc::ConfigKind kind) {
    std::size_t used = 0;
    if (kind == trtmc::ConfigKind::String)
        return input;
    if (kind == trtmc::ConfigKind::Bool) {
        if (input != "true" && input != "false")
            throw std::invalid_argument("expected a boolean Config value");
        return input == "true";
    }
    if (kind == trtmc::ConfigKind::I64) {
        const auto result = std::stoll(input, &used);
        if (used != input.size())
            throw std::invalid_argument("expected an integer Config value");
        return static_cast<std::int64_t>(result);
    }
    if (kind == trtmc::ConfigKind::F64) {
        const auto result = std::stod(input, &used);
        if (used != input.size() || !std::isfinite(result))
            throw std::invalid_argument("expected a finite numeric Config value");
        return result;
    }
    throw std::invalid_argument("unexpected Config kind");
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLE RUNTIME_ROOT text|tokens INPUT [KEY=VALUE ...]\n";
        return 2;
    }
    try {
        trtmc::TextContinuationRequest request;
        if (std::string_view(argv[3]) == "text") {
            request.prefix = std::string(argv[4]);
        } else if (std::string_view(argv[3]) == "tokens") {
            std::ifstream file(argv[4], std::ios::binary | std::ios::ate);
            const auto bytes = file.tellg();
            if (!file || bytes < 0 || bytes % sizeof(std::int32_t) != 0)
                throw std::invalid_argument("token input must contain int32 values");
            file.seekg(0);
            std::vector<std::int32_t> ids(static_cast<std::size_t>(bytes) / sizeof(std::int32_t));
            if (bytes > 0)
                file.read(reinterpret_cast<char*>(ids.data()), bytes);
            if (!file)
                throw std::runtime_error("could not read token input");
            request.prefix = std::move(ids);
        } else {
            throw std::invalid_argument("input must be text or tokens");
        }
        auto result = [&] {
            trtmc::LoadOptions options;
            options.runtime_root = argv[2];
            auto model = trtmc::Model::load(argv[1], options);
            auto task = model.task<trtmc::TextContinuation>();
            const auto fields = task.config_fields();
            trtmc::Config config;
            for (int i = 5; i < argc; ++i) {
                const std::string argument(argv[i]);
                const auto equal = argument.find('=');
                if (equal == std::string::npos)
                    throw std::invalid_argument("Config argument must be KEY=VALUE");
                const auto name = argument.substr(0, equal);
                const auto field =
                    std::find_if(fields.begin(), fields.end(),
                                 [&](const auto& item) { return item.name == name; });
                if (field == fields.end())
                    throw std::invalid_argument("unknown Config field: " + name);
                config.add(name, value(argument.substr(equal + 1), field->kind));
            }
            return task.run(request, config);
        }();
        request.prefix = std::string{};
        std::cout << "{\"text\":";
        json_string(result.text());
        std::cout << ",\"token_ids\":[";
        const auto ids = result.token_ids();
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
                  << "],\"setup_ms\":" << result.setup_ms()
                  << ",\"prefill_ms\":" << result.prefill_ms()
                  << ",\"decode_ms\":" << result.decode_ms() << "}\n";
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
