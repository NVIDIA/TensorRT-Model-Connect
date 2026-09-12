/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path) {
    const auto header = json{
        {"format", 1},
        {"family", "language_fixture"},
        {"task", "images_text_to_text"},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::ofstream out(path, std::ios::binary);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((std::uint64_t(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!out)
        throw std::runtime_error("failed to write language fixture bundle");
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]);
    const auto model = root / "language-cli.bundle", image = root / "language-cli-image.png";
    try {
        bundle(model);
        trtmc::cli::io::save_png(image.string(), std::vector<float>{0.25F, 0, 0}, 1, 1);
        const std::vector<std::string> base{
            "trtmc", "run", model.string(), "--runtime-root", root.string(), "--prompt", "Hello"};
        auto invoke = [&](std::initializer_list<std::string> extra) {
            auto args = base;
            args.insert(args.end(), extra.begin(), extra.end());
            return run(std::move(args));
        };
        const auto response = invoke({"--image", image.string()});
        check(response.status == 0, "existing run --image call reaches the VLM Task SDK");
        if (response.status != 0)
            throw std::runtime_error(response.error);
        const auto value = json::parse(response.output);
        const auto text = value.at("text").get<std::string>();
        check(text.find("Hello") != std::string::npos && text.find("I:") != std::string::npos,
              "image data and text arrive as typed parts, not a serialized placeholder");
        check(value.contains("token_ids") && value.contains("setup_ms") &&
                  value.contains("decode_ms"),
              "text/tokens/timings retain the existing CLI output shape");
        const auto two = invoke({"--images", json::array({image.string(), image.string()}).dump()});
        check(two.status == 0, "ordered multi-image input uses the same semantic Task");
        const auto unknown = invoke({"--image", image.string(), "--set", "unknown=1"});
        check(unknown.status != 0 && unknown.output.empty(), "VLM config is not silently ignored");
        const auto missing = invoke({});
        check(missing.status != 0 && missing.output.empty(),
              "media-required bundle does not fabricate an image");
        const auto wrong = invoke({"--image", image.string(), "--task", "text_continuation"});
        check(wrong.status != 0 && wrong.output.empty(),
              "explicit unsupported Task does not retry the VLM");
    } catch (const std::exception& error) {
        std::cerr << "Unexpected: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(model);
    std::filesystem::remove(image);
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
