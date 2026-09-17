/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/io.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, const std::string& task) {
    const auto header = json{
        {"format", 1},
        {"family", "image_fixture"},
        {"task", task},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((std::uint64_t(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    std::ostringstream out, err;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), out, err);
    return {status, out.str(), err.str()};
}
bool near(float value, float expected) {
    return std::abs(value - expected) < 0.005F;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]);
    const auto model = root / "image-cli.bundle";
    const auto output = root / "image-cli-output.png";
    const auto first = root / "image-cli-first.png", second = root / "image-cli-second.png";
    const auto latent = root / "image-cli-latents.f32", mask = root / "image-cli-mask.f32";
    const auto prompts = root / "image-cli-prompts.txt", batch_dir = root / "image-cli-batch";
    try {
        bundle(model, "text_to_image");
        trtmc::cli::io::save_png(first.string(), std::vector<float>{0.2F, 0, 0}, 1, 1);
        trtmc::cli::io::save_png(second.string(), std::vector<float>{0.8F, 0, 0}, 1, 1);
        auto invoke = [&](std::initializer_list<std::string> extra) {
            std::vector<std::string> args{
                "trtmc",        "generate-image", model.string(), "--runtime-root",
                root.string(),  "--prompt",       "red",          "--output",
                output.string()};
            args.insert(args.end(), extra.begin(), extra.end());
            return run(std::move(args));
        };
        auto generated = invoke({});
        check(generated.status == 0, "minimal image command uses family defaults and public SDK");
        if (generated.status != 0)
            throw std::runtime_error(generated.error);
        const auto metadata = json::parse(generated.output);
        check(metadata.at("height") == 1 && metadata.at("width") == 1 &&
                  metadata.at("output") == output.string(),
              "existing image JSON fields retained");
        check(near(trtmc::cli::io::read_image(output.string()).pixels.at(0), 0.25F),
              "actual PNG contains the family output");
        {
            const float values[]{0.25F, 0, -0.125F};
            std::ofstream file(latent, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        check(invoke({"--initial-latents-raw", latent.string()}).status == 0,
              "initial latents use the existing raw file without shape flags");
        check(near(trtmc::cli::io::read_image(output.string()).pixels.at(0), 0.5F),
              "typed float32 replay reaches image family");
        check(invoke({"--image", first.string()}).status == 0,
              "source image selects editing before execution");
        check(near(trtmc::cli::io::read_image(output.string()).pixels.at(1), 0.2F),
              "source image is not silently discarded");
        check(invoke({"--images", json::array({first.string(), second.string()}).dump()}).status ==
                  0,
              "ordered multi-image editing uses the same typed contract");
        const auto edited = trtmc::cli::io::read_image(output.string());
        check(near(edited.pixels.at(1), 0.2F) && near(edited.pixels.at(2), 0.8F),
              "reference image order is preserved");
        {
            const float value = 0;
            std::ofstream file(mask, std::ios::binary);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        check(invoke({"--image", first.string(), "--mask", mask.string()}).status == 0,
              "typed HW mask selects masked image generation");
        check(near(trtmc::cli::io::read_image(output.string()).pixels.at(0), 0.2F),
              "zero mask preserves source content");
        for (const auto& extra : std::vector<std::vector<std::string>>{
                 {"--set", "level=2.0"},
                 {"--set", "missing=1"},
                 {"--image", first.string(), "--images", "[]"},
                 {"--task", "text_to_image", "--image", first.string()},
                 {"--mask", mask.string()},
                 {"--images", "[]"}}) {
            std::vector<std::string> args{
                "trtmc",        "generate-image", model.string(), "--runtime-root",
                root.string(),  "--prompt",       "red",          "--output",
                output.string()};
            args.insert(args.end(), extra.begin(), extra.end());
            const auto result = run(std::move(args));
            check(result.status != 0 && result.output.empty(),
                  "invalid image input/config is not ignored");
        }
        {
            std::ofstream file(prompts);
            file << "first\n\nthird\n";
        }
        std::vector<std::string> batch_args{
            "trtmc",     "generate-image-batch", model.string(), "--runtime-root",  root.string(),
            "--prompts", prompts.string(),       "--output",     batch_dir.string()};
        auto batch = run(batch_args);
        check(batch.status == 0 && json::parse(batch.output).at("outputs").size() == 3,
              "batch retains empty prompts and uses family defaults without required seeds");
        batch_args.insert(batch_args.end(), {"--seeds", "0,64,255"});
        batch = run(batch_args);
        check(batch.status == 0, "per-item seeds reach the declared batch Config");
        check(near(trtmc::cli::io::read_image((batch_dir / "0.png").string()).pixels.at(2), 0) &&
                  near(trtmc::cli::io::read_image((batch_dir / "2.png").string()).pixels.at(2), 1),
              "batch seeds are neither merged nor replaced by one shared value");
        bundle(model, "worker");
        std::filesystem::remove(output);
        const auto worker = invoke({});
        check(worker.status == 0 && json::parse(worker.output).at("worker") == true &&
                  !std::filesystem::exists(output),
              "non-output participant returns worker without writing a PNG");
        bundle(model, "bad_empty");
        const auto malformed = invoke({});
        check(malformed.status != 0 && malformed.output.empty(),
              "invalid empty output is not a worker");
    } catch (const std::exception& error) {
        std::cerr << "Unexpected: " << error.what() << '\n';
        ++failures;
    }
    for (const auto& path : {model, output, first, second, latent, mask, prompts})
        std::filesystem::remove(path);
    for (const auto* name : {"0.png", "1.png", "2.png"})
        std::filesystem::remove(batch_dir / name);
    std::filesystem::remove(batch_dir);
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
