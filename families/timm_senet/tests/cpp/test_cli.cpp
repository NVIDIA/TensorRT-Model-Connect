/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/bundle.h"
#include "trtmc/internal/cli.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#ifdef TRTMC_FAMILY_CLI_FIXTURE
#include "trtmc/runtime/family_factory.h"
#include "trtmc/task.h"
namespace {
class Fixture final : public trtmc::IImageClassification {
  public:
    const char* task() const noexcept override { return "classification"; }
    trtmc::ClassificationResult classify(const float* pixels, std::int32_t height,
                                         std::int32_t width) override {
        if (width != 2 || height != 1 || pixels[0] != 1.0F || pixels[1] != 0.0F ||
            pixels[2] != 128.0F / 255.0F || pixels[4] != 1.0F)
            throw std::invalid_argument("owner changed decoded image shape, order or range");
        return {{-2.0F, 4.0F, 0.5F}, 1, 4.0F};
    }
};
} // namespace
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext&) {
    return new Fixture();
}
#else
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const fs::path& path, const std::string& family) {
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", "classification"},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream output(path, std::ios::binary);
    output.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
}
struct Capture {
    std::string output, error;
};
void emit(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->output.append(data, size);
}
void reject(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->error.append(data, size);
}
void contract(const fs::path& runtime_root, const fs::path& root) {
    const auto path = root / "model.bundle";
    const auto image = root / "image.ppm";
    bundle(path, "timm_senet");
    {
        std::ofstream output(image, std::ios::binary);
        output << "P6\n2 1\n255\n";
        const unsigned char pixels[] = {255, 0, 128, 0, 255, 0};
        output.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
    }
    auto invoke = [&](Json values, const char* handler = "classify",
                      const std::string& fallback = "") {
        Capture captured;
        const auto root_value = fallback.empty() ? runtime_root.string() : fallback;
        const auto status = trtmc_family_cli_v1(handler, values.dump().c_str(), root_value.c_str(),
                                                &captured, emit, reject);
        return std::pair{status, captured};
    };
    Json values{{"bundle", path.string()}, {"image", image.string()}};
    const auto result = invoke(values);
    check(result.first == 0, "owner classification succeeds through native callback and runtime");
    if (result.first == 0) {
        const auto actual = Json::parse(result.second.output);
        check(actual.at("logits") == Json({-2.0, 4.0, 0.5}) && actual.at("top_class") == 1 &&
                  actual.at("top_score") == 4.0,
              "classification preserves logits and argmax without softmax");
        check(actual.size() == 3, "legacy classification output shape remains unchanged");
    }
    auto explicit_root = values;
    explicit_root["runtime_root"] = runtime_root.string();
    check(invoke(explicit_root, "classify", (root / "missing").string()).first == 0,
          "explicit runtime root overrides the installed default");
    explicit_root["runtime_root"] = (root / "missing").string();
    check(invoke(explicit_root).first != 0, "invalid explicit root never retries the default");
    auto rtx = values;
    rtx["cuda_graphs"] = true;
    check(invoke(rtx).first != 0,
          "RTX graph flag reaches loader validation instead of being ignored");
    rtx = values;
    rtx["runtime_cache"] = (root / "cache").string();
    check(invoke(rtx).first != 0,
          "RTX cache flag reaches loader validation instead of being ignored");
    check(invoke(values, "unknown").first != 0, "unknown owner handler is rejected");
    bundle(path, "another_family");
    check(invoke(values).first != 0, "wrong-family bundle is rejected");
    bundle(path, "timm_senet");
    std::ofstream(image) << "invalid image";
    const auto invalid_image = invoke(values);
    check(invalid_image.first != 0 &&
              invalid_image.second.error.find("decode") != std::string::npos,
          "invalid image fails in owner decoding before task execution");
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const auto root = fs::temp_directory_path() / ("timm_senet-cli-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        contract(argv[1], root);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures ? 1 : 0;
}
#endif
