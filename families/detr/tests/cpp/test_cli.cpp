/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/bundle.h"
#include "trtmc/internal/cli.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
int failures = 0, task_loads = 0;
bool bad_output = false, loaded_graphs = false;
std::string loaded_runtime, loaded_cache;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
class FakeModel final : public trtmc::IObjectDetection {
  public:
    trtmc::ObjectDetectionResult detect(const float* pixels, std::int32_t height,
                                        std::int32_t width) override {
        check(height == 1 && width == 2, "owner passes image dimensions");
        check(pixels[0] == 1.0F && pixels[1] == 128.0F / 255.0F && pixels[2] == 0.0F,
              "owner passes normalized HWC RGB");
        return {{{0.0F, 0.0F, 2.0F, 1.0F,
                  bad_output ? std::numeric_limits<float>::infinity() : 0.9F, 4}},
                height,
                width};
    }
};
void write_bundle(const fs::path& path, const std::string& family) {
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", "object_detection"},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream out(path, std::ios::binary);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write("PLAN", 4);
}
struct Capture {
    int status;
    std::string output, error;
};
void output(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->output.append(data, size);
}
void error(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->error.append(data, size);
}
Capture invoke(const char* handler, const Json& values) {
    Capture result{};
    result.status = trtmc_family_cli_v1(handler, values.dump().c_str(), "/installed/runtime",
                                        &result, output, error);
    return result;
}
void contract(const fs::path& root) {
    const auto bundle = root / "model.bundle", image = root / "input.ppm";
    write_bundle(bundle, "detr");
    {
        std::ofstream out(image, std::ios::binary);
        out << "P6\n2 1\n255\n";
        const std::array<unsigned char, 6> rgb{255, 128, 0, 0, 255, 64};
        out.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    }
    Json values{{"bundle", bundle.string()}, {"image", image.string()}};
    const auto first = invoke("detect", values);
    check(first.status == 0 && first.error.empty(), "valid owner request succeeds");
    const auto output = Json::parse(first.output);
    check(output.at("boxes") == Json({0.0, 0.0, 2.0, 1.0}), "detection coordinates are preserved");
    check(output.at("scores").at(0).get<float>() == 0.9F && output.at("classes") == Json({4}),
          "detection scores and classes are preserved");
    check(output.at("image_height") == 1 && output.at("image_width") == 2,
          "image dimensions are preserved");

    check(loaded_runtime == "/installed/runtime" && loaded_cache.empty() && !loaded_graphs,
          "default load options are preserved");
    values.update({{"runtime_root", "/override/runtime"},
                   {"runtime_cache", "runtime.cache"},
                   {"cuda_graphs", true}});
    check(invoke("detect", values).status == 0, "explicit runtime options succeed");
    check(loaded_runtime == "/override/runtime" && loaded_cache == "runtime.cache" && loaded_graphs,
          "runtime options reach the loader");
    bad_output = true;
    const auto invalid_output = invoke("detect", values);
    check(invalid_output.status != 0 && invalid_output.output.empty(),
          "non-finite runtime results fail closed");
    bad_output = false;
    const auto before = task_loads;
    values["image"] = (root / "missing.png").string();
    check(invoke("detect", values).status != 0 && task_loads == before,
          "invalid image fails before model loading");
    values["image"] = image.string();
    write_bundle(bundle, "another_family");
    check(invoke("detect", values).status != 0 && task_loads == before,
          "wrong-family bundle fails before model loading");
    check(invoke("unknown", values).status != 0 && task_loads == before,
          "unknown handler never loads a task");
}
} // namespace

namespace trtmc {
std::unique_ptr<ITask> load_task(const BundleReader&, const std::string& runtime_root,
                                 std::uint64_t, const std::string& runtime_cache,
                                 bool cuda_graphs) {
    ++task_loads;
    loaded_runtime = runtime_root;
    loaded_cache = runtime_cache;
    loaded_graphs = cuda_graphs;
    return std::make_unique<FakeModel>();
}
} // namespace trtmc
int main() {
    const auto root = fs::temp_directory_path() / ("trtmc-detr-cli-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        contract(root);
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
