/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/sdk_dispatch.h"
#include "trtmc/action.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path) {
    const auto header = json{{"format", 1},
                             {"family", "action_fixture"},
                             {"task", "image_state_to_action_chunk"},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), header.size());
    out.write("PLAN", 4);
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& arg : arguments)
        argv.push_back(arg.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
void exercise(const std::filesystem::path& root) {
    const auto bundle_path = root / "action_cli.bundle";
    const auto image = root / "action_cli.ppm", state = root / "action_cli_state.f32",
               saved = root / "action_cli_output.f32";
    bundle(bundle_path);
    {
        std::ofstream out(image, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << "P6\n1 1\n255\n";
        const unsigned char pixel[] = {255, 128, 64};
        out.write(reinterpret_cast<const char*>(pixel), sizeof(pixel));
    }
    {
        std::ofstream out(state, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        const float values[] = {2, 4};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    const std::vector<std::string> base = {"trtmc",          "control",     bundle_path.string(),
                                           "--runtime-root", root.string(), "--image",
                                           image.string(),   "--state",     state.string()};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto arguments = base;
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return run(std::move(arguments));
    };
    const auto defaults = invoke({});
    check(defaults.status == 0, "control works through the stateless public action SDK");
    if (defaults.status != 0)
        throw std::runtime_error(defaults.error);
    auto value = json::parse(defaults.output);
    check(value.at("actions") == json::array({3, -3, 4, 8}) && value.at("num_actions") == 2 &&
              value.at("action_dim") == 2 && value.at("within_training_bounds") == false &&
              value.at("inference_ms") == 101,
          "existing control JSON fields preserve raw action values, order, bounds and timing");
    check(value.at("schema").at("domain") == "fixture.default" &&
              value.at("schema").at("component_names") == json::array({"left", "right"}) &&
              value.at("schema").at("units").empty() &&
              value.at("schema").at("normalization") == "unnormalized" &&
              value.at("timestamps_seconds").empty(),
          "additional action schema does not invent units or a frequency");
    const auto output = invoke({"--output", saved.string(), "--set", "tag=custom"});
    check(output.status == 0, "control reuses existing binary output and family config");
    if (output.status == 0) {
        auto custom = json::parse(output.output);
        check(custom.at("schema").at("domain") == "fixture.custom",
              "family config reaches the selected Task");
        std::ifstream input(saved, std::ios::binary | std::ios::ate);
        check(input && input.tellg() == static_cast<std::streamoff>(4 * sizeof(float)),
              "raw output retains old contiguous float32 layout");
        input.seekg(0);
        float values[4] = {};
        input.read(reinterpret_cast<char*>(values), sizeof(values));
        check(input && values[0] == 3 && values[1] == -3 && values[2] == 4 && values[3] == 8,
              "binary action file contains original values without clipping or renormalization");
    }
    for (const auto& extra :
         std::vector<std::vector<std::string>>{{"--set", "unknown=1"},
                                               {"--set", "tag=first", "--set", "tag=second"},
                                               {"--set", "tag=1", "--set", "tag=2"},
                                               {"--task", "image_state_action_queue"},
                                               {"--task", "text_continuation"}}) {
        auto arguments = base;
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        const auto rejected = run(std::move(arguments));
        check(rejected.status != 0 && rejected.output.empty(),
              "invalid or unsupported control requests do not produce actions or fallback");
    }
    const auto empty = invoke({"--set", "tag="});
    check(empty.status == 0 && json::parse(empty.output).at("schema").at("domain") == "fixture.",
          "explicit empty family string remains distinct from a missing default");
    const auto queue = invoke({"--task", "image_state_action_queue"});
    check(queue.status != 0 && queue.error.find("persistent action queue") != std::string::npos,
          "one-shot CLI does not pretend to provide a persistent queue");
    auto invalid = base;
    invalid.back() = (root / "absent-state.f32").string();
    const auto missing = run(std::move(invalid));
    check(missing.status != 0 && missing.output.empty(),
          "unreadable state never reaches model execution");
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    auto model = trtmc::Model::load(bundle_path.string(), options);
    trtmc::cli::Command unhandled;
    std::ostringstream untouched;
    check(!trtmc::cli::dispatch_sdk_action(unhandled, model, "unrelated_task", untouched) &&
              untouched.str().empty(),
          "unmatched command group has no input/output side effects");
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return failures ? 1 : 0;
}
