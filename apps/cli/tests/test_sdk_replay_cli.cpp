/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, std::string task) {
    const auto header = json{
        {"format", 1},
        {"family", "numeric_fixture"},
        {"task", std::move(task)},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
void raw(const std::filesystem::path& path, std::initializer_list<float> values) {
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write(reinterpret_cast<const char*>(values.begin()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
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
    const auto all = root / "replay_cli.bundle", restricted = root / "replay_cli_restricted.bundle",
               initial = root / "replay_cli_initial.f32",
               condition = root / "replay_cli_condition.f32", mask = root / "replay_cli_mask.f32",
               noise = root / "replay_cli_noise.f32", schedule = root / "replay_cli_schedule.f32",
               empty = root / "replay_cli_empty.f32", wrong = root / "replay_cli_wrong.f32";
    bundle(all, "latent_replay_to_text");
    bundle(restricted, "regression_only");
    raw(initial, {9, 8, 7, 6});
    raw(condition, {1, 2, 3, 4});
    raw(mask, {1, 0.5F});
    raw(noise, {10, 11, 12, 13, 14, 15, 16, 17});
    raw(schedule, {0, 0.25F, 0.75F, 1});
    raw(empty, {});
    raw(wrong, {1, 2, 3});
    const std::vector<std::string> base{"trtmc", "run", all.string(), "--runtime-root",
                                        root.string()};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        return run(std::move(args));
    };

    const auto initial_only = invoke({"--initial-latents-raw", initial.string()});
    check(initial_only.status == 0,
          "existing initial-latent command needs neither task selection nor shape flags");
    if (initial_only.status != 0)
        throw std::runtime_error(initial_only.error);
    const auto initial_value = json::parse(initial_only.output);
    check(initial_value.at("text") == "|replayed" &&
              initial_value.at("token_ids") == json::array({9, -1, 50}),
          "initial replay retains family noise and schedule defaults");

    const auto noise_only =
        invoke({"--sde-noise-raw", noise.string(), "--sampling-steps-raw", schedule.string()});
    check(noise_only.status == 0, "promptless SDE-only replay selects its typed contract");
    if (noise_only.status == 0)
        check(json::parse(noise_only.output).at("token_ids") == json::array({-1, 17, 25}),
              "SDE-only raw input uses family initial-state policy and typed sampling schedule");
    const auto both =
        invoke({"--prompt", "actual prompt", "--initial-latents-raw", initial.string(),
                "--sde-noise-raw", noise.string(), "--sampling-steps-raw", schedule.string()});
    check(both.status == 0, "initial-plus-noise replay preserves old raw command spelling");
    if (both.status == 0) {
        const auto value = json::parse(both.output);
        check(value.at("text") == "actual prompt|replayed" &&
                  value.at("token_ids") == json::array({9, 17, 25}),
              "prompt and both replay inputs cross the public ABI unchanged");
    }
    const auto conditioned =
        invoke({"--prompt", "preserved", "--condition-latents-raw", condition.string(),
                "--condition-mask-raw", mask.string(), "--sde-noise-raw", noise.string(),
                "--sampling-steps-raw", schedule.string()});
    check(conditioned.status == 0,
          "paired raw conditioning selects the conditioned Task without an extra flag");
    if (conditioned.status == 0) {
        const auto value = json::parse(conditioned.output);
        check(value.at("text") == "conditioned" &&
                  value.at("token_ids") == json::array({1, 50, -1, 17, 25, 9}),
              "raw condition/mask, prompt and SDE reach family with its condition precedence");
    }
    const auto explicit_task = invoke({"--task", "latent_replay_to_text", "--prompt", "",
                                       "--initial-latents-raw", initial.string()});
    check(explicit_task.status == 0, "an explicit matching Task remains usable");

    const std::vector<std::vector<std::string>> invalid = {
        {"--task", "latent_replay_to_text"},
        {"--condition-latents-raw", condition.string()},
        {"--condition-mask-raw", mask.string()},
        {"--task", "text_continuation", "--initial-latents-raw", initial.string()},
        {"--task", "latent_replay_to_text", "--condition-latents-raw", condition.string(),
         "--condition-mask-raw", mask.string()},
        {"--task", "latent_conditioned_text_generation", "--initial-latents-raw", initial.string()},
        {"--initial-latents-raw", empty.string()},
        {"--initial-latents-raw", wrong.string()},
        {"--initial-latents-raw", initial.string(), "--sampling-steps-raw", schedule.string(),
         "--set", "sampling_steps=[0,0.5,1]"},
        {"--initial-latents-raw", initial.string(), "--set", "unknown=1"},
        {"--sde-noise-raw", noise.string()},
    };
    for (const auto& extra : invalid) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        const auto result = run(std::move(args));
        check(result.status != 0 && result.output.empty(),
              "invalid replay presence/type/layout/schedule/duplicates fail without output");
    }
    const auto unsupported = run({"trtmc", "run", restricted.string(), "--runtime-root",
                                  root.string(), "--initial-latents-raw", initial.string()});
    check(unsupported.status != 0 && unsupported.output.empty() &&
              unsupported.error.find("does not support") != std::string::npos,
          "operand-selected missing Task fails explicitly without old-method retry");
    for (const auto& path :
         {all, restricted, initial, condition, mask, noise, schedule, empty, wrong})
        std::filesystem::remove(path);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        ++failures;
    }
    if (!failures)
        std::cout << "ALL PASSED\n";
    return failures ? 1 : 0;
}
